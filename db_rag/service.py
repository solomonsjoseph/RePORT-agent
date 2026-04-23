from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from utils.llm_response import coerce_text_content


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "local_data" / "db_rag_source"
SCHEMA_DIR = SOURCE_ROOT / "reviewed_annotated_json_files"
EXCEL_DIR = SOURCE_ROOT / "filtered_excel_files"
RUNTIME_ROOT = PROJECT_ROOT / "runtime" / "db_rag"
CHROMA_DIR = RUNTIME_ROOT / "chroma_db"
DUCKDB_PATH = RUNTIME_ROOT / "report.duckdb"
MANIFEST_PATH = RUNTIME_ROOT / "manifest.json"
EMBEDDING_MODEL = "text-embedding-3-small"

_DANGEROUS_SQL = ("DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "ATTACH", "DETACH", "CREATE", "REPLACE")


@dataclass
class DbRagTableHit:
    table: str
    text: str


@dataclass
class DbRagColumnHit:
    table: str
    column: str
    text: str

    def as_prompt_line(self) -> str:
        return f"{self.table}.{self.column}"


@dataclass
class DbRagContext:
    tables: list[DbRagTableHit] = field(default_factory=list)
    columns: list[DbRagColumnHit] = field(default_factory=list)
    table_context: str = ""
    column_context: str = ""

    @property
    def table_names(self) -> list[str]:
        return [entry.table for entry in self.tables]

    @property
    def column_names(self) -> list[str]:
        return [entry.column for entry in self.columns]


def _extract_sql(text: str) -> str:
    text = str(text or "").strip()
    match = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    for prefix in ("SELECT", "WITH"):
        idx = text.upper().find(prefix)
        if idx >= 0:
            return text[idx:].strip()
    return text


def _validate_sql(sql: str) -> tuple[bool, str | None]:
    sql_upper = str(sql or "").strip().upper()
    if not sql_upper.startswith(("SELECT", "WITH")):
        return False, "Only read-only SELECT/WITH SQL is allowed."
    if any(keyword in sql_upper for keyword in _DANGEROUS_SQL):
        return False, "SQL contains disallowed mutating or DDL keywords."

    try:
        import sqlglot

        sqlglot.parse_one(sql, dialect="duckdb")
    except ModuleNotFoundError:
        return True, None
    except Exception as exc:  # pragma: no cover - defensive validation
        return False, str(exc)
    return True, None


class OpenAIEmbeddingFunction:
    def __init__(self, model: str = EMBEDDING_MODEL):
        from openai import OpenAI

        load_dotenv()
        self.client = OpenAI()
        self.model = model

    @staticmethod
    def name() -> str:
        return "openai"

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self.__call__(input)

    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(input), 100):
            batch = input[start : start + 100]
            response = self.client.embeddings.create(model=self.model, input=batch)
            embeddings.extend(item.embedding for item in response.data)
        return embeddings


@dataclass
class DbRagService:
    llm: Any

    def readiness(self) -> dict[str, Any]:
        missing_runtime = [path for path in (DUCKDB_PATH, MANIFEST_PATH) if not path.exists()]
        if missing_runtime or not CHROMA_DIR.exists():
            source_hint = ""
            if not SCHEMA_DIR.exists() or not EXCEL_DIR.exists():
                source_hint = (
                    " Raw source folders are missing under local_data/db_rag_source/ "
                    "(expected reviewed_annotated_json_files/ and filtered_excel_files/)."
                )
            return {
                "ready": False,
                "message": (
                    "DB-RAG assets are not initialized in this repo."
                    f"{source_hint} Copy the source data into local_data/db_rag_source/ and run "
                    "`python -m db_rag.bootstrap --rebuild`."
                ),
            }

        try:
            import chromadb  # noqa: F401
            import duckdb  # noqa: F401
        except ModuleNotFoundError as exc:
            return {
                "ready": False,
                "message": (
                    f"DB-RAG dependency '{exc.name}' is not installed. Install the project requirements and run "
                    "`python -m db_rag.bootstrap --rebuild`."
                ),
            }

        return {"ready": True, "message": ""}

    def _load_collections(self):
        import chromadb

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        ef = OpenAIEmbeddingFunction()
        table_collection = client.get_collection("table_summaries", embedding_function=ef)
        column_collection = client.get_collection("column_chunks", embedding_function=ef)
        return table_collection, column_collection

    def _retrieve_context(self, question: str) -> dict[str, Any]:
        table_collection, column_collection = self._load_collections()

        table_result = table_collection.query(
            query_texts=[question],
            n_results=4,
            include=["documents", "metadatas"],
        )
        tables = []
        for document, metadata in zip(table_result["documents"][0], table_result["metadatas"][0]):
            tables.append({"table": metadata["table"], "text": document})

        selected_tables = [entry["table"] for entry in tables]
        column_result = column_collection.query(
            query_texts=[question],
            n_results=12,
            include=["documents", "metadatas"],
        )
        columns = []
        for document, metadata in zip(column_result["documents"][0], column_result["metadatas"][0]):
            if metadata["table"] not in selected_tables:
                continue
            columns.append(
                {
                    "table": metadata["table"],
                    "column": metadata["column"],
                    "text": document,
                }
            )

        return {
            "tables": tables,
            "columns": columns,
            "table_context": "\n\n".join(entry["text"] for entry in tables),
            "column_context": "\n\n".join(entry["text"] for entry in columns),
        }

    def answer_question(self, question: str) -> dict[str, Any]:
        context = self._retrieve_context(question)
        response = self.llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are answering a question about the RePORT clinical database using only retrieved "
                        "database metadata and schema context. Answer directly from the retrieved context. "
                        "If the user's question would require row-level computation or cohort extraction, say "
                        "which tables/columns appear relevant and state that a read-only SQL extraction would be "
                        "needed for an exact numeric answer."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question:\n{question}\n\n"
                        f"Table context:\n{context['table_context'] or 'none'}\n\n"
                        f"Column context:\n{context['column_context'] or 'none'}"
                    )
                ),
            ]
        )
        return {
            "answer": coerce_text_content(getattr(response, "content", "")),
            "retrieval_summary": {
                "tables": [entry["table"] for entry in context["tables"]],
                "columns": [entry["column"] for entry in context["columns"]],
            },
        }

    def _generate_sql(self, question: str, context: dict[str, Any]) -> str:
        response = self.llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a DuckDB SQL expert for the RePORT clinical research database. "
                        "Use only retrieved tables and columns. Return only read-only DuckDB SQL using "
                        "SELECT or WITH. Never emit mutating or DDL statements."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question:\n{question}\n\n"
                        f"Table context:\n{context['table_context'] or 'none'}\n\n"
                        f"Column context:\n{context['column_context'] or 'none'}"
                    )
                ),
            ]
        )
        sql = _extract_sql(coerce_text_content(getattr(response, "content", "")))
        valid, error = _validate_sql(sql)
        if not valid:
            raise ValueError(error or "SQL validation failed.")
        return sql

    def execute_sql_flow(self, question: str) -> dict[str, Any]:
        import duckdb

        context = self._retrieve_context(question)
        sql = self._generate_sql(question, context)
        db = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        dataframe = db.execute(sql).fetchdf()
        answer = f"Read-only SQL execution completed with {len(dataframe)} result row(s)."
        return {
            "answer": answer,
            "sql": sql,
            "dataframe": dataframe,
            "source_tables": [entry["table"] for entry in context["tables"]],
        }
