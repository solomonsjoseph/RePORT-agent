from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
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
INDEX_ROOT = RUNTIME_ROOT / "indexes"
MANIFEST_ROOT = RUNTIME_ROOT / "manifests"
CHROMA_DIR = RUNTIME_ROOT / "chroma_db"
DUCKDB_PATH = RUNTIME_ROOT / "report.duckdb"
MANIFEST_PATH = RUNTIME_ROOT / "manifest.json"
EMBEDDING_MODEL = "OpenAI/text-embedding-3-small"
SUPPORTED_DB_RAG_EMBEDDING_MODELS = (
    "OpenAI/text-embedding-3-small",
    "Qwen/Qwen3-Embedding-4B",
    "Qwen/Qwen3-Embedding-8B",
)
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_DB_RAG_CONTEXT_FALLBACK_ANSWER = "I could not answer from the retrieved DB-RAG context."
_DB_RAG_CONTEXT_FALLBACK_RATIONALE = "Invalid structured response from the model."

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


@dataclass
class DbRagQaAnswer:
    answer: str
    needs_sql: bool
    rationale: str
    relevant_tables: list[str]
    relevant_columns: list[str]


@dataclass
class ColumnSelectionCandidate:
    selection_id: str
    question: str
    tables: list[str]
    columns: list[dict[str, str]]
    rationale: str
    feedback_history: list[dict[str, Any]] = field(default_factory=list)
    status: str = "awaiting_review"


@dataclass
class PreparedSqlCandidate:
    question: str
    sql: str
    tables: list[str]
    columns: list[dict[str, str]]
    selection_id: str
    status: str = "prepared"


@dataclass
class SqlExecutionResult:
    answer: str
    sql: str
    dataframe: Any
    source_tables: list[str]


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


def _parse_json_object(text: str) -> dict[str, Any]:
    raw_text = str(text or "").strip()
    if not raw_text:
        return {}

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    return parsed if isinstance(parsed, dict) else {}


def _default_selection_id(question: str, context: DbRagContext, feedback_history: list[dict[str, Any]]) -> str:
    columns = [{"table": entry.table, "column": entry.column} for entry in context.columns]
    payload = {
        "question": question,
        "tables": context.table_names,
        "columns": columns,
        "feedback_history": feedback_history,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return f"sel-{digest[:12]}"


def _build_invalid_column_selection_candidate(
    question: str,
    context: DbRagContext,
    feedback_history: list[dict[str, Any]],
) -> ColumnSelectionCandidate:
    return ColumnSelectionCandidate(
        selection_id=_default_selection_id(question, context, feedback_history),
        question=question,
        tables=[],
        columns=[],
        rationale=_DB_RAG_CONTEXT_FALLBACK_RATIONALE,
        feedback_history=feedback_history,
    )


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


def embedding_model_slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")


def resolve_db_rag_embedding_model() -> str:
    load_dotenv()
    model = str(os.getenv("DB_RAG_EMBEDDING_MODEL", "") or "").strip()
    if not model:
        raise ValueError("DB_RAG_EMBEDDING_MODEL is not set. Set it in .env or export it in your shell.")
    if model not in SUPPORTED_DB_RAG_EMBEDDING_MODELS:
        supported = ", ".join(SUPPORTED_DB_RAG_EMBEDDING_MODELS)
        raise ValueError(f"Unsupported DB_RAG_EMBEDDING_MODEL '{model}'. Supported values: {supported}.")
    return model


def chroma_dir_for_model(model: str) -> Path:
    return INDEX_ROOT / embedding_model_slug(model)


def manifest_path_for_model(model: str) -> Path:
    return MANIFEST_ROOT / f"{embedding_model_slug(model)}.json"


def load_manifest_for_model(model: str) -> dict[str, Any]:
    path = manifest_path_for_model(model)
    return json.loads(path.read_text(encoding="utf-8"))


class OpenAIEmbeddingFunction:
    def __init__(self, model: str | None = None):
        from openai import OpenAI

        load_dotenv()
        resolved_model = model or resolve_db_rag_embedding_model()
        client_kwargs: dict[str, str] = {}
        if resolved_model.startswith("Qwen/"):
            api_key = str(os.getenv("DB_RAG_OPENROUTER_API_KEY", "") or "").strip()
            if not api_key:
                raise ValueError("DB_RAG_OPENROUTER_API_KEY is required for Qwen embeddings.")
            client_kwargs["api_key"] = api_key
            client_kwargs["base_url"] = str(
                os.getenv("DB_RAG_OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL) or DEFAULT_OPENROUTER_BASE_URL
            ).strip()
        self.client = OpenAI(**client_kwargs)
        self.model = resolved_model

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
        try:
            model = resolve_db_rag_embedding_model()
        except ValueError as exc:
            return {"ready": False, "message": str(exc)}

        chroma_dir = chroma_dir_for_model(model)
        manifest_path = manifest_path_for_model(model)
        missing_runtime = [path for path in (DUCKDB_PATH, manifest_path) if not path.exists()]
        if missing_runtime or not chroma_dir.exists():
            source_hint = ""
            if not SCHEMA_DIR.exists() or not EXCEL_DIR.exists():
                source_hint = (
                    " Raw source folders are missing under local_data/db_rag_source/ "
                    "(expected reviewed_annotated_json_files/ and filtered_excel_files/)."
                )
            return {
                "ready": False,
                "message": (
                    f"DB-RAG assets are not initialized for embedding model '{model}'."
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

        model = resolve_db_rag_embedding_model()
        load_manifest_for_model(model)
        client = chromadb.PersistentClient(path=str(chroma_dir_for_model(model)))
        ef = OpenAIEmbeddingFunction(model=model)
        table_collection = client.get_collection("table_summaries", embedding_function=ef)
        column_collection = client.get_collection("column_chunks", embedding_function=ef)
        return table_collection, column_collection

    def retrieve_context(self, question: str) -> DbRagContext:
        table_collection, column_collection = self._load_collections()

        table_result = table_collection.query(
            query_texts=[question],
            n_results=4,
            include=["documents", "metadatas"],
        )
        tables: list[DbRagTableHit] = []
        for document, metadata in zip(table_result["documents"][0], table_result["metadatas"][0]):
            tables.append(DbRagTableHit(table=metadata["table"], text=document))

        selected_tables = {entry.table for entry in tables}
        column_result = column_collection.query(
            query_texts=[question],
            n_results=12,
            include=["documents", "metadatas"],
        )
        columns: list[DbRagColumnHit] = []
        for document, metadata in zip(column_result["documents"][0], column_result["metadatas"][0]):
            if metadata["table"] not in selected_tables:
                continue
            columns.append(
                DbRagColumnHit(
                    table=metadata["table"],
                    column=metadata["column"],
                    text=document,
                )
            )

        return DbRagContext(
            tables=tables,
            columns=columns,
            table_context="\n\n".join(entry.text for entry in tables),
            column_context="\n\n".join(entry.text for entry in columns),
        )

    def answer_question(self, question: str) -> dict[str, Any]:
        context = self.retrieve_context(question)
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
                        f"Table context:\n{context.table_context or 'none'}\n\n"
                        f"Column context:\n{context.column_context or 'none'}"
                    )
                ),
            ]
        )
        return {
            "answer": coerce_text_content(getattr(response, "content", "")),
            "retrieval_summary": {
                "tables": context.table_names,
                "columns": context.column_names,
            },
        }

    def answer_from_context(self, question: str, context: DbRagContext) -> DbRagQaAnswer:
        response = self.llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are classifying whether a RePORT database question can be answered from retrieved "
                        "metadata and schema context alone.\n"
                        "Use the retrieved table and column descriptions to answer overview, schema, variable, and "
                        "other metadata questions directly.\n"
                        "Mark needs_sql true when the user asks for row-level records, subsets, counts, filters, "
                        "aggregations, or any other request that depends on actual data values rather than schema.\n"
                        "Return only a JSON object with exactly these keys: "
                        '{"answer": string, "needs_sql": boolean, "rationale": string}.\n'
                        "If the request can be answered from metadata/schema context, set needs_sql to false.\n"
                        "If the request needs SQL for an exact answer, set needs_sql to true and explain why in rationale."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question:\n{question}\n\n"
                        f"Table context:\n{context.table_context or 'none'}\n\n"
                        f"Column context:\n{context.column_context or 'none'}"
                    )
                ),
            ]
        )
        parsed = _parse_json_object(coerce_text_content(getattr(response, "content", "")))
        needs_sql_value = parsed.get("needs_sql")
        if not parsed or not isinstance(needs_sql_value, bool):
            return DbRagQaAnswer(
                answer=_DB_RAG_CONTEXT_FALLBACK_ANSWER,
                needs_sql=True,
                rationale=_DB_RAG_CONTEXT_FALLBACK_RATIONALE,
                relevant_tables=context.table_names,
                relevant_columns=context.column_names,
            )

        answer_text = str(parsed.get("answer", "") or "").strip() or _DB_RAG_CONTEXT_FALLBACK_ANSWER
        rationale = str(parsed.get("rationale", "") or "").strip() or _DB_RAG_CONTEXT_FALLBACK_RATIONALE
        needs_sql = needs_sql_value
        return DbRagQaAnswer(
            answer=answer_text,
            needs_sql=needs_sql,
            rationale=rationale,
            relevant_tables=context.table_names,
            relevant_columns=context.column_names,
        )

    def prepare_column_selection(
        self,
        question: str,
        context: DbRagContext,
        feedback_history: list[dict[str, Any]] | None = None,
    ) -> ColumnSelectionCandidate:
        normalized_feedback_history = list(feedback_history or [])
        valid_tables = set(context.table_names)
        valid_columns = {(entry.table, entry.column) for entry in context.columns}
        response = self.llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are preparing a column selection candidate for a RePORT database question. "
                        "Use the retrieved table and column context plus any reviewer feedback to select only "
                        "exact tables and exact table.column pairs relevant to the question. "
                        "Return only a JSON object with exactly these keys: "
                        '{"selection_id": string, "rationale": string, "tables": [string], '
                        '"columns": [{"table": string, "column": string, "description": string}]}.'
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question:\n{question}\n\n"
                        f"Table context:\n{context.table_context or 'none'}\n\n"
                        f"Column context:\n{context.column_context or 'none'}\n\n"
                        f"Feedback history:\n{json.dumps(normalized_feedback_history, indent=2, sort_keys=True)}"
                    )
                ),
            ]
        )
        parsed = _parse_json_object(coerce_text_content(getattr(response, "content", "")))
        if not parsed:
            return _build_invalid_column_selection_candidate(question, context, normalized_feedback_history)

        raw_tables = parsed.get("tables")
        tables: list[str] = []
        if isinstance(raw_tables, list):
            for value in raw_tables:
                table = str(value or "").strip()
                if table and table in valid_tables and table not in tables:
                    tables.append(table)

        raw_columns = parsed.get("columns")
        columns: list[dict[str, str]] = []
        if isinstance(raw_columns, list):
            for item in raw_columns:
                if not isinstance(item, dict):
                    continue
                table = str(item.get("table", "") or "").strip()
                column = str(item.get("column", "") or "").strip()
                if not table or not column or (table, column) not in valid_columns:
                    continue
                description = str(item.get("description", "") or "").strip()
                columns.append(
                    {
                        "table": table,
                        "column": column,
                        "description": description,
                    }
                )

        if not tables and columns:
            for column in columns:
                table = column["table"]
                if table not in tables:
                    tables.append(table)

        selection_id = str(parsed.get("selection_id", "") or "").strip() or _default_selection_id(
            question,
            context,
            normalized_feedback_history,
        )
        rationale = str(parsed.get("rationale", "") or "").strip()

        if not tables and not columns:
            return _build_invalid_column_selection_candidate(question, context, normalized_feedback_history)

        return ColumnSelectionCandidate(
            selection_id=selection_id,
            question=question,
            tables=tables,
            columns=columns,
            rationale=rationale,
            feedback_history=normalized_feedback_history,
        )

    def prepare_sql_candidate(
        self,
        question: str,
        approved_selection: ColumnSelectionCandidate,
    ) -> PreparedSqlCandidate:
        if approved_selection.status != "approved":
            raise ValueError("prepare_sql_candidate requires an approved column selection.")

        approved_tables = list(approved_selection.tables)
        approved_columns = [
            {
                "table": str(column["table"]),
                "column": str(column["column"]),
                "description": str(column.get("description", "") or ""),
            }
            for column in approved_selection.columns
        ]
        response = self.llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a DuckDB SQL expert for the RePORT clinical research database. "
                        "Use only the approved tables and approved columns listed in the prompt. "
                        "Do not reference any other tables or columns. "
                        "Return only read-only DuckDB SQL using SELECT or WITH. "
                        "Never emit mutating or DDL statements."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question:\n{question}\n\n"
                        f"Approved tables:\n{json.dumps(approved_tables, indent=2)}\n\n"
                        f"Approved columns:\n{json.dumps(approved_columns, indent=2, sort_keys=True)}"
                    )
                ),
            ]
        )
        sql = _extract_sql(coerce_text_content(getattr(response, "content", "")))
        valid, error = _validate_sql(sql)
        if not valid:
            raise ValueError(error or "SQL validation failed.")
        return PreparedSqlCandidate(
            question=question,
            sql=sql,
            tables=approved_tables,
            columns=approved_columns,
            selection_id=approved_selection.selection_id,
        )

    def execute_prepared_sql(self, candidate: PreparedSqlCandidate) -> SqlExecutionResult:
        import duckdb

        valid, error = _validate_sql(candidate.sql)
        if not valid:
            raise ValueError(error or "SQL validation failed.")

        db = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        try:
            dataframe = db.execute(candidate.sql).fetchdf()
        finally:
            db.close()

        return SqlExecutionResult(
            answer=f"Read-only SQL execution completed with {len(dataframe)} result row(s).",
            sql=candidate.sql,
            dataframe=dataframe,
            source_tables=list(candidate.tables),
        )

    def execute_sql_flow(self, question: str) -> dict[str, Any]:
        context = self.retrieve_context(question)
        prepared = self.prepare_sql_candidate(
            question,
            ColumnSelectionCandidate(
                selection_id=_default_selection_id(question, context, []),
                question=question,
                tables=context.table_names,
                columns=[
                    {
                        "table": entry.table,
                        "column": entry.column,
                        "description": entry.text,
                    }
                    for entry in context.columns
                ],
                rationale="Legacy execute_sql_flow compatibility path.",
                status="approved",
            ),
        )
        result = self.execute_prepared_sql(prepared)
        return {
            "answer": result.answer,
            "sql": result.sql,
            "dataframe": result.dataframe,
            "source_tables": result.source_tables,
        }
