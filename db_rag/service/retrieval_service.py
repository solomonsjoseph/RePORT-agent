from __future__ import annotations

import importlib.util
from typing import Any

from utils.performance import timing_stage

from .models import DbRagColumnHit, DbRagContext, DbRagIntent, DbRagTableHit
from .constraints import _constraint_set_from_payload, _filter_and_inject_context_columns
from ..config import (
    DUCKDB_PATH,
    EXCEL_DIR,
    SCHEMA_DIR,
    chroma_dir_for_model,
    load_manifest_for_model,
    manifest_path_for_model,
    resolve_db_rag_embedding_model,
)
from ..retrieval import decompose_query, retrieve_context_records, retrieve_single_query
from ..vectorstore import OpenAIEmbeddingFunction


class DbRagRetrievalMixin:
    llm: Any
    indexing_model: str | None

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
                    "`python -m db_rag.build_index --rebuild`."
                ),
            }

        for dependency in ("chromadb", "duckdb"):
            if importlib.util.find_spec(dependency) is None:
                return {
                    "ready": False,
                    "message": (
                        f"DB-RAG dependency '{dependency}' is not installed. Install the project requirements and run "
                        "`python -m db_rag.build_index --rebuild`."
                    ),
                }

        return {"ready": True, "message": ""}

    def _load_collections(self):
        import chromadb

        model = self.indexing_model or resolve_db_rag_embedding_model()
        with timing_stage("db_rag.retrieval.load_collections", model=model):
            load_manifest_for_model(model)
            client = chromadb.PersistentClient(path=str(chroma_dir_for_model(model)))
            ef = OpenAIEmbeddingFunction(model=model)
            table_collection = client.get_collection("table_summaries", embedding_function=ef)
            column_collection = client.get_collection("column_chunks", embedding_function=ef)
        return table_collection, column_collection

    def decompose_query(self, question: str) -> list[str]:
        return decompose_query(self.llm, question)

    def _retrieve_context_single_query(self, table_collection, column_collection, query: str) -> DbRagContext:
        table_rows, column_rows = retrieve_single_query(table_collection, column_collection, query)
        tables = [DbRagTableHit(**entry) for entry in table_rows]
        columns = [DbRagColumnHit(**entry) for entry in column_rows]
        return DbRagContext(
            tables=tables,
            columns=columns,
            table_context="\n\n".join(entry.text for entry in tables),
            column_context="\n\n".join(entry.text for entry in columns),
        )

    def retrieve_context(
        self,
        question: str,
        *,
        debug: bool = False,
        reranker_model: str | None = None,
    ) -> DbRagContext:
        with timing_stage("db_rag.retrieval.retrieve_context", reranker_model=reranker_model):
            table_rows, column_rows = self.retrieve_context_records(
                question,
                debug=debug,
                reranker_model=reranker_model,
            )
        tables = [DbRagTableHit(**entry) for entry in table_rows]
        columns = [DbRagColumnHit(**entry) for entry in column_rows]
        return DbRagContext(
            tables=tables,
            columns=columns,
            table_context="\n\n".join(entry.text for entry in tables),
            column_context="\n\n".join(entry.text for entry in columns),
        )

    def retrieve_context_records(
        self,
        question: str,
        *,
        debug: bool = False,
        reranker_model: str | None = None,
        required_tables: list[str] | None = None,
        excluded_tables: list[str] | None = None,
    ) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        table_collection, column_collection = self._load_collections()
        with timing_stage("db_rag.retrieval.retrieve_context_records", reranker_model=reranker_model):
            return retrieve_context_records(
                self.llm,
                table_collection,
                column_collection,
                question,
                reranker_model=reranker_model,
                debug=debug,
                required_tables=required_tables,
                excluded_tables=excluded_tables,
            )

    def retrieve_context_for_intent(
        self,
        intent: DbRagIntent,
        *,
        reranker_model: str | None = None,
    ) -> DbRagContext:
        constraints = _constraint_set_from_payload(intent)
        table_rows, column_rows = self.retrieve_context_records(
            intent.goal_text,
            reranker_model=reranker_model,
            required_tables=list(constraints.required_tables),
            excluded_tables=list(constraints.excluded_tables),
        )
        column_rows = _filter_and_inject_context_columns(column_rows, constraints)
        return DbRagContext(
            tables=[DbRagTableHit(**entry) for entry in table_rows],
            columns=[DbRagColumnHit(**entry) for entry in column_rows],
            table_context="\n\n".join(entry["text"] for entry in table_rows),
            column_context="\n\n".join(entry["text"] for entry in column_rows),
        )
