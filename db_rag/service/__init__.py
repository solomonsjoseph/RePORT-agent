from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from utils.llm_response import coerce_text_content

from . import schema
from .classifier import classify_pending_reply as _classify_pending_reply
from .models import (
    ColumnSelectionCandidate,
    DbRagColumnHit,
    DbRagContext,
    DbRagIntent,
    DbRagQaAnswer,
    DbRagTableHit,
    FeedbackConstraintSet,
    PreparedSqlCandidate,
    SqlExecutionResult,
)
from .constraints import (
    _build_invalid_column_selection_candidate,
    _constraint_set_from_payload,
    _default_selection_id,
    _enforce_selection_constraints,
    _filter_and_inject_context_columns,
    _merge_constraint_sets,
    _normalize_previous_selection,
)
from .schema import (
    _lookup_schema_column,
    _resolve_explicit_schema_mentions,
    _schema_column_catalog,
    _schema_table_names,
)

from ..config import (
    CHROMA_DIR,
    DEFAULT_OPENROUTER_BASE_URL,
    DUCKDB_PATH,
    EMBEDDING_MODEL,
    EXCEL_DIR,
    INDEX_ROOT,
    MANIFEST_PATH,
    MANIFEST_ROOT,
    PROJECT_ROOT,
    RUNTIME_ROOT,
    SCHEMA_DIR,
    SOURCE_ROOT,
    SUPPORTED_DB_RAG_EMBEDDING_MODELS,
    chroma_dir_for_model,
    embedding_model_slug,
    load_manifest_for_model,
    manifest_path_for_model,
    resolve_db_rag_embedding_model,
    resolve_db_rag_reply_classifier_model,
)
from ..generation import (
    DB_RAG_CONTEXT_FALLBACK_ANSWER,
    DB_RAG_CONTEXT_FALLBACK_RATIONALE,
    build_sql_policy_text,
    default_selection_id,
    extract_sql,
    is_unanswerable_response,
    parse_json_object,
    validate_sql,
)
from ..retrieval import PAIRED_FORMS, decompose_query, retrieve_context_records, retrieve_single_query
from ..vectorstore import OpenAIEmbeddingFunction


class DbRagUnanswerableError(ValueError):
    pass


@dataclass
class DbRagService:
    llm: Any

    def classify_pending_reply(
        self,
        *,
        pending_kind: str,
        pending_question: str,
        user_reply: str,
        recent_transcript: str,
    ) -> dict[str, Any]:
        del pending_kind
        return _classify_pending_reply(
            pending_question=pending_question,
            user_reply=user_reply,
            recent_transcript=recent_transcript,
            resolve_model=resolve_db_rag_reply_classifier_model,
        )

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

        try:
            import chromadb  # noqa: F401
            import duckdb  # noqa: F401
        except ModuleNotFoundError as exc:
            return {
                "ready": False,
                "message": (
                    f"DB-RAG dependency '{exc.name}' is not installed. Install the project requirements and run "
                    "`python -m db_rag.build_index --rebuild`."
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
        table_collection, column_collection = self._load_collections()
        table_rows, column_rows = retrieve_context_records(
            self.llm,
            table_collection,
            column_collection,
            question,
            reranker_model=reranker_model,
            debug=debug,
        )
        tables = [DbRagTableHit(**entry) for entry in table_rows]
        columns = [DbRagColumnHit(**entry) for entry in column_rows]
        return DbRagContext(
            tables=tables,
            columns=columns,
            table_context="\n\n".join(entry.text for entry in tables),
            column_context="\n\n".join(entry.text for entry in columns),
        )

    def retrieve_context_for_intent(
        self,
        intent: DbRagIntent,
        *,
        reranker_model: str | None = None,
    ) -> DbRagContext:
        constraints = _constraint_set_from_payload(intent)
        table_collection, column_collection = self._load_collections()
        table_rows, column_rows = retrieve_context_records(
            self.llm,
            table_collection,
            column_collection,
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

    def ground_feedback_constraints(
        self,
        question: str,
        context: DbRagContext,
        *,
        feedback_history: list[dict[str, Any]] | None = None,
        previous_selection: Any = None,
        intent_snapshot: dict[str, Any] | None = None,
    ) -> FeedbackConstraintSet:
        from langchain_core.messages import HumanMessage, SystemMessage

        normalized_feedback_history = list(feedback_history or [])
        normalized_previous_selection = _normalize_previous_selection(previous_selection)
        response = self.llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are grounding human review feedback for a RePORT DB-RAG column selection task. "
                        "Map concept-level feedback onto exact schema table names and exact table.column strings "
                        "using only the provided retrieved context and previous selection. "
                        "Return JSON only with keys: "
                        '{"goal_text": string, "required_tables": [string], "required_columns": [string], '
                        '"excluded_tables": [string], "excluded_columns": [string]}.'
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question:\n{question}\n\n"
                        f"Table context:\n{context.table_context or 'none'}\n\n"
                        f"Column context:\n{context.column_context or 'none'}\n\n"
                        f"Intent snapshot:\n{json.dumps(intent_snapshot or {}, indent=2, sort_keys=True)}\n\n"
                        f"Feedback history:\n{json.dumps(normalized_feedback_history, indent=2, sort_keys=True)}\n\n"
                        f"Previous selection candidate:\n{json.dumps(normalized_previous_selection, indent=2, sort_keys=True)}"
                    )
                ),
            ]
        )
        parsed = parse_json_object(coerce_text_content(getattr(response, "content", ""))) or {}
        return _constraint_set_from_payload(parsed)

    def answer_question(self, question: str) -> dict[str, Any]:
        from langchain_core.messages import HumanMessage, SystemMessage

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
        from langchain_core.messages import HumanMessage, SystemMessage

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
        parsed = parse_json_object(coerce_text_content(getattr(response, "content", "")))
        needs_sql_value = parsed.get("needs_sql")
        if not parsed or not isinstance(needs_sql_value, bool):
            return DbRagQaAnswer(
                answer=DB_RAG_CONTEXT_FALLBACK_ANSWER,
                needs_sql=True,
                rationale=DB_RAG_CONTEXT_FALLBACK_RATIONALE,
                relevant_tables=context.table_names,
                relevant_columns=context.column_names,
            )

        answer_text = str(parsed.get("answer", "") or "").strip() or DB_RAG_CONTEXT_FALLBACK_ANSWER
        rationale = str(parsed.get("rationale", "") or "").strip() or DB_RAG_CONTEXT_FALLBACK_RATIONALE
        return DbRagQaAnswer(
            answer=answer_text,
            needs_sql=needs_sql_value,
            rationale=rationale,
            relevant_tables=context.table_names,
            relevant_columns=context.column_names,
        )

    def resolve_intent(
        self,
        question: str,
        context: DbRagContext,
        prior_intent: dict[str, Any] | None = None,
    ) -> DbRagIntent:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = self.llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are normalizing a RePORT DB-RAG request into a structured intent. "
                        "Return only JSON with keys intent_id, goal_text, mode, population, requested_fields, filters."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question:\n{question}\n\n"
                        f"Table context:\n{context.table_context or 'none'}\n\n"
                        f"Column context:\n{context.column_context or 'none'}\n\n"
                        f"Prior intent:\n{json.dumps(prior_intent or {}, indent=2, sort_keys=True)}"
                    )
                ),
            ]
        )
        parsed = parse_json_object(coerce_text_content(getattr(response, "content", ""))) or {}
        return DbRagIntent(
            intent_id=str(parsed.get("intent_id") or default_selection_id(question, context.table_names, [], [])),
            source_question=question,
            goal_text=str(parsed.get("goal_text") or question).strip(),
            mode=str(parsed.get("mode") or "metadata").strip(),
            population=str(parsed.get("population") or "").strip() or None,
            requested_fields=[str(value).strip() for value in list(parsed.get("requested_fields") or []) if str(value).strip()],
            filters=[str(value).strip() for value in list(parsed.get("filters") or []) if str(value).strip()],
        )

    def validate_selection_against_intent(
        self,
        candidate: ColumnSelectionCandidate,
        intent: DbRagIntent,
    ) -> tuple[bool, str]:
        selected_tables = set(candidate.tables)
        selected_columns = {f'{entry["table"]}.{entry["column"]}' for entry in candidate.columns}
        for table in intent.required_tables:
            if table not in selected_tables:
                return False, f"Missing required table: {table}"
        for column in intent.required_columns:
            if column not in selected_columns:
                return False, f"Missing required column: {column}"
        for table in intent.excluded_tables:
            if table in selected_tables:
                return False, f"Selection includes excluded table: {table}"
        for column in intent.excluded_columns:
            if column in selected_columns:
                return False, f"Selection includes excluded column: {column}"
        return True, ""

    def update_intent_from_feedback(
        self,
        intent: DbRagIntent | dict[str, Any],
        feedback_history: list[dict[str, str]],
    ) -> DbRagIntent:
        from langchain_core.messages import HumanMessage, SystemMessage

        base = intent if isinstance(intent, DbRagIntent) else DbRagIntent(**intent)
        response = self.llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are updating a structured DB-RAG intent from human review feedback. "
                        "Return JSON only with keys: "
                        '{"goal_text": string, "required_tables": [string], "required_columns": [string], '
                        '"excluded_tables": [string], "excluded_columns": [string]}. '
                        "Use exact schema table names and exact 'table.column' strings when possible."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Current intent:\n{json.dumps(base.__dict__, indent=2, sort_keys=True)}\n\n"
                        f"Feedback history:\n{json.dumps(list(feedback_history or []), indent=2, sort_keys=True)}"
                    )
                ),
            ]
        )
        parsed = parse_json_object(coerce_text_content(getattr(response, "content", ""))) or {}

        schema_tables = _schema_table_names()
        required_tables: list[str] = []
        required_columns: list[str] = []
        excluded_tables: list[str] = []
        excluded_columns: list[str] = []

        for table in [*list(base.required_tables), *list(parsed.get("required_tables") or [])]:
            table_name = str(table or "").strip()
            if table_name and table_name in schema_tables and table_name not in required_tables:
                required_tables.append(table_name)

        for item in [*list(base.required_columns), *list(parsed.get("required_columns") or [])]:
            value = str(item or "").strip()
            if "." not in value:
                continue
            table_name, column_name = value.split(".", 1)
            if _lookup_schema_column(table_name, column_name) is None:
                continue
            normalized = f"{table_name}.{column_name}"
            if normalized not in required_columns:
                required_columns.append(normalized)

        for table in [*list(base.excluded_tables), *list(parsed.get("excluded_tables") or [])]:
            table_name = str(table or "").strip()
            if table_name and table_name in schema_tables and table_name not in excluded_tables:
                excluded_tables.append(table_name)

        for item in [*list(base.excluded_columns), *list(parsed.get("excluded_columns") or [])]:
            value = str(item or "").strip()
            if "." not in value:
                continue
            table_name, column_name = value.split(".", 1)
            if _lookup_schema_column(table_name, column_name) is None:
                continue
            normalized = f"{table_name}.{column_name}"
            if normalized not in excluded_columns:
                excluded_columns.append(normalized)

        # Exclusions take precedence over requirements.
        required_tables = [table for table in required_tables if table not in excluded_tables]
        required_columns = [column for column in required_columns if column not in excluded_columns]

        goal_text = str(parsed.get("goal_text") or base.goal_text).strip() or base.goal_text
        return DbRagIntent(
            intent_id=base.intent_id,
            source_question=base.source_question,
            goal_text=goal_text,
            mode=base.mode,
            population=base.population,
            requested_fields=list(base.requested_fields),
            filters=list(base.filters),
            required_tables=required_tables,
            required_columns=required_columns,
            excluded_tables=excluded_tables,
            excluded_columns=excluded_columns,
            feedback_history=list(feedback_history),
            status=base.status,
        )

    def prepare_column_selection(
        self,
        question: str,
        context: DbRagContext,
        feedback_history: list[dict[str, Any]] | None = None,
        previous_selection: Any = None,
        intent_snapshot: dict[str, Any] | None = None,
    ) -> ColumnSelectionCandidate:
        from langchain_core.messages import HumanMessage, SystemMessage

        normalized_feedback_history = list(feedback_history or [])
        normalized_previous_selection = _normalize_previous_selection(previous_selection)
        base_constraints = _constraint_set_from_payload(intent_snapshot)
        grounded_constraints = self.ground_feedback_constraints(
            question,
            context,
            feedback_history=normalized_feedback_history,
            previous_selection=previous_selection,
            intent_snapshot=intent_snapshot,
        )
        constraints = _merge_constraint_sets(base_constraints, grounded_constraints)
        valid_tables = set(context.table_names)
        valid_columns = {(entry.table, entry.column) for entry in context.columns}
        explicit_schema_columns = _resolve_explicit_schema_mentions(question, normalized_feedback_history)
        explicit_valid_columns = [
            column
            for column in explicit_schema_columns
            if _lookup_schema_column(column["table"], column["column"]) is not None
        ]
        constrained_columns = [
            column
            for column in normalized_previous_selection.get("columns", [])
            if _lookup_schema_column(column["table"], column["column"]) is not None
        ]
        response = self.llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are preparing a column selection candidate for a RePORT database question. "
                        "Use the retrieved table and column context plus any reviewer feedback to select only "
                        "exact tables and exact table.column pairs relevant to the question. "
                        "If the prompt includes a previous reviewed selection or explicit schema constraints, "
                        "preserve those exact table.column pairs unless the reviewer feedback explicitly replaces them. "
                        "Do not drop required exact columns just because they were not top-ranked in retrieval. "
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
                        f"Intent snapshot:\n{json.dumps(intent_snapshot or {}, indent=2, sort_keys=True)}\n\n"
                        f"Grounded constraints:\n{json.dumps(constraints.__dict__, indent=2, sort_keys=True)}\n\n"
                        f"Feedback history:\n{json.dumps(normalized_feedback_history, indent=2, sort_keys=True)}\n\n"
                        f"Previous selection candidate:\n{json.dumps(normalized_previous_selection, indent=2, sort_keys=True)}\n\n"
                        f"Explicit schema constraints:\n{json.dumps(explicit_valid_columns, indent=2, sort_keys=True)}"
                    )
                ),
            ]
        )
        parsed = parse_json_object(coerce_text_content(getattr(response, "content", "")))
        if not parsed:
            return _build_invalid_column_selection_candidate(question, context, normalized_feedback_history)

        raw_tables = parsed.get("tables")
        tables: list[str] = []
        if isinstance(raw_tables, list):
            for value in raw_tables:
                table = str(value or "").strip()
                if table and table in valid_tables and table not in tables and table not in constraints.excluded_tables:
                    tables.append(table)

        raw_columns = parsed.get("columns")
        columns: list[dict[str, str]] = []
        if isinstance(raw_columns, list):
            for item in raw_columns:
                if not isinstance(item, dict):
                    continue
                table = str(item.get("table", "") or "").strip()
                column = str(item.get("column", "") or "").strip()
                if not table or not column:
                    continue
                if table in constraints.excluded_tables:
                    continue
                if f"{table}.{column}" in constraints.excluded_columns:
                    continue
                schema_entry = _lookup_schema_column(table, column)
                if (table, column) not in valid_columns and schema_entry is None:
                    continue
                description = str(item.get("description", "") or "").strip()
                if not description and schema_entry is not None:
                    description = schema_entry["description"]
                columns.append({"table": table, "column": column, "description": description})

        constrained_pairs = {
            (column["table"], column["column"]): column
            for column in [*explicit_valid_columns, *constrained_columns]
            if column["table"] not in constraints.excluded_tables
            and f'{column["table"]}.{column["column"]}' not in constraints.excluded_columns
        }
        current_pairs = {(column["table"], column["column"]) for column in columns}
        for pair, column in constrained_pairs.items():
            if pair in current_pairs:
                continue
            columns.append(dict(column))
            current_pairs.add(pair)

        columns = [
            column
            for column in columns
            if column["table"] not in constraints.excluded_tables
            and f'{column["table"]}.{column["column"]}' not in constraints.excluded_columns
        ]

        if not tables and columns:
            for column in columns:
                table = column["table"]
                if table not in tables:
                    tables.append(table)

        for column in columns:
            table = column["table"]
            if table not in tables:
                tables.append(table)

        selection_id = str(parsed.get("selection_id", "") or "").strip() or default_selection_id(
            question,
            context.table_names,
            [{"table": entry.table, "column": entry.column} for entry in context.columns],
            normalized_feedback_history,
        )
        rationale = str(parsed.get("rationale", "") or "").strip()

        if not columns:
            return _build_invalid_column_selection_candidate(question, context, normalized_feedback_history)

        candidate = ColumnSelectionCandidate(
            selection_id=selection_id,
            question=question,
            tables=tables,
            columns=columns,
            rationale=rationale,
            feedback_history=normalized_feedback_history,
        )
        return _enforce_selection_constraints(candidate, context, constraints)

    def prepare_sql_candidate(
        self,
        question: str,
        approved_selection: ColumnSelectionCandidate,
    ) -> PreparedSqlCandidate:
        from langchain_core.messages import HumanMessage, SystemMessage

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
                        "Never emit mutating or DDL statements. "
                        "If the approved schema cannot answer the question, return exactly "
                        "'UNANSWERABLE: <brief reason>'.\n\n"
                        f"{build_sql_policy_text()}"
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
        sql = extract_sql(coerce_text_content(getattr(response, "content", "")))
        if is_unanswerable_response(sql):
            raise DbRagUnanswerableError(sql)
        valid, error = validate_sql(sql)
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

        valid, error = validate_sql(candidate.sql)
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

    def repair_prepared_sql_candidate(self, candidate: PreparedSqlCandidate, error_message: str) -> PreparedSqlCandidate:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = self.llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are fixing a DuckDB SQL query for the RePORT clinical research database. "
                        "Use only the approved tables and approved columns. "
                        "Return only read-only DuckDB SQL using SELECT or WITH. "
                        "Never emit mutating or DDL statements.\n\n"
                        f"{build_sql_policy_text()}"
                    )
                ),
                HumanMessage(
                    content=(
                        f"Original question:\n{candidate.question}\n\n"
                        f"Approved tables:\n{json.dumps(list(candidate.tables), indent=2)}\n\n"
                        f"Approved columns:\n{json.dumps(list(candidate.columns), indent=2, sort_keys=True)}\n\n"
                        f"SQL that failed:\n{candidate.sql}\n\n"
                        f"Execution/validation error:\n{error_message}\n\n"
                        "Fix the SQL and return only the corrected SQL."
                    )
                ),
            ]
        )
        repaired_sql = extract_sql(coerce_text_content(getattr(response, "content", "")))
        if is_unanswerable_response(repaired_sql):
            raise DbRagUnanswerableError(repaired_sql)
        valid, error = validate_sql(repaired_sql)
        if not valid:
            raise ValueError(error or "SQL validation failed.")
        return PreparedSqlCandidate(
            question=candidate.question,
            sql=repaired_sql,
            tables=list(candidate.tables),
            columns=[dict(column) for column in candidate.columns],
            selection_id=candidate.selection_id,
            status=candidate.status,
        )

    def execute_sql_flow(
        self,
        question: str,
        *,
        debug: bool = False,
        reranker_model: str | None = None,
    ) -> dict[str, Any]:
        context = self.retrieve_context(question, debug=debug, reranker_model=reranker_model)
        approved_selection = ColumnSelectionCandidate(
            selection_id=default_selection_id(
                question,
                context.table_names,
                [{"table": entry.table, "column": entry.column} for entry in context.columns],
                [],
            ),
            question=question,
            tables=context.table_names,
            columns=[
                {"table": entry.table, "column": entry.column, "description": entry.text}
                for entry in context.columns
            ],
            rationale="Legacy execute_sql_flow compatibility path.",
            status="approved",
        )
        debug_payload = (
            {
                "question": question,
                "retrieved_tables": context.table_names,
                "retrieved_columns": [entry.as_prompt_line() for entry in context.columns],
            }
            if debug
            else {}
        )
        max_retries = 2
        candidate: PreparedSqlCandidate | None = None
        result: SqlExecutionResult | None = None
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                if candidate is None:
                    candidate = self.prepare_sql_candidate(question, approved_selection)
                result = self.execute_prepared_sql(candidate)
                break
            except DbRagUnanswerableError as exc:
                return {
                    "answer": str(exc),
                    "sql": "",
                    "dataframe": None,
                    "source_tables": context.table_names,
                    "debug": debug_payload,
                }
            except ValueError as exc:
                last_error = exc
                if attempt >= max_retries:
                    raise
                repair_seed = candidate or PreparedSqlCandidate(
                    question=question,
                    sql="SELECT 1",
                    tables=list(approved_selection.tables),
                    columns=[dict(column) for column in approved_selection.columns],
                    selection_id=approved_selection.selection_id,
                )
                candidate = self.repair_prepared_sql_candidate(repair_seed, str(exc))
            except Exception as exc:
                last_error = exc
                if candidate is None or attempt >= max_retries:
                    raise
                candidate = self.repair_prepared_sql_candidate(candidate, str(exc))
        if result is None or candidate is None:
            if last_error is not None:
                raise last_error
            raise ValueError("SQL execution did not produce a result.")
        if debug:
            debug_payload["sql_tables"] = list(candidate.tables)
            debug_payload["sql_columns"] = [f"{column['table']}.{column['column']}" for column in candidate.columns]
        return {
            "answer": result.answer,
            "sql": result.sql,
            "dataframe": result.dataframe,
            "source_tables": result.source_tables,
            "debug": debug_payload,
        }
