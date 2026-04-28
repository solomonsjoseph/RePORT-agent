from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
import re
from typing import Any

from utils.llm_response import coerce_text_content

from .config import (
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
)
from .generation import (
    DB_RAG_CONTEXT_FALLBACK_ANSWER,
    DB_RAG_CONTEXT_FALLBACK_RATIONALE,
    default_selection_id,
    extract_sql,
    parse_json_object,
    validate_sql,
)
from .retrieval import PAIRED_FORMS, decompose_query, retrieve_context_records, retrieve_single_query
from .vectorstore import OpenAIEmbeddingFunction


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
class DbRagIntent:
    intent_id: str
    source_question: str
    goal_text: str
    mode: str
    population: str | None
    requested_fields: list[str] = field(default_factory=list)
    filters: list[str] = field(default_factory=list)
    required_tables: list[str] = field(default_factory=list)
    required_columns: list[str] = field(default_factory=list)
    excluded_tables: list[str] = field(default_factory=list)
    excluded_columns: list[str] = field(default_factory=list)
    feedback_history: list[dict[str, str]] = field(default_factory=list)
    status: str = "active"


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


@lru_cache(maxsize=1)
def _schema_column_catalog() -> tuple[dict[tuple[str, str], dict[str, str]], dict[str, list[dict[str, str]]]]:
    by_pair: dict[tuple[str, str], dict[str, str]] = {}
    by_column: dict[str, list[dict[str, str]]] = {}
    for schema_file in sorted(SCHEMA_DIR.glob("*.json")):
        schema = json.loads(schema_file.read_text(encoding="utf-8"))
        table = str(schema.get("form_name", "") or "").strip()
        variables = dict(schema.get("variables", {}) or {})
        if not table:
            continue
        for column, info in variables.items():
            column_name = str(column or "").strip()
            if not column_name:
                continue
            entry = {
                "table": table,
                "column": column_name,
                "description": str((info or {}).get("description", "") or "").strip(),
            }
            by_pair[(table, column_name)] = entry
            by_column.setdefault(column_name, []).append(entry)
    return by_pair, by_column


def _lookup_schema_column(table: str, column: str) -> dict[str, str] | None:
    by_pair, _by_column = _schema_column_catalog()
    return by_pair.get((str(table or "").strip(), str(column or "").strip()))


def _schema_table_names() -> set[str]:
    by_pair, _ = _schema_column_catalog()
    return {table for table, _column in by_pair}


def _resolve_explicit_schema_mentions(
    question: str,
    feedback_history: list[dict[str, Any]],
) -> list[dict[str, str]]:
    by_pair, by_column = _schema_column_catalog()
    texts = [str(question or "")]
    texts.extend(str(item.get("feedback") or "") for item in feedback_history if isinstance(item, dict))
    haystack = "\n".join(texts)
    haystack_lower = haystack.lower()

    matched: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for (table, column), entry in by_pair.items():
        token = f"{table}.{column}".lower()
        if token in haystack_lower and (table, column) not in seen_pairs:
            matched.append(dict(entry))
            seen_pairs.add((table, column))

    for column_name, entries in by_column.items():
        if len(entries) != 1:
            continue
        if re.search(rf"\b{re.escape(column_name)}\b", haystack, re.IGNORECASE):
            entry = entries[0]
            pair = (entry["table"], entry["column"])
            if pair not in seen_pairs:
                matched.append(dict(entry))
                seen_pairs.add(pair)

    return matched


def _normalize_previous_selection(previous_selection: Any) -> dict[str, Any]:
    if isinstance(previous_selection, ColumnSelectionCandidate):
        return {
            "selection_id": previous_selection.selection_id,
            "question": previous_selection.question,
            "tables": list(previous_selection.tables),
            "columns": [dict(column) for column in previous_selection.columns],
            "rationale": previous_selection.rationale,
            "feedback_history": list(previous_selection.feedback_history),
            "status": previous_selection.status,
        }
    if isinstance(previous_selection, dict):
        return {
            "selection_id": str(previous_selection.get("selection_id", "") or "").strip(),
            "question": str(previous_selection.get("question", "") or "").strip(),
            "tables": [str(value or "").strip() for value in list(previous_selection.get("tables") or []) if str(value or "").strip()],
            "columns": [
                {
                    "table": str(column.get("table", "") or "").strip(),
                    "column": str(column.get("column", "") or "").strip(),
                    "description": str(column.get("description", "") or "").strip(),
                }
                for column in list(previous_selection.get("columns") or [])
                if isinstance(column, dict)
                and str(column.get("table", "") or "").strip()
                and str(column.get("column", "") or "").strip()
            ],
            "rationale": str(previous_selection.get("rationale", "") or "").strip(),
            "feedback_history": list(previous_selection.get("feedback_history") or []),
            "status": str(previous_selection.get("status", "") or "").strip(),
        }
    return {
        "selection_id": "",
        "question": "",
        "tables": [],
        "columns": [],
        "rationale": "",
        "feedback_history": [],
        "status": "",
    }


def _build_invalid_column_selection_candidate(
    question: str,
    context: DbRagContext,
    feedback_history: list[dict[str, Any]],
) -> ColumnSelectionCandidate:
    columns = [{"table": entry.table, "column": entry.column} for entry in context.columns]
    return ColumnSelectionCandidate(
        selection_id=default_selection_id(question, context.table_names, columns, feedback_history),
        question=question,
        tables=[],
        columns=[],
        rationale=DB_RAG_CONTEXT_FALLBACK_RATIONALE,
        feedback_history=feedback_history,
    )


def _default_selection_id(question: str, context: DbRagContext, feedback_history: list[dict[str, Any]]) -> str:
    columns = [{"table": entry.table, "column": entry.column} for entry in context.columns]
    return default_selection_id(question, context.table_names, columns, feedback_history)


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
        table_collection, column_collection = self._load_collections()
        table_rows, column_rows = retrieve_context_records(
            self.llm,
            table_collection,
            column_collection,
            intent.goal_text,
            reranker_model=reranker_model,
            required_tables=list(intent.required_tables),
            excluded_tables=list(intent.excluded_tables),
        )
        return DbRagContext(
            tables=[DbRagTableHit(**entry) for entry in table_rows],
            columns=[DbRagColumnHit(**entry) for entry in column_rows],
            table_context="\n\n".join(entry["text"] for entry in table_rows),
            column_context="\n\n".join(entry["text"] for entry in column_rows),
        )

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
        intent_excluded_tables = {
            str(value or "").strip()
            for value in list((intent_snapshot or {}).get("excluded_tables") or [])
            if str(value or "").strip()
        }
        intent_excluded_columns = {
            str(value or "").strip()
            for value in list((intent_snapshot or {}).get("excluded_columns") or [])
            if str(value or "").strip()
        }
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
                if table and table in valid_tables and table not in tables and table not in intent_excluded_tables:
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
                if table in intent_excluded_tables:
                    continue
                if f"{table}.{column}" in intent_excluded_columns:
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
            if column["table"] not in intent_excluded_tables
            and f'{column["table"]}.{column["column"]}' not in intent_excluded_columns
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
            if column["table"] not in intent_excluded_tables
            and f'{column["table"]}.{column["column"]}' not in intent_excluded_columns
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
        sql = extract_sql(coerce_text_content(getattr(response, "content", "")))
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
                        "Never emit mutating or DDL statements."
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
        prepared = self.prepare_sql_candidate(
            question,
            ColumnSelectionCandidate(
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
            ),
        )
        result = self.execute_prepared_sql(prepared)
        return {
            "answer": result.answer,
            "sql": result.sql,
            "dataframe": result.dataframe,
            "source_tables": result.source_tables,
            "debug": (
                {
                    "question": question,
                    "retrieved_tables": context.table_names,
                    "retrieved_columns": [entry.as_prompt_line() for entry in context.columns],
                    "sql_tables": list(prepared.tables),
                    "sql_columns": [f"{column['table']}.{column['column']}" for column in prepared.columns],
                }
                if debug
                else {}
            ),
        }
