from __future__ import annotations

import ast
from types import SimpleNamespace
from typing import Any
from uuid import uuid4
import re

from langchain_core.messages import AIMessage
from db_rag.service.schema import _lookup_schema_variable_metadata
from utils.dataset_artifacts import (
    DEFAULT_RUNTIME_ROOT,
    persist_dataset_artifact,
    register_dataset_artifact,
)
from utils.message_window import window_messages

from ...state import AgentState, MetaKeys
from ...state_views import get_conversation_events
from ...workflow_config import DB_RAG_RECENT_TURNS
from ...conversation_events import (
    append_conversation_event,
    build_assistant_event,
    build_clarification_event,
    build_error_event,
    build_review_request_event,
    build_sql_event,
    store_thread_artifact,
)
from ...memory import complete_task, latest_user_intent, link_user_intent_completed_task
from ..state_helpers import clear_clarification_meta

_SUPPORTED_PROVIDERS = {"openai", "anthropic"}
_NON_INFORMATIVE_FOLLOWUPS = {
    "k",
    "kk",
    "ok",
    "okay",
    "thanks",
    "thank you",
    "thx",
    "got it",
    "sounds good",
    "please continue",
    "continue",
}
_EXTRACTION_OPT_IN_PROMPT = (
    "Would you like me to identify the tables and columns needed for a data extraction "
    "from this database question?"
)
_POPULATION_SCOPE_LABELS = {"specified", "ambiguous", "not_population_scoped", "unknown"}
_POPULATION_SCOPE_POPULATIONS = {"index_case", "household_contact", "both"}
_POPULATION_SCOPE_COLUMN_WARNING = (
    "Warning: This database contains both index cases and household contacts. "
    "Your question did not specify which population to use, so please verify that the selected columns "
    "match your intended population."
)
_POPULATION_SCOPE_METADATA_WARNING = (
    "Warning: This database contains both index cases and household contacts. "
    "Your question did not specify which population to use, so please verify that the answer "
    "matches your intended population."
)


def _read_value(payload: Any, field: str, default: Any = None) -> Any:
    if isinstance(payload, dict):
        return payload.get(field, default)
    return getattr(payload, field, default)


def _normalize_population_scope(value: Any) -> dict[str, Any]:
    raw = dict(value) if isinstance(value, dict) else {}
    label = str(raw.get("label") or "").strip().lower()
    if label not in _POPULATION_SCOPE_LABELS:
        label = "unknown"
    population = str(raw.get("population") or "").strip().lower() or None
    if population not in _POPULATION_SCOPE_POPULATIONS:
        population = None
    try:
        confidence = float(raw.get("confidence", 0.0) or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    reason = str(raw.get("reason") or "").strip()
    warning = str(raw.get("warning") or "").strip()
    return {
        "label": label,
        "population": population,
        "warning": warning,
        "confidence": confidence,
        "reason": reason,
    }


def _population_scope_warning(scope: Any, *, target: str) -> str:
    normalized = _normalize_population_scope(scope)
    if normalized.get("label") != "ambiguous":
        return ""
    if target == "metadata":
        return _POPULATION_SCOPE_METADATA_WARNING
    return _POPULATION_SCOPE_COLUMN_WARNING


def _with_population_scope_warning(text: str, scope: Any, *, target: str) -> str:
    warning = _population_scope_warning(scope, target=target)
    body = str(text or "").strip()
    if not warning:
        return body
    if body.startswith(warning):
        return body
    return f"{warning}\n\n{body}" if body else warning


def _string_list(values: Any) -> list[str]:
    result: list[str] = []
    for value in list(values or []):
        text = str(value or "").strip()
        if text:
            result.append(text)
    return result


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def _structured_column_refs(values: Any, *, allow_strings: bool = False) -> list[Any]:
    result: list[Any] = []
    for value in list(values or []):
        entry = _coerce_mapping(value)
        if entry:
            table = str(entry.get("table") or "").strip()
            column = str(entry.get("column") or entry.get("field") or "").strip()
            if not table or not column:
                continue
            normalized: dict[str, Any] = {"table": table, "column": column}
            semantic = str(entry.get("semantic") or "").strip()
            if semantic:
                normalized["semantic"] = semantic
            if normalized not in result:
                result.append(normalized)
            continue
        if allow_strings:
            text = str(value or "").strip()
            if text and text not in result:
                result.append(text)
    return result


def _structured_filter_refs(values: Any) -> list[Any]:
    result: list[Any] = []
    for value in list(values or []):
        entry = _coerce_mapping(value)
        if entry:
            table = str(entry.get("table") or "").strip()
            column = str(entry.get("column") or entry.get("field") or "").strip()
            if not table or not column:
                continue
            normalized: dict[str, Any] = {"table": table, "column": column}
            for key in ("operator", "value", "semantic"):
                if key in entry and str(entry.get(key) or "").strip():
                    normalized[key] = str(entry.get(key) or "").strip()
            if normalized not in result:
                result.append(normalized)
            continue
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def _table_names_from_context(context: Any) -> list[str]:
    direct_names = _read_value(context, "table_names")
    if direct_names:
        return _string_list(direct_names)

    tables: list[str] = []
    for entry in list(_read_value(context, "tables", []) or []):
        table = str(_read_value(entry, "table", "") or "").strip()
        if table:
            tables.append(table)
    return tables


def _column_names_from_context(context: Any) -> list[str]:
    direct_names = _read_value(context, "column_names")
    if direct_names:
        return _string_list(direct_names)

    columns: list[str] = []
    for entry in list(_read_value(context, "columns", []) or []):
        column = str(_read_value(entry, "column", "") or "").strip()
        if column:
            columns.append(column)
    return columns


def _serialize_columns(columns: Any) -> list[dict[str, str]]:
    serialized: list[dict[str, str]] = []
    for column in list(columns or []):
        table = str(_read_value(column, "table", "") or "").strip()
        column_name = str(_read_value(column, "column", "") or "").strip()
        description = str(_read_value(column, "description", "") or "").strip()
        if not table or not column_name:
            continue
        serialized.append(
            {
                "table": table,
                "column": column_name,
                "description": description,
            }
        )
    return serialized


def _artifact_content(state: AgentState, artifact_id: str | None) -> dict[str, Any]:
    artifact_key = str(artifact_id or "").strip()
    if not artifact_key:
        return {}
    artifacts = dict(state.get("artifacts") or {})
    files = dict(artifacts.get("files") or {})
    artifact = dict(files.get(artifact_key) or {})
    content = artifact.get("content")
    return dict(content) if isinstance(content, dict) else {}


def _completed_sql_task_texts(
    candidate: Any,
    *,
    sql_candidate_artifact: dict[str, Any],
    approved_selection: dict[str, Any],
) -> tuple[str, str]:
    candidate_source = str(_read_value(candidate, "source_question", "") or "").strip()
    candidate_question = str(_read_value(candidate, "question", "") or "").strip()
    candidate_goal = str(_read_value(candidate, "goal_text", "") or "").strip()
    sql_source = str(sql_candidate_artifact.get("source_question") or "").strip()
    sql_goal = str(sql_candidate_artifact.get("goal_text") or "").strip()
    selection_source = str(approved_selection.get("source_question") or "").strip()
    selection_goal = str(approved_selection.get("goal_text") or "").strip()

    source_question = candidate_source or candidate_question or sql_source or selection_source
    goal_text = candidate_goal or candidate_source or candidate_question or sql_goal or selection_goal or selection_source
    return source_question, goal_text


def _intent_id_from_snapshot(payload: dict[str, Any]) -> str:
    snapshot = payload.get("intent_snapshot")
    if not isinstance(snapshot, dict):
        return ""
    return str(snapshot.get("intent_id") or "").strip()


def _originating_user_intent_id(
    state: AgentState,
    candidate: Any,
    *,
    sql_candidate_artifact: dict[str, Any],
    approved_selection: dict[str, Any],
) -> str | None:
    candidate_intent_id = (
        str(_read_value(candidate, "intent_id", "") or "").strip()
        or _intent_id_from_snapshot(sql_candidate_artifact)
        or _intent_id_from_snapshot(approved_selection)
    )
    if not candidate_intent_id:
        return None

    originating_intent = latest_user_intent(state, kind="db_rag_query")
    if not isinstance(originating_intent, dict):
        return None
    if originating_intent.get("active_intent_id") != candidate_intent_id:
        return None
    intent_id = originating_intent.get("intent_id")
    return str(intent_id).strip() if isinstance(intent_id, str) and intent_id.strip() else None


def _serialize_column_selection(selection: Any) -> dict[str, Any]:
    return {
        "selection_id": str(_read_value(selection, "selection_id", "") or "").strip(),
        "question": str(_read_value(selection, "question", "") or "").strip(),
        "tables": _string_list(_read_value(selection, "tables", [])),
        "columns": _serialize_columns(_read_value(selection, "columns", [])),
        "rationale": str(_read_value(selection, "rationale", "") or "").strip(),
        "feedback_history": list(_read_value(selection, "feedback_history", []) or []),
        "status": str(_read_value(selection, "status", "awaiting_review") or "awaiting_review").strip(),
        "selection_source": str(_read_value(selection, "selection_source", "legacy") or "legacy").strip(),
        "fallback_reason": str(_read_value(selection, "fallback_reason", "") or "").strip(),
        "raw_model_output": str(_read_value(selection, "raw_model_output", "") or "").strip(),
        "population_scope": _normalize_population_scope(_read_value(selection, "population_scope", {})),
    }


def _deserialize_column_selection(payload: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        selection_id=str(payload.get("selection_id") or "").strip(),
        question=str(payload.get("question") or "").strip(),
        tables=_string_list(payload.get("tables", [])),
        columns=_serialize_columns(payload.get("columns", [])),
        rationale=str(payload.get("rationale") or "").strip(),
        feedback_history=list(payload.get("feedback_history") or []),
        status=str(payload.get("status") or "awaiting_review").strip(),
        selection_source=str(payload.get("selection_source") or "legacy").strip(),
        fallback_reason=str(payload.get("fallback_reason") or "").strip(),
        raw_model_output=str(payload.get("raw_model_output") or "").strip(),
        population_scope=_normalize_population_scope(payload.get("population_scope") or {}),
    )


def _serialize_prepared_sql_candidate(candidate: Any) -> dict[str, Any]:
    return {
        "question": str(_read_value(candidate, "question", "") or "").strip(),
        "sql": str(_read_value(candidate, "sql", "") or "").strip(),
        "tables": _string_list(_read_value(candidate, "tables", [])),
        "columns": _serialize_columns(_read_value(candidate, "columns", [])),
        "selection_id": str(_read_value(candidate, "selection_id", "") or "").strip(),
        "status": str(_read_value(candidate, "status", "prepared") or "prepared").strip(),
        "population_scope": _normalize_population_scope(_read_value(candidate, "population_scope", {})),
    }


def _deserialize_prepared_sql_candidate(payload: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        question=str(payload.get("question") or "").strip(),
        sql=str(payload.get("sql") or "").strip(),
        tables=_string_list(payload.get("tables", [])),
        columns=_serialize_columns(payload.get("columns", [])),
        selection_id=str(payload.get("selection_id") or "").strip(),
        status=str(payload.get("status") or "prepared").strip(),
        population_scope=_normalize_population_scope(payload.get("population_scope") or {}),
    )


def _serialize_context_summary(context: Any) -> dict[str, list[str]]:
    return {
        "tables": _table_names_from_context(context),
        "columns": _column_names_from_context(context),
    }


def _serialize_context_pool(context: Any) -> dict[str, list[dict[str, Any]]]:
    tables = []
    for entry in list(_read_value(context, "tables", []) or []):
        table_name = str(_read_value(entry, "table", "") or "").strip()
        text = str(_read_value(entry, "text", "") or "").strip()
        if table_name:
            tables.append({"table": table_name, "text": text})

    columns = []
    for entry in list(_read_value(context, "columns", []) or []):
        table_name = str(_read_value(entry, "table", "") or "").strip()
        column_name = str(_read_value(entry, "column", "") or "").strip()
        text = str(_read_value(entry, "text", "") or "").strip()
        score = _read_value(entry, "score", None)
        if table_name and column_name:
            payload = {"table": table_name, "column": column_name, "text": text}
            if isinstance(score, (int, float)):
                payload["score"] = float(score)
            columns.append(payload)

    return {"tables": tables, "columns": columns}


def _serialize_intent(intent: Any) -> dict[str, Any]:
    payload = dict(intent) if isinstance(intent, dict) else getattr(intent, "__dict__", {})
    required_columns = _structured_column_refs(payload.get("required_columns", []), allow_strings=True)
    for column in _structured_column_refs(payload.get("requested_fields", [])):
        if column not in required_columns:
            required_columns.append(column)
    return {
        "intent_id": str(payload.get("intent_id") or "").strip(),
        "source_question": str(payload.get("source_question") or "").strip(),
        "goal_text": str(payload.get("goal_text") or "").strip(),
        "mode": str(payload.get("mode") or "").strip(),
        "population": str(payload.get("population") or "").strip() or None,
        "filters": _structured_filter_refs(payload.get("filters", [])),
        "required_tables": _string_list(payload.get("required_tables", [])),
        "required_columns": required_columns,
        "excluded_tables": _string_list(payload.get("excluded_tables", [])),
        "excluded_columns": _structured_column_refs(payload.get("excluded_columns", []), allow_strings=True),
        "feedback_history": list(payload.get("feedback_history") or []),
        "status": str(payload.get("status") or "active").strip(),
    }


def _intent_snapshot(intent: dict[str, Any]) -> dict[str, Any]:
    return {
        "intent_id": intent["intent_id"],
        "goal_text": intent["goal_text"],
        "population": intent.get("population"),
        "filters": list(intent.get("filters") or []),
        "required_tables": list(intent.get("required_tables") or []),
        "required_columns": list(intent.get("required_columns") or []),
        "excluded_tables": list(intent.get("excluded_tables") or []),
        "excluded_columns": list(intent.get("excluded_columns") or []),
    }


def _bootstrap_active_intent(rag_state: dict[str, Any], question: str) -> dict[str, Any]:
    existing = _serialize_intent(rag_state.get("active_intent") or {})
    if existing.get("goal_text"):
        return existing
    review = dict(rag_state.get("pending_column_review") or {})
    review_question = str(review.get("question") or question).strip()
    return {
        "intent_id": f"intent:{review.get('selection_id') or 'bootstrap'}",
        "source_question": review_question,
        "goal_text": review_question,
        "mode": "extraction",
        "population": None,
        "filters": [],
        "required_tables": [],
        "required_columns": [],
        "excluded_tables": [],
        "excluded_columns": [],
        "feedback_history": list(review.get("feedback_history") or []),
        "status": "active",
    }


def _new_rag_state(*, question: str, intent: dict[str, Any], context_summary: dict[str, list[str]]) -> dict[str, Any]:
    return {
        "thread_status": "idle",
        "active_thread": True,
        "active_intent": intent,
        "pending_extraction_opt_in": None,
        "pending_column_review_artifact_id": None,
        "approved_column_selection_artifact_id": None,
        "pending_sql_candidate_artifact_id": None,
        "last_database_question": question,
        "last_retrieval_context": context_summary,
        "error": None,
    }


def _reset_active_workflow_for_new_question(
    rag_state: dict[str, Any],
    *,
    question: str,
    intent: dict[str, Any],
    context_summary: dict[str, list[str]],
) -> dict[str, Any]:
    del rag_state
    return _new_rag_state(question=question, intent=intent, context_summary=context_summary)


def _build_pending_extraction_opt_in(intent: dict[str, Any]) -> dict[str, Any]:
    goal_text = str(intent["goal_text"] or "").strip()
    prompt = (
        f'For your request: "{goal_text}"\n\n{_EXTRACTION_OPT_IN_PROMPT}'
        if goal_text
        else _EXTRACTION_OPT_IN_PROMPT
    )
    return {
        "status": "awaiting_reply",
        "prompt": prompt,
        "intent_id": intent["intent_id"],
        "goal_text": goal_text,
    }


def _build_subset_dataset_id() -> str:
    return f"subset-{uuid4().hex[:8]}"


def _projection_aliases_to_selected_columns(
    sql: str,
    selected_columns: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    sql_text = str(sql or "").strip()
    if not sql_text:
        return {}

    selected_by_pair = {
        (str(col.get("table") or "").strip(), str(col.get("column") or "").strip()): dict(col)
        for col in list(selected_columns or [])
        if str(col.get("column") or "").strip()
    }
    selected_by_column: dict[str, list[dict[str, str]]] = {}
    for col in list(selected_columns or []):
        col_name = str(col.get("column") or "").strip()
        if not col_name:
            continue
        selected_by_column.setdefault(col_name, []).append(dict(col))

    def _match_selected(source_table: str, source_name: str) -> dict[str, str] | None:
        matched = None
        if source_table:
            matched = selected_by_pair.get((source_table, source_name))
        if matched is None:
            candidates = selected_by_column.get(source_name) or []
            if len(candidates) == 1:
                matched = candidates[0]
        return dict(matched) if matched is not None else None

    try:
        import sqlglot
        from sqlglot import exp

        expression = sqlglot.parse_one(sql_text, dialect="duckdb")
    except Exception:
        alias_map: dict[str, dict[str, str]] = {}
        # Fallback for environments without sqlglot: extract simple `source AS alias` projections.
        # Handles sources like `IC_AGE`, `form.IC_AGE`, `"IC_AGE"`, `"form"."IC_AGE"`.
        for source_table_token, source_col_token, output_alias_token in re.findall(
            r'(?is)(?:"?([A-Za-z_][A-Za-z0-9_]*)"?\.)?"?([A-Za-z_][A-Za-z0-9_]*)"?\s+AS\s+"?([A-Za-z_][A-Za-z0-9_]*)"?',
            sql_text,
        ):
            source_table = str(source_table_token or "").strip()
            source_name = str(source_col_token or "").strip()
            output_name = str(output_alias_token or "").strip()
            if not source_name or not output_name:
                continue
            matched = _match_selected(source_table, source_name)
            if matched is not None:
                alias_map[output_name] = matched
        return alias_map

    alias_map: dict[str, dict[str, str]] = {}
    select = expression.find(exp.Select)
    if select is None:
        return alias_map

    for projection in list(select.expressions or []):
        output_name = str(getattr(projection, "alias_or_name", "") or "").strip()
        if not output_name:
            continue

        source_column = None
        for column_ref in projection.find_all(exp.Column):
            source_column = column_ref
            break
        if source_column is None:
            continue

        source_table = str(getattr(source_column, "table", "") or "").strip()
        source_name = str(getattr(source_column, "name", "") or "").strip()
        if not source_name:
            continue

        matched = _match_selected(source_table, source_name)
        if matched is not None:
            alias_map[output_name] = dict(matched)

    return alias_map


def _selected_columns_by_output_position(
    dataframe: Any,
    selected_columns: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    output_columns = [str(name or "").strip() for name in list(getattr(dataframe, "columns", []))]
    approved_columns = [
        dict(column)
        for column in list(selected_columns or [])
        if str(column.get("column") or "").strip()
    ]
    # Conservative fallback: only infer by position when the projection cardinality matches exactly.
    if not output_columns or len(output_columns) != len(approved_columns):
        return {}
    return {
        output_columns[idx]: approved_columns[idx]
        for idx in range(len(output_columns))
        if output_columns[idx]
    }


def _build_subset_schema(
    dataframe: Any,
    selected_columns: list[dict[str, str]],
    *,
    sql: str = "",
) -> dict[str, dict[str, Any]]:
    selected_by_name = {
        str(column.get("column") or "").strip(): column
        for column in list(selected_columns or [])
        if str(column.get("column") or "").strip()
    }
    selected_by_alias = _projection_aliases_to_selected_columns(sql, selected_columns)
    selected_by_position = _selected_columns_by_output_position(dataframe, selected_columns)

    schema: dict[str, dict[str, Any]] = {}
    dtypes = getattr(dataframe, "dtypes", None)
    for column_name in list(getattr(dataframe, "columns", [])):
        column_key = str(column_name)
        meta: dict[str, Any] = {}
        selected = (
            selected_by_name.get(column_key)
            or selected_by_alias.get(column_key)
            or selected_by_position.get(column_key)
            or {}
        )
        table_name = str(selected.get("table") or "").strip()
        reviewed_meta = _lookup_schema_variable_metadata(table_name, column_key)
        if reviewed_meta is None:
            selected_column_name = str(selected.get("column") or "").strip()
            if selected_column_name:
                reviewed_meta = _lookup_schema_variable_metadata(table_name, selected_column_name)
        description = str(
            (reviewed_meta or {}).get("description") or selected.get("description") or ""
        ).strip()
        if description:
            meta["description"] = description
        for field in ("values", "depends_on", "condition", "section_context"):
            value = (reviewed_meta or {}).get(field)
            if value is not None and value != "":
                meta[field] = value
        if dtypes is not None:
            meta["dataType"] = str(dtypes[column_name])
        schema[column_key] = meta
    return schema


def _persist_sql_subset_artifact(
    state: AgentState,
    candidate: Any,
    approved_selection: dict[str, Any],
    execution_result: Any,
    *,
    selection_artifact_id: str | None,
    sql_candidate_artifact_id: str | None,
) -> tuple[AgentState, dict[str, Any]]:
    meta = dict(state.get("meta") or {})
    thread_id = str(meta.get(MetaKeys.THREAD_ID) or "").strip()
    if not thread_id:
        raise ValueError("DB-RAG SQL subset persistence requires a thread_id in state meta.")

    selected_tables = _string_list(_read_value(approved_selection, "tables", []) or _read_value(candidate, "tables", []))
    selected_columns = _serialize_columns(
        _read_value(approved_selection, "columns", []) or _read_value(candidate, "columns", [])
    )
    artifact = persist_dataset_artifact(
        runtime_root=DEFAULT_RUNTIME_ROOT,
        thread_id=thread_id,
        dataset_id=_build_subset_dataset_id(),
        kind="subset",
        dataframe=_read_value(execution_result, "dataframe"),
        schema=_build_subset_schema(
            _read_value(execution_result, "dataframe"),
            selected_columns,
            sql=str(_read_value(execution_result, "sql", _read_value(candidate, "sql", "")) or "").strip(),
        ),
        provenance={
            "source": "db_rag_sql",
            "source_question": str(_read_value(candidate, "source_question", _read_value(candidate, "question", "")) or "").strip(),
            "goal_text": str(_read_value(candidate, "goal_text", _read_value(candidate, "question", "")) or "").strip(),
            "sql": str(_read_value(execution_result, "sql", _read_value(candidate, "sql", "")) or "").strip(),
            "source_tables": _string_list(
                _read_value(execution_result, "source_tables", []) or _read_value(candidate, "tables", [])
            ),
            "selected_tables": selected_tables,
            "selected_columns": selected_columns,
            "selection_id": str(
                _read_value(approved_selection, "selection_id", _read_value(candidate, "selection_id", "")) or ""
            ).strip(),
            "selection_artifact_id": str(selection_artifact_id or "").strip(),
            "sql_candidate_artifact_id": str(sql_candidate_artifact_id or "").strip(),
            "feedback_history": list(_read_value(approved_selection, "feedback_history", []) or []),
        },
    )
    return register_dataset_artifact(state, artifact, make_active=True), artifact


def _replace_rag_state(state: AgentState, rag_state: dict[str, Any]) -> AgentState:
    agents = dict(state.get("agents") or {})
    agents["rag_db_qa"] = rag_state
    return {
        **state,
        "agents": agents,
    }


def _current_user_turn_hash(state: AgentState) -> str | None:
    meta = dict(state.get("meta") or {})
    return str(meta.get(MetaKeys.LAST_USER_MESSAGE_HASH) or "").strip() or None


def _append_ai_response(state: AgentState, text: str) -> AgentState:
    messages = list(state.get("messages", []))
    messages.append(AIMessage(content=text))
    output = dict(state.get("output") or {})
    output["qa_response"] = text
    observations = list(state.get("observations", []))
    observations.append("rag_db_qa: responded to database question")
    return {
        **state,
        "messages": messages,
        "output": output,
        "observations": observations,
    }


def _append_assistant_event(state: AgentState, text: str) -> AgentState:
    return append_conversation_event(
        state,
        build_assistant_event(
            actor="rag_db_qa",
            user_turn_hash=_current_user_turn_hash(state),
            text=text,
            status="done",
        ),
    )


def _append_clarification_event(state: AgentState, text: str) -> AgentState:
    return append_conversation_event(
        state,
        build_clarification_event(
            actor="rag_db_qa",
            user_turn_hash=_current_user_turn_hash(state),
            text=text,
            status="active",
        ),
    )


def _store_column_selection_artifact(
    state: AgentState,
    *,
    selection: dict[str, Any],
    source_question: str,
    goal_text: str,
    intent_snapshot: dict[str, Any],
    retrieval_summary: dict[str, list[str]],
    retrieval_pool: dict[str, list[dict[str, Any]]],
    summary: str,
) -> tuple[AgentState, str]:
    updated_state = store_thread_artifact(
        state,
        {
            "kind": "db_rag_column_selection",
            "producer": "rag_db_qa",
            "mime": "application/json",
            "summary": summary,
            "content": {
                "selection_id": selection["selection_id"],
                "source_question": source_question,
                "goal_text": goal_text,
                "intent_snapshot": intent_snapshot,
                "retrieval_summary": retrieval_summary,
                "retrieval_pool": retrieval_pool,
                "tables": list(selection["tables"]),
                "columns": list(selection["columns"]),
                "rationale": selection["rationale"],
                "feedback_history": list(selection.get("feedback_history") or []),
                "status": selection.get("status", "awaiting_review"),
                "selection_source": selection.get("selection_source", "legacy"),
                "fallback_reason": selection.get("fallback_reason", ""),
                "raw_model_output": selection.get("raw_model_output", ""),
                "population_scope": _normalize_population_scope(selection.get("population_scope") or {}),
                "review_prompt": str(selection.get("review_prompt") or ""),
            },
        },
    )
    artifact_id = next(reversed(dict(updated_state.get("artifacts") or {}).get("files") or {}))
    return updated_state, artifact_id


def _store_sql_candidate_artifact(
    state: AgentState,
    *,
    candidate: dict[str, Any],
    selection_artifact_id: str,
    source_question: str,
    goal_text: str,
    intent_snapshot: dict[str, Any],
    population_scope: dict[str, Any] | None = None,
) -> tuple[AgentState, str]:
    updated_state = store_thread_artifact(
        state,
        {
            "kind": "db_rag_sql_candidate",
            "producer": "rag_db_qa",
            "mime": "application/json",
            "summary": "Prepared read-only SQL candidate for DB-RAG review.",
            "content": {
                "sql_candidate_id": candidate.get("sql_candidate_id") or f"sql:{candidate['selection_id']}",
                "selection_artifact_id": selection_artifact_id,
                "source_question": source_question,
                "goal_text": goal_text,
                "intent_snapshot": intent_snapshot,
                "tables": list(candidate["tables"]),
                "columns": list(candidate["columns"]),
                "sql": candidate["sql"],
                "status": candidate.get("status", "prepared"),
                "population_scope": _normalize_population_scope(
                    population_scope or candidate.get("population_scope") or {}
                ),
            },
        },
    )
    artifact_id = next(reversed(dict(updated_state.get("artifacts") or {}).get("files") or {}))
    return updated_state, artifact_id


def _append_sql_candidate_events(state: AgentState, candidate: dict[str, Any], artifact_id: str) -> AgentState:
    updated_state = append_conversation_event(
        state,
        build_sql_event(
            actor="rag_db_qa",
            user_turn_hash=_current_user_turn_hash(state),
            artifact_id=artifact_id,
            text="Prepared read-only SQL candidate.",
            status="done",
        ),
    )
    return append_conversation_event(
        updated_state,
        build_review_request_event(
            actor="rag_db_qa",
            user_turn_hash=_current_user_turn_hash(updated_state),
            review_kind="rag_db_sql_execution",
            text="Prepared SQL is awaiting explicit human review before execution.",
            artifact_id=artifact_id,
            status="active",
        ),
    )


def _append_column_review_request_event(state: AgentState, *, artifact_id: str, text: str) -> AgentState:
    return append_conversation_event(
        state,
        build_review_request_event(
            actor="rag_db_qa",
            user_turn_hash=_current_user_turn_hash(state),
            review_kind="rag_db_column_selection",
            text=text,
            artifact_id=artifact_id,
            status="active",
        ),
    )


def _clear_output_error(state: AgentState) -> AgentState:
    output = dict(state.get("output") or {})
    output.pop("error", None)
    return {
        **state,
        "output": output,
    }


def _append_sql_error_response(state: AgentState, error_payload: dict[str, str]) -> AgentState:
    message = str(error_payload.get("message") or "Unknown DB-RAG SQL error.")
    response_text = (
        "DB-RAG SQL execution failed and needs revision before it can continue.\n\n"
        f"Details: {error_payload.get('type')}: {message}"
    )
    messages = list(state.get("messages", []))
    messages.append(AIMessage(content=response_text))
    output = dict(state.get("output") or {})
    output["qa_response"] = response_text
    output["error"] = error_payload
    observations = list(state.get("observations", []))
    observations.append("rag_db_qa: sql execution failed")
    updated = {
        **state,
        "messages": messages,
        "output": output,
        "observations": observations,
    }
    updated = _append_assistant_event(updated, response_text)
    return append_conversation_event(
        updated,
        build_error_event(
            actor="rag_db_qa",
            user_turn_hash=_current_user_turn_hash(updated),
            text=response_text,
            error=error_payload,
            status="error",
        ),
    )


def _store_sql_candidate_output(state: AgentState, candidate: dict[str, Any]) -> AgentState:
    output = dict(state.get("output") or {})
    output["generated_sql"] = str(candidate.get("sql") or "").strip()
    output["prepared_sql_candidate"] = dict(candidate)
    return {
        **state,
        "output": output,
    }


def _execute_prepared_sql_candidate(
    state: AgentState,
    rag_state: dict[str, Any],
    candidate: Any,
    service,
) -> AgentState:
    active_task = dict(rag_state.get("active_task") or {})
    approved_review = dict(rag_state.get("pending_column_review") or {})
    selection_artifact_id = str(
        rag_state.get("approved_column_selection_artifact_id")
        or rag_state.get("pending_column_review_artifact_id")
        or ""
    ).strip()
    sql_candidate_artifact_id = str(rag_state.get("pending_sql_candidate_artifact_id") or "").strip()
    approved_sql_review_artifact_id = str(rag_state.get("sql_review_approved_artifact_id") or "").strip()
    approved_selection = _artifact_content(state, selection_artifact_id)
    sql_candidate_artifact = _artifact_content(state, sql_candidate_artifact_id)
    review_snapshot = dict(approved_selection or approved_review)
    try:
        if not approved_selection:
            raise ValueError("Approved column-selection artifact is missing during SQL execution.")
        if not sql_candidate_artifact_id or approved_sql_review_artifact_id != sql_candidate_artifact_id:
            raise PermissionError("SQL execution requires explicit human approval of the current SQL candidate.")

        execution_result = service.execute_prepared_sql(candidate)

        persisted_state, artifact = _persist_sql_subset_artifact(
            state,
            candidate,
            approved_selection,
            execution_result,
            selection_artifact_id=selection_artifact_id or None,
            sql_candidate_artifact_id=sql_candidate_artifact_id or None,
        )
    except Exception as exc:
        error_payload = {
            "category": "db_rag_sql",
            "type": "MissingSqlReviewApproval" if isinstance(exc, PermissionError) else type(exc).__name__,
            "message": str(exc),
        }
        updated = _append_sql_error_response(state, error_payload)
        updated["meta"] = clear_clarification_meta(updated.get("meta") or {})
        rag_state.pop("pending_sql_candidate", None)
        review_snapshot["status"] = "needs_revision"
        feedback_history = list(review_snapshot.get("feedback_history") or [])
        feedback_history.append(
            {
                "action": "sql_execution_error",
                "feedback": str(exc),
            }
        )
        review_snapshot["feedback_history"] = feedback_history
        rag_state["pending_column_review"] = review_snapshot
        rag_state["error"] = error_payload
        rag_state["last_database_question"] = candidate.question
        rag_state["thread_status"] = "error"
        rag_state.pop("sql_review_approved_artifact_id", None)
        return _finalize_rag_state(updated, rag_state, status="error", active_thread=True)

    reviewed_tables = _format_tables(
        _string_list(_read_value(approved_selection, "tables", []) or _read_value(candidate, "tables", []))
    )
    reviewed_columns = _format_columns(
        _serialize_columns(_read_value(approved_selection, "columns", []) or _read_value(candidate, "columns", []))
    )
    response_text = (
        f'{_read_value(execution_result, "answer", "Read-only SQL execution completed.")}\n\n'
        f"Reviewed tables:\n{reviewed_tables}\n\n"
        f"Reviewed columns:\n{reviewed_columns}\n\n"
        f'SQL used:\n{_format_sql_block(str(_read_value(execution_result, "sql", "") or ""))}\n\n'
        f'Saved dataset id: {artifact["id"]}'
    )
    response_text = _with_population_scope_warning(
        response_text,
        sql_candidate_artifact.get("population_scope") or approved_selection.get("population_scope") or {},
        target="columns",
    )
    updated = _append_ai_response(persisted_state, response_text)
    updated = _append_assistant_event(updated, response_text)
    updated = _store_sql_candidate_output(updated, _serialize_prepared_sql_candidate(candidate))
    updated = _clear_output_error(updated)
    updated["meta"] = clear_clarification_meta(updated.get("meta") or {})
    source_question, goal_text = _completed_sql_task_texts(
        candidate,
        sql_candidate_artifact=sql_candidate_artifact,
        approved_selection=approved_selection,
    )
    originating_user_intent_id = _originating_user_intent_id(
        updated,
        candidate,
        sql_candidate_artifact=sql_candidate_artifact,
        approved_selection=approved_selection,
    )
    provenance = {
        "producer_node": "rag_db_qa",
        "selection_id": str(getattr(candidate, "selection_id", "") or ""),
    }
    if originating_user_intent_id:
        provenance["originating_user_intent_id"] = originating_user_intent_id
    updated = complete_task(
        updated,
        kind="db_rag_sql_extraction",
        source_question=source_question,
        goal_text=goal_text,
        label=f"DB-RAG SQL extraction: {source_question.strip()[:80]}",
        summary=f"Reviewed SQL executed and saved dataset {artifact['id']}.",
        artifact_refs={
            "selection_artifact_id": selection_artifact_id,
            "sql_candidate_artifact_id": sql_candidate_artifact_id,
            "dataset_artifact_id": artifact["id"],
        },
        parent_task_id=active_task.get("parent_task_id"),
        relationship_to_parent=active_task.get("relationship_to_parent"),
        provenance=provenance,
    )
    if originating_user_intent_id:
        task_id = updated["memory"]["last_task_id_by_kind"].get("db_rag_sql_extraction")
        if isinstance(task_id, str):
            updated = link_user_intent_completed_task(
                updated,
                intent_id=originating_user_intent_id,
                task_id=task_id,
            )
    rag_state.pop("pending_sql_candidate", None)
    rag_state.pop("pending_column_review", None)
    rag_state.pop("active_task", None)
    rag_state.pop("sql_review_approved_artifact_id", None)
    rag_state["pending_sql_candidate_artifact_id"] = None
    rag_state["pending_column_review_artifact_id"] = None
    rag_state["approved_column_selection_artifact_id"] = None
    rag_state["error"] = None
    rag_state["last_database_question"] = candidate.question
    rag_state["thread_status"] = "completed"
    return _finalize_rag_state(updated, rag_state, status="done", active_thread=False)


def _format_tables(tables: list[str]) -> str:
    if not tables:
        return "none"
    return "\n".join(f"- {table}" for table in tables)


def _format_columns(columns: list[dict[str, str]]) -> str:
    if not columns:
        return "none"
    lines: list[str] = []
    for column in columns:
        label = f'- {column["table"]}.{column["column"]}'
        description = str(column.get("description") or "").strip()
        if description:
            label = f"{label}: {description}"
        lines.append(label)
    return "\n".join(lines)


def _format_sql_block(sql: str) -> str:
    sql_text = str(sql or "").strip()
    if not sql_text:
        return "```sql\n-- none\n```"
    return f"```sql\n{sql_text}\n```"


def _format_column_review_response(answer_text: str, selection: dict[str, Any], *, revised: bool = False) -> str:
    if revised:
        return _with_population_scope_warning(
            "I refreshed the DB-RAG column selection based on your feedback.\n\nPlease review the updated selection in the panel below.",
            selection.get("population_scope") or {},
            target="columns",
        )
    if answer_text:
        text = f"{answer_text}\n\nPlease review the proposed column selection in the panel below."
    else:
        text = "Please review the proposed DB-RAG column selection in the panel below."
    return _with_population_scope_warning(text, selection.get("population_scope") or {}, target="columns")


def _format_sql_candidate_response(candidate: dict[str, Any]) -> str:
    return _with_population_scope_warning(
        "I prepared a read-only SQL candidate from the approved DB-RAG selection.\n\n"
        "Please review the SQL details in the panel below before execution.",
        candidate.get("population_scope") or {},
        target="columns",
    )


def _render_db_rag_recent_turns(state: AgentState, question: str) -> str:
    events = get_conversation_events(state)
    if events:
        lines: list[str] = []
        for event in events[-(DB_RAG_RECENT_TURNS * 4):]:
            event_type = str(event.get("type") or "")
            text = str(event.get("text") or "").strip()
            if not text:
                continue
            if event_type == "user":
                speaker = "User"
            elif event_type in {"assistant", "clarification"}:
                speaker = "Assistant"
            elif event_type == "review_decision":
                speaker = "User review"
            else:
                continue
            lines.append(f"{speaker}: {text}")
        if lines:
            transcript = "\n".join(lines)
            return f"Latest user request:\n{question}\n\nRecent conversation:\n{transcript}"

    messages = list(state.get("messages", []))
    windowed = window_messages(messages, max_turns=DB_RAG_RECENT_TURNS)
    lines: list[str] = []
    for message in windowed:
        role = getattr(message, "type", None)
        if role not in {"human", "ai"}:
            continue
        content = str(getattr(message, "content", "") or "").strip()
        if not content:
            continue
        speaker = "User" if role == "human" else "Assistant"
        lines.append(f"{speaker}: {content}")
    transcript = "\n".join(lines) if lines else "none"
    return f"Latest user request:\n{question}\n\nRecent conversation:\n{transcript}"


def _finalize_rag_state(
    state: AgentState,
    rag_state: dict[str, Any],
    *,
    status: str,
    active_thread: bool,
) -> AgentState:
    rag_state["status"] = status
    rag_state["active_thread"] = active_thread
    return _replace_rag_state(state, rag_state)
