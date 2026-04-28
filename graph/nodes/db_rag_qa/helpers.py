from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import uuid4
import re

from langchain_core.messages import AIMessage
from utils.dataset_artifacts import (
    DEFAULT_RUNTIME_ROOT,
    persist_dataset_artifact,
    register_dataset_artifact,
)
from utils.message_window import window_messages

from ...state import AgentState, MetaKeys
from ...workflow_config import DB_RAG_RECENT_TURNS, MAX_ERROR_ITERATIONS
from ..state_helpers import clear_clarification_meta

_SUPPORTED_PROVIDERS = {"openai", "anthropic"}
_AFFIRMATIVE_REPLIES = {
    "y",
    "yes",
    "yeah",
    "yep",
    "sure",
    "ok",
    "okay",
    "please do",
    "do it",
    "go ahead",
}
_NEGATIVE_REPLIES = {
    "n",
    "no",
    "nope",
    "nah",
    "not now",
    "don't",
    "do not",
}
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


def _looks_like_substantive_db_followup(text: str) -> bool:
    normalized = " ".join(str(text or "").strip().lower().split())
    if not normalized:
        return False

    tokens = normalized.split()
    if len(tokens) <= 2 and normalized in {"maybe", "not sure", "unsure"}:
        return False

    if re.search(r"\b[A-Z]{2,}_[A-Z0-9_]+\b", str(text or "")):
        return True

    db_cues = {
        "form",
        "variable",
        "column",
        "table",
        "field",
        "subset",
        "cohort",
        "index",
        "outcome",
        "join",
        "filter",
    }
    if any(token in db_cues for token in tokens):
        return True

    return len(tokens) >= 6
def _read_value(payload: Any, field: str, default: Any = None) -> Any:
    if isinstance(payload, dict):
        return payload.get(field, default)
    return getattr(payload, field, default)


def _string_list(values: Any) -> list[str]:
    result: list[str] = []
    for value in list(values or []):
        text = str(value or "").strip()
        if text:
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


def _serialize_column_selection(selection: Any) -> dict[str, Any]:
    return {
        "selection_id": str(_read_value(selection, "selection_id", "") or "").strip(),
        "question": str(_read_value(selection, "question", "") or "").strip(),
        "tables": _string_list(_read_value(selection, "tables", [])),
        "columns": _serialize_columns(_read_value(selection, "columns", [])),
        "rationale": str(_read_value(selection, "rationale", "") or "").strip(),
        "feedback_history": list(_read_value(selection, "feedback_history", []) or []),
        "status": str(_read_value(selection, "status", "awaiting_review") or "awaiting_review").strip(),
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
    )


def _serialize_prepared_sql_candidate(candidate: Any) -> dict[str, Any]:
    return {
        "question": str(_read_value(candidate, "question", "") or "").strip(),
        "sql": str(_read_value(candidate, "sql", "") or "").strip(),
        "tables": _string_list(_read_value(candidate, "tables", [])),
        "columns": _serialize_columns(_read_value(candidate, "columns", [])),
        "selection_id": str(_read_value(candidate, "selection_id", "") or "").strip(),
        "status": str(_read_value(candidate, "status", "prepared") or "prepared").strip(),
    }


def _deserialize_prepared_sql_candidate(payload: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        question=str(payload.get("question") or "").strip(),
        sql=str(payload.get("sql") or "").strip(),
        tables=_string_list(payload.get("tables", [])),
        columns=_serialize_columns(payload.get("columns", [])),
        selection_id=str(payload.get("selection_id") or "").strip(),
        status=str(payload.get("status") or "prepared").strip(),
    )


def _serialize_context_summary(context: Any) -> dict[str, list[str]]:
    return {
        "tables": _table_names_from_context(context),
        "columns": _column_names_from_context(context),
    }


def _serialize_intent(intent: Any) -> dict[str, Any]:
    payload = dict(intent) if isinstance(intent, dict) else getattr(intent, "__dict__", {})
    return {
        "intent_id": str(payload.get("intent_id") or "").strip(),
        "source_question": str(payload.get("source_question") or "").strip(),
        "goal_text": str(payload.get("goal_text") or "").strip(),
        "mode": str(payload.get("mode") or "").strip(),
        "population": str(payload.get("population") or "").strip() or None,
        "requested_fields": _string_list(payload.get("requested_fields", [])),
        "filters": _string_list(payload.get("filters", [])),
        "required_tables": _string_list(payload.get("required_tables", [])),
        "required_columns": _string_list(payload.get("required_columns", [])),
        "excluded_tables": _string_list(payload.get("excluded_tables", [])),
        "excluded_columns": _string_list(payload.get("excluded_columns", [])),
        "feedback_history": list(payload.get("feedback_history") or []),
        "status": str(payload.get("status") or "active").strip(),
    }


def _intent_snapshot(intent: dict[str, Any]) -> dict[str, Any]:
    return {
        "intent_id": intent["intent_id"],
        "goal_text": intent["goal_text"],
        "population": intent.get("population"),
        "requested_fields": list(intent.get("requested_fields") or []),
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
        "requested_fields": [],
        "filters": [],
        "required_tables": [],
        "required_columns": [],
        "excluded_tables": [],
        "excluded_columns": [],
        "feedback_history": list(review.get("feedback_history") or []),
        "status": "active",
    }


def _build_subset_dataset_id() -> str:
    return f"subset-{uuid4().hex[:8]}"


def _build_subset_schema(dataframe: Any, selected_columns: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    selected_by_name = {
        str(column.get("column") or "").strip(): column
        for column in list(selected_columns or [])
        if str(column.get("column") or "").strip()
    }

    schema: dict[str, dict[str, str]] = {}
    dtypes = getattr(dataframe, "dtypes", None)
    for column_name in list(getattr(dataframe, "columns", [])):
        column_key = str(column_name)
        meta: dict[str, str] = {}
        selected = selected_by_name.get(column_key, {})
        description = str(selected.get("description") or "").strip()
        if description:
            meta["description"] = description
        table_name = str(selected.get("table") or "").strip()
        if table_name:
            meta["notes"] = f"Source table: {table_name}"
        if dtypes is not None:
            meta["dataType"] = str(dtypes[column_name])
        schema[column_key] = meta
    return schema


def _persist_sql_subset_artifact(
    state: AgentState,
    candidate: Any,
    approved_review: Any,
    execution_result: Any,
) -> tuple[AgentState, dict[str, Any]]:
    meta = dict(state.get("meta") or {})
    thread_id = str(meta.get(MetaKeys.THREAD_ID) or "").strip()
    if not thread_id:
        raise ValueError("DB-RAG SQL subset persistence requires a thread_id in state meta.")

    selected_tables = _string_list(_read_value(approved_review, "tables", []) or _read_value(candidate, "tables", []))
    selected_columns = _serialize_columns(
        _read_value(approved_review, "columns", []) or _read_value(candidate, "columns", [])
    )
    artifact = persist_dataset_artifact(
        runtime_root=DEFAULT_RUNTIME_ROOT,
        thread_id=thread_id,
        dataset_id=_build_subset_dataset_id(),
        kind="subset",
        dataframe=_read_value(execution_result, "dataframe"),
        schema=_build_subset_schema(_read_value(execution_result, "dataframe"), selected_columns),
        provenance={
            "source": "db_rag_sql",
            "question": str(_read_value(candidate, "question", "") or "").strip(),
            "sql": str(_read_value(execution_result, "sql", _read_value(candidate, "sql", "")) or "").strip(),
            "source_tables": _string_list(
                _read_value(execution_result, "source_tables", []) or _read_value(candidate, "tables", [])
            ),
            "selected_tables": selected_tables,
            "selected_columns": selected_columns,
            "selection_id": str(
                _read_value(candidate, "selection_id", _read_value(approved_review, "selection_id", "")) or ""
            ).strip(),
            "feedback_history": list(_read_value(approved_review, "feedback_history", []) or []),
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
        "DB-RAG SQL execution failed and the workflow stopped.\n\n"
        f"Details: {error_payload.get('type')}: {message}"
    )
    messages = list(state.get("messages", []))
    messages.append(AIMessage(content=response_text))
    output = dict(state.get("output") or {})
    output["qa_response"] = response_text
    output["error"] = error_payload
    observations = list(state.get("observations", []))
    observations.append("rag_db_qa: sql execution failed")
    return {
        **state,
        "messages": messages,
        "output": output,
        "observations": observations,
    }


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
    approved_review = dict(rag_state.get("pending_column_review") or {})
    active_candidate = candidate
    try:
        for attempt in range(MAX_ERROR_ITERATIONS + 1):
            try:
                execution_result = service.execute_prepared_sql(active_candidate)
                break
            except Exception as exc:
                can_retry = (
                    attempt < MAX_ERROR_ITERATIONS
                    and hasattr(service, "repair_prepared_sql_candidate")
                )
                if not can_retry:
                    raise
                active_candidate = service.repair_prepared_sql_candidate(active_candidate, str(exc))

        persisted_state, artifact = _persist_sql_subset_artifact(state, active_candidate, approved_review, execution_result)
    except Exception as exc:
        error_payload = {
            "category": "db_rag_sql",
            "type": type(exc).__name__,
            "message": str(exc),
        }
        updated = _append_sql_error_response(state, error_payload)
        updated["meta"] = clear_clarification_meta(updated.get("meta") or {})
        rag_state.pop("pending_sql_candidate", None)
        rag_state.pop("pending_column_review", None)
        rag_state["error"] = error_payload
        rag_state["last_database_question"] = candidate.question
        rag_state["thread_status"] = "error"
        return _finalize_rag_state(updated, rag_state, status="error", active_thread=False)

    response_text = (
        f'{_read_value(execution_result, "answer", "Read-only SQL execution completed.")}\n\n'
        f'SQL used:\n{_format_sql_block(str(_read_value(execution_result, "sql", "") or ""))}\n\n'
        f'Saved dataset id: {artifact["id"]}'
    )
    updated = _append_ai_response(persisted_state, response_text)
    updated = _store_sql_candidate_output(updated, _serialize_prepared_sql_candidate(active_candidate))
    updated = _clear_output_error(updated)
    updated["meta"] = clear_clarification_meta(updated.get("meta") or {})
    rag_state.pop("pending_sql_candidate", None)
    rag_state.pop("pending_column_review", None)
    rag_state["error"] = None
    rag_state["last_database_question"] = active_candidate.question
    rag_state["thread_status"] = "completed"
    return _finalize_rag_state(updated, rag_state, status="done", active_thread=True)


def _question_from_stale_qa_followup(state: AgentState, latest_question: str) -> str | None:
    meta = dict(state.get("meta") or {})
    if (
        meta.get(MetaKeys.CLARIFICATION_KIND) != "qa_followup"
        or meta.get(MetaKeys.CLARIFICATION_RETURN_NODE) != "qa"
    ):
        return None
    pending_question = str(meta.get(MetaKeys.PENDING_QUESTION) or "").strip()
    if not pending_question or not latest_question:
        return None
    return f"{pending_question}\n\nUser clarification: {latest_question}"


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
        return (
            "I refreshed the DB-RAG column selection based on your feedback.\n\n"
            "Please review the updated selection in the panel below."
        )
    if answer_text:
        return f"{answer_text}\n\nPlease review the proposed column selection in the panel below."
    return "Please review the proposed DB-RAG column selection in the panel below."


def _format_sql_candidate_response(candidate: dict[str, Any]) -> str:
    return (
        "I prepared a read-only SQL candidate from the approved DB-RAG selection.\n\n"
        "Please review the SQL details in the panel below before execution."
    )


def _classify_opt_in_reply(text: str) -> str:
    normalized = " ".join(str(text or "").strip().lower().split())
    if not normalized:
        return "unknown"
    if normalized in _AFFIRMATIVE_REPLIES:
        return "yes"
    if normalized in _NEGATIVE_REPLIES:
        return "no"
    return "unknown"


def _is_bare_acknowledgement(text: str) -> bool:
    return _classify_opt_in_reply(text) in {"yes", "no"}


def _is_non_informative_followup(text: str) -> bool:
    normalized = " ".join(str(text or "").strip().lower().split())
    if not normalized:
        return True
    if normalized in _NON_INFORMATIVE_FOLLOWUPS:
        return True
    if _classify_opt_in_reply(normalized) in {"yes", "no"}:
        return True
    return False


def _render_db_rag_recent_turns(state: AgentState, question: str) -> str:
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
