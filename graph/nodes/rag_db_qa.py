from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage
from utils.dataset_artifacts import (
    DEFAULT_RUNTIME_ROOT,
    persist_dataset_artifact,
    register_dataset_artifact,
)

from ..state import AgentState, MetaKeys
from .state_helpers import clear_clarification_meta, get_agent_state, set_clarification_meta
from .tool_routing import latest_user_message

NODE_NAME = "rag_db_qa"
NODE_CAPABILITY = (
    "Handle database-grounded questions using retrieval over the local RePORT DB-RAG assets. "
    "Answer metadata questions from retrieved context, pause for human column review when SQL is "
    "needed, prepare read-only SQL only from approved selections, and keep prepared SQL candidates "
    "pending for explicit human review before execution."
)

_SUPPORTED_PROVIDERS = {"openai", "anthropic"}


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
    try:
        execution_result = service.execute_prepared_sql(candidate)
        persisted_state, artifact = _persist_sql_subset_artifact(state, candidate, approved_review, execution_result)
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
    updated = _store_sql_candidate_output(updated, _serialize_prepared_sql_candidate(candidate))
    updated = _clear_output_error(updated)
    updated["meta"] = clear_clarification_meta(updated.get("meta") or {})
    rag_state.pop("pending_sql_candidate", None)
    rag_state.pop("pending_column_review", None)
    rag_state["error"] = None
    rag_state["last_database_question"] = candidate.question
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
    intro = "I refreshed the DB-RAG column selection based on the review feedback." if revised else answer_text
    return (
        f"{intro}\n\n"
        "Selected tables:\n"
        f"{_format_tables(list(selection.get('tables') or []))}\n\n"
        "Selected columns:\n"
        f"{_format_columns(list(selection.get('columns') or []))}\n\n"
        f"Rationale: {selection.get('rationale') or 'No rationale provided.'}\n\n"
        "SQL will be generated only after approval of this column selection."
    )


def _format_sql_candidate_response(candidate: dict[str, Any]) -> str:
    return (
        "I prepared a read-only SQL candidate from the approved DB-RAG selection.\n\n"
        "Approved tables:\n"
        f"{_format_tables(list(candidate.get('tables') or []))}\n\n"
        "Approved columns:\n"
        f"{_format_columns(list(candidate.get('columns') or []))}\n\n"
        "Prepared SQL:\n"
        f"{_format_sql_block(str(candidate.get('sql') or ''))}\n\n"
        "Please review the SQL before execution."
    )


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


def rag_db_qa_node(
    state: AgentState,
    llm,
    *,
    provider: str,
    service,
    reranker_model: str | None = None,
    question_override: str | None = None,
) -> AgentState:
    del llm

    rag_state = dict(get_agent_state(state, "rag_db_qa"))

    latest_question = latest_user_message(state)
    question = str(question_override or _question_from_stale_qa_followup(state, latest_question) or latest_question)

    if provider not in _SUPPORTED_PROVIDERS:
        updated = _append_ai_response(
            state,
            "DB-RAG currently requires an OpenAI or Anthropic provider. Switch the model provider and try again.",
        )
        updated = _clear_output_error(updated)
        updated["meta"] = clear_clarification_meta(updated.get("meta") or {})
        rag_state["error"] = None
        return _finalize_rag_state(updated, rag_state, status="done", active_thread=False)

    readiness = service.readiness() if service is not None else {"ready": False, "message": "DB-RAG service is unavailable."}
    if not readiness.get("ready"):
        updated = _append_ai_response(state, str(readiness.get("message") or "DB-RAG assets are not ready."))
        updated = _clear_output_error(updated)
        updated["meta"] = clear_clarification_meta(updated.get("meta") or {})
        rag_state["error"] = None
        return _finalize_rag_state(updated, rag_state, status="done", active_thread=False)

    pending_sql_candidate = dict(rag_state.get("pending_sql_candidate") or {})
    if pending_sql_candidate and str(pending_sql_candidate.get("status") or "").strip() == "prepared":
        candidate = _deserialize_prepared_sql_candidate(pending_sql_candidate)
        updated = _append_ai_response(state, _format_sql_candidate_response(_serialize_prepared_sql_candidate(candidate)))
        updated = _store_sql_candidate_output(updated, _serialize_prepared_sql_candidate(candidate))
        updated = _clear_output_error(updated)
        updated["meta"] = clear_clarification_meta(updated.get("meta") or {})
        rag_state["error"] = None
        rag_state["last_database_question"] = candidate.question
        rag_state["thread_status"] = "awaiting_sql_review"
        return _finalize_rag_state(updated, rag_state, status="done", active_thread=True)

    pending_column_review = dict(rag_state.get("pending_column_review") or {})
    if pending_column_review.get("status") == "needs_revision":
        active_intent = _bootstrap_active_intent(rag_state, question)
        feedback_history = list(pending_column_review.get("feedback_history") or [])
        revised_intent = _serialize_intent(service.update_intent_from_feedback(active_intent, feedback_history))
        snapshot = _intent_snapshot(revised_intent)
        context = service.retrieve_context_for_intent(SimpleNamespace(**revised_intent), reranker_model=reranker_model)
        selection = service.prepare_column_selection(
            revised_intent["goal_text"],
            context,
            feedback_history=feedback_history,
            previous_selection=pending_column_review,
            intent_snapshot=snapshot,
        )
        review_payload = _serialize_column_selection(selection)
        review_payload["question"] = revised_intent["goal_text"]
        review_payload["goal_text"] = revised_intent["goal_text"]
        review_payload["feedback_history"] = list(review_payload.get("feedback_history") or feedback_history)
        review_payload["status"] = "awaiting_review"

        updated = _append_ai_response(state, _format_column_review_response("", review_payload, revised=True))
        updated = _clear_output_error(updated)
        updated["meta"] = clear_clarification_meta(updated.get("meta") or {})
        rag_state["active_intent"] = revised_intent
        rag_state["intent_snapshot_for_selection"] = snapshot
        rag_state["last_database_question"] = revised_intent["goal_text"]
        rag_state["last_retrieval_context"] = _serialize_context_summary(context)
        rag_state["pending_column_review"] = review_payload
        rag_state.pop("pending_extraction_opt_in", None)
        rag_state.pop("pending_sql_candidate", None)
        rag_state["error"] = None
        rag_state["thread_status"] = "awaiting_column_review"
        return _finalize_rag_state(updated, rag_state, status="done", active_thread=True)

    if pending_column_review.get("status") == "approved" and not pending_sql_candidate:
        approved_question = str(pending_column_review.get("question") or rag_state.get("last_database_question") or question).strip()
        approved_selection = _deserialize_column_selection(pending_column_review)
        try:
            prepared_candidate = service.prepare_sql_candidate(approved_question, approved_selection)
        except Exception as exc:
            error_payload = {
                "category": "db_rag_sql",
                "type": type(exc).__name__,
                "message": str(exc),
            }
            response_text = (
                "I could not prepare valid read-only SQL from the approved DB-RAG selection.\n\n"
                f"Details: {error_payload['type']}: {error_payload['message']}"
            )
            updated = _append_ai_response(state, response_text)
            output = dict(updated.get("output") or {})
            output["error"] = error_payload
            updated["output"] = output
            updated["meta"] = clear_clarification_meta(updated.get("meta") or {})
            rag_state["error"] = error_payload
            return _finalize_rag_state(updated, rag_state, status="error", active_thread=True)

        candidate_payload = _serialize_prepared_sql_candidate(prepared_candidate)
        candidate_payload["question"] = approved_question
        candidate_payload["status"] = "prepared"

        updated = _append_ai_response(state, _format_sql_candidate_response(candidate_payload))
        updated = _store_sql_candidate_output(updated, candidate_payload)
        updated = _clear_output_error(updated)
        updated["meta"] = clear_clarification_meta(updated.get("meta") or {})
        rag_state["last_database_question"] = approved_question
        rag_state["pending_sql_candidate"] = candidate_payload
        rag_state["error"] = None
        rag_state["thread_status"] = "awaiting_sql_review"
        return _finalize_rag_state(updated, rag_state, status="done", active_thread=True)

    context = service.retrieve_context(question, reranker_model=reranker_model)
    answer = service.answer_from_context(question, context)
    context_summary = _serialize_context_summary(context)
    intent = _serialize_intent(service.resolve_intent(question, context, prior_intent=dict(rag_state.get("active_intent") or {})))

    rag_state["active_intent"] = intent
    rag_state["last_database_question"] = question
    rag_state["last_retrieval_context"] = context_summary
    rag_state["error"] = None

    if not bool(_read_value(answer, "needs_sql", False)):
        updated = _append_ai_response(state, str(_read_value(answer, "answer", "") or "").strip())
        updated = _clear_output_error(updated)
        updated["meta"] = clear_clarification_meta(updated.get("meta") or {})
        rag_state.pop("pending_extraction_opt_in", None)
        rag_state.pop("pending_column_review", None)
        rag_state.pop("pending_sql_candidate", None)
        rag_state["thread_status"] = "answered_metadata"
        return _finalize_rag_state(updated, rag_state, status="done", active_thread=True)

    opt_in_prompt = "Would you like me to identify the tables and columns suitable for this extraction?"
    answer_text = str(_read_value(answer, "answer", "") or "").strip()
    response_text = answer_text if opt_in_prompt.lower() in answer_text.lower() else (
        f"{answer_text}\n\n{opt_in_prompt}" if answer_text else opt_in_prompt
    )
    updated = _append_ai_response(state, response_text)
    updated = _clear_output_error(updated)
    updated["meta"] = set_clarification_meta(
        updated.get("meta", {}),
        return_node="rag_db_qa",
        kind="rag_db_extraction_opt_in",
        pending_question=opt_in_prompt,
    )
    rag_state["pending_extraction_opt_in"] = {
        "question": opt_in_prompt,
        "intent_id": intent["intent_id"],
        "goal_text": intent["goal_text"],
        "status": "awaiting_reply",
    }
    rag_state.pop("intent_snapshot_for_selection", None)
    rag_state.pop("pending_column_review", None)
    rag_state.pop("pending_sql_candidate", None)
    rag_state["thread_status"] = "awaiting_extraction_opt_in"
    return _finalize_rag_state(updated, rag_state, status="done", active_thread=True)
