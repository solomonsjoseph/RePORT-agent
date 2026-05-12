from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from db_rag.service.errors import DbRagUnanswerableError

from ...memory import complete_task, upsert_user_intent_from_db_rag_intent
from ...state import AgentState, MetaKeys
from ..clarification_contracts import (
    CLARIFICATION_KIND_DB_RAG_RECOVERABLE_ERROR,
    CLARIFICATION_KIND_RAG_DB_EXTRACTION_OPT_IN,
    build_expected,
)
from ..state_helpers import clear_clarification_meta, get_agent_state, set_clarification_meta
from ..tool_routing import latest_user_message
from utils.performance import collect_timings, timing_stage
from .helpers import (
    _SUPPORTED_PROVIDERS,
    _append_ai_response,
    _append_assistant_event,
    _append_clarification_event,
    _append_column_review_request_event,
    _append_sql_candidate_events,
    _bootstrap_active_intent,
    _build_pending_extraction_opt_in,
    _clear_output_error,
    _deserialize_column_selection,
    _finalize_rag_state,
    _format_column_review_response,
    _format_sql_candidate_response,
    _intent_snapshot,
    _normalize_population_scope,
    _read_value,
    _render_db_rag_recent_turns,
    _reset_active_workflow_for_new_question,
    _serialize_column_selection,
    _serialize_context_pool,
    _serialize_context_summary,
    _serialize_intent,
    _serialize_prepared_sql_candidate,
    _store_column_selection_artifact,
    _store_sql_candidate_artifact,
    _store_sql_candidate_output,
    _with_population_scope_warning,
)

_EXPLICIT_EXTRACTION_CUES = (
    "sql",
    "extract",
    "subset",
    "filter",
    "count",
    "cohort",
    "rows",
    "generate the sql",
    "run the query",
)

_RESOLVED_TASK_META_KEYS = (
    MetaKeys.RESOLVED_TASK_ID,
    MetaKeys.RESOLVED_TASK_KIND,
    MetaKeys.RESOLVED_TASK_RELATIONSHIP,
    MetaKeys.RESOLVED_TASK_INTENDED_ACTION,
    MetaKeys.RESOLVED_TASK_USER_MESSAGE_HASH,
    "resolved_task_meta_consumed",
)

_REVISION_SYNTHETIC_MARKERS = (
    "User revision request:",
    "Parent extraction goal:",
    "Parent task summary:",
    "Prior SQL:",
)


def _artifact_content(state: AgentState, artifact_id: str | None) -> dict[str, Any] | None:
    artifact_key = str(artifact_id or "").strip()
    if not artifact_key:
        return None
    artifacts = dict(state.get("artifacts") or {})
    files = dict(artifacts.get("files") or {})
    artifact = files.get(artifact_key)
    if not isinstance(artifact, dict):
        return None
    content = artifact.get("content")
    return dict(content) if isinstance(content, dict) else None


def _replace_artifact_content(
    state: AgentState,
    *,
    artifact_id: str | None,
    content: dict[str, Any],
) -> AgentState:
    artifact_key = str(artifact_id or "").strip()
    if not artifact_key:
        return state
    artifacts = dict(state.get("artifacts") or {})
    files = dict(artifacts.get("files") or {})
    artifact = dict(files.get(artifact_key) or {})
    if not artifact:
        return state
    artifact["content"] = dict(content)
    files[artifact_key] = artifact
    artifacts["files"] = files
    return {
        **state,
        "artifacts": artifacts,
    }


def _clear_resolved_task_meta(state: AgentState) -> AgentState:
    meta = dict(state.get("meta") or {})
    for key in _RESOLVED_TASK_META_KEYS:
        meta.pop(key, None)
    return {
        **state,
        "meta": meta,
    }


def _is_explicit_extraction_question(question: str) -> bool:
    normalized = " ".join(str(question or "").strip().lower().split())
    if not normalized:
        return False
    return any(cue in normalized for cue in _EXPLICIT_EXTRACTION_CUES)


def _classify_population_scope_for_question(
    service,
    *,
    question: str,
    context: Any,
    active_intent: dict[str, Any] | None = None,
    intent_snapshot: dict[str, Any] | None = None,
    referenced_artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not hasattr(service, "classify_population_scope"):
        return _normalize_population_scope({})
    try:
        result = service.classify_population_scope(
            question=question,
            context=context,
            active_intent=active_intent,
            intent_snapshot=intent_snapshot,
            referenced_artifacts=referenced_artifacts,
        )
    except Exception:
        return _normalize_population_scope({})
    return _normalize_population_scope(result)


def _resolve_intent_for_question(
    *,
    service,
    question: str,
    context: Any,
    prior_intent: dict[str, Any],
    force_extraction: bool,
) -> dict[str, Any]:
    resolved = _serialize_intent(
        service.resolve_intent(
            question,
            context,
            prior_intent=prior_intent,
        )
    )
    if not resolved.get("intent_id"):
        resolved["intent_id"] = f"intent:{question}"
    resolved["source_question"] = question
    resolved["goal_text"] = str(resolved.get("goal_text") or question).strip() or question
    resolved["status"] = "active"
    if force_extraction:
        resolved["mode"] = "extraction"
    elif not resolved.get("mode"):
        resolved["mode"] = "metadata"
    return resolved


def _fallback_extraction_intent(*, intent_id: str, goal_text: str) -> dict[str, Any]:
    return {
        "intent_id": intent_id,
        "source_question": goal_text,
        "goal_text": goal_text,
        "mode": "extraction",
        "population": None,
        "filters": [],
        "required_tables": [],
        "required_columns": [],
        "excluded_tables": [],
        "excluded_columns": [],
        "feedback_history": [],
        "status": "active",
    }


def _source_message_hash(state: AgentState) -> str | None:
    meta = state.get("meta") or {}
    value = meta.get(MetaKeys.RAG_DB_SOURCE_MESSAGE_HASH) or meta.get(MetaKeys.LAST_USER_MESSAGE_HASH)
    if value is None:
        return None
    return str(value).strip() or None


def _upsert_db_rag_user_intent(
    state: AgentState,
    *,
    active_intent: dict[str, Any],
    status: str,
    force_new: bool = False,
) -> AgentState:
    return upsert_user_intent_from_db_rag_intent(
        state,
        active_intent=active_intent,
        source_message_hash=_source_message_hash(state),
        status=status,
        force_new=force_new,
    )


def _retrieve_context_for_intent(service, intent: dict[str, Any], reranker_model: str | None) -> Any:
    if hasattr(service, "retrieve_context_for_intent"):
        return service.retrieve_context_for_intent(SimpleNamespace(**intent), reranker_model=reranker_model)
    return service.retrieve_context(str(intent.get("goal_text") or ""), reranker_model=reranker_model)


def _is_missing_join_key_preflight_error(exc: Exception) -> bool:
    if not isinstance(exc, DbRagUnanswerableError):
        return False
    message = str(exc)
    return (
        "Multi-table extraction requires an approved join key column" in message
        and "Missing approved subject/family ID join keys" in message
    )


def _reopen_column_review_for_sql_preflight(
    state: AgentState,
    rag_state: dict[str, Any],
    *,
    selection_artifact_id: str,
    selection_payload: dict[str, Any],
    error_message: str,
) -> AgentState:
    review = dict(selection_payload)
    review["status"] = "needs_revision"
    feedback_history = list(review.get("feedback_history") or [])
    feedback_history.append(
        {
            "action": "sql_preflight_missing_join_keys",
            "feedback": (
                "SQL preflight found a multi-table extraction without approved join keys for every selected table. "
                f"{error_message}"
            ),
        }
    )
    review["feedback_history"] = feedback_history

    updated = _replace_artifact_content(state, artifact_id=selection_artifact_id, content=review)
    clarification_text = (
        "SQL generation needs join-key clarification before it can proceed. "
        "Please include approved SUBJID or FID columns for every selected table, "
        "or revise the selection to a single-table extraction."
    )
    updated = _append_ai_response(updated, _format_column_review_response(clarification_text, review))
    updated = _append_assistant_event(updated, updated["output"]["qa_response"])
    updated = _append_column_review_request_event(
        updated,
        artifact_id=selection_artifact_id,
        text="DB-RAG column selection needs join-key clarification before SQL generation.",
    )
    updated = _clear_output_error(updated)
    updated["meta"] = clear_clarification_meta(updated.get("meta") or {})

    rag_state["pending_column_review_artifact_id"] = selection_artifact_id
    rag_state["approved_column_selection_artifact_id"] = None
    rag_state["pending_sql_candidate_artifact_id"] = None
    rag_state["pending_column_review"] = review
    rag_state.pop("pending_sql_candidate", None)
    rag_state["error"] = None
    rag_state["thread_status"] = "awaiting_column_review"
    return _finalize_rag_state(updated, rag_state, status="done", active_thread=True)


def _build_sql_preparation_recovery_question(
    service,
    *,
    error_payload: dict[str, Any],
    selection_payload: dict[str, Any],
) -> str:
    fallback = (
        "I could not prepare valid read-only SQL from the approved DB-RAG selection. "
        "Please clarify what should change in the selected tables/columns, filters, or join keys so I can regenerate "
        "the DB-RAG column selection and try again."
    )
    if not hasattr(service, "build_recoverable_error_clarification"):
        return fallback

    context = {
        "goal_text": str(selection_payload.get("goal_text") or selection_payload.get("source_question") or ""),
        "tables": list(selection_payload.get("tables") or []),
        "columns": list(selection_payload.get("columns") or []),
        "feedback_history": list(selection_payload.get("feedback_history") or []),
    }
    try:
        result = service.build_recoverable_error_clarification(
            workflow="db_rag_sql_preparation",
            error_payload=error_payload,
            context=context,
        )
    except Exception:
        return fallback
    if not isinstance(result, dict):
        return fallback
    question = str(result.get("clarification_question") or "").strip()
    return question or fallback


def _ask_sql_preparation_recovery_clarification(
    state: AgentState,
    rag_state: dict[str, Any],
    *,
    service,
    selection_artifact_id: str,
    selection_payload: dict[str, Any],
    error_payload: dict[str, Any],
) -> AgentState:
    prompt = _build_sql_preparation_recovery_question(
        service,
        error_payload=error_payload,
        selection_payload=selection_payload,
    )
    updated = _append_ai_response(state, prompt)
    updated = _append_assistant_event(updated, updated["output"]["qa_response"])
    updated = _append_clarification_event(updated, prompt)
    updated = _clear_output_error(updated)
    updated["meta"] = set_clarification_meta(
        updated.get("meta", {}),
        return_node="rag_db_qa",
        kind=CLARIFICATION_KIND_DB_RAG_RECOVERABLE_ERROR,
        pending_question=prompt,
        expected=build_expected(CLARIFICATION_KIND_DB_RAG_RECOVERABLE_ERROR),
    )

    rag_state["pending_recoverable_error"] = {
        "status": "awaiting_reply",
        "stage": "sql_preparation",
        "selection_artifact_id": selection_artifact_id,
        "error_payload": dict(error_payload),
        "prompt": prompt,
    }
    rag_state["pending_sql_candidate_artifact_id"] = None
    rag_state.pop("pending_sql_candidate", None)
    rag_state["error"] = None
    rag_state["thread_status"] = "awaiting_recoverable_error_clarification"
    return _finalize_rag_state(updated, rag_state, status="done", active_thread=True)


def _handle_pending_recoverable_error_reply(
    state: AgentState,
    rag_state: dict[str, Any],
    *,
    service,
    reranker_model: str | None,
    question: str,
) -> AgentState:
    pending = dict(rag_state.get("pending_recoverable_error") or {})
    if pending.get("status") != "awaiting_reply":
        return _finish_terminal_response(
            state,
            {**rag_state, "pending_recoverable_error": None},
            text="The DB-RAG recovery state is incomplete. Please restate the database request.",
            thread_status="answered_metadata",
            active_thread=True,
        )

    stage = str(pending.get("stage") or "").strip()
    if stage != "sql_preparation":
        return _finish_terminal_response(
            state,
            {**rag_state, "pending_recoverable_error": None},
            text="The DB-RAG recovery state is not supported. Please restate the database request.",
            thread_status="answered_metadata",
            active_thread=True,
        )

    selection_artifact_id = str(pending.get("selection_artifact_id") or "").strip()
    selection_payload = _artifact_content(state, selection_artifact_id)
    if not selection_artifact_id or not selection_payload:
        return _finish_terminal_response(
            state,
            {**rag_state, "pending_recoverable_error": None},
            text="I could not recover the DB-RAG column selection that needs revision. Please restate the database request.",
            thread_status="answered_metadata",
            active_thread=True,
        )

    error_payload = dict(pending.get("error_payload") or {})
    error_text = str(error_payload.get("message") or "SQL preparation failed.").strip()
    review = dict(selection_payload)
    review["status"] = "needs_revision"
    feedback_history = list(review.get("feedback_history") or [])
    feedback_history.append(
        {
            "action": "sql_preparation_recovery_clarification",
            "feedback": f"SQL preparation failed: {error_text}. User clarification: {question}",
        }
    )
    review["feedback_history"] = feedback_history

    updated = _replace_artifact_content(state, artifact_id=selection_artifact_id, content=review)
    updated["meta"] = clear_clarification_meta(updated.get("meta") or {})
    rag_state["pending_recoverable_error"] = None
    rag_state["pending_column_review_artifact_id"] = selection_artifact_id
    rag_state["approved_column_selection_artifact_id"] = None
    rag_state["pending_sql_candidate_artifact_id"] = None
    rag_state["pending_column_review"] = review
    rag_state.pop("pending_sql_candidate", None)
    rag_state["error"] = None
    rag_state["thread_status"] = "awaiting_column_review"
    return _handle_pending_column_review_resume(
        updated,
        rag_state,
        service=service,
        reranker_model=reranker_model,
        question=question,
    )


def _finish_terminal_response(
    state: AgentState,
    rag_state: dict[str, Any],
    *,
    text: str,
    thread_status: str,
    active_thread: bool,
) -> AgentState:
    updated = _append_ai_response(state, text)
    updated = _append_assistant_event(updated, updated["output"]["qa_response"])
    updated = _clear_output_error(updated)
    updated["meta"] = clear_clarification_meta(updated.get("meta") or {})
    rag_state["error"] = None
    rag_state["thread_status"] = thread_status
    return _finalize_rag_state(updated, rag_state, status="done", active_thread=active_thread)


def _start_column_review(
    state: AgentState,
    rag_state: dict[str, Any],
    *,
    service,
    question: str,
    intent: dict[str, Any],
    context: Any,
    feedback_history: list[dict[str, Any]] | None = None,
    previous_selection: dict[str, Any] | None = None,
    revised: bool,
    population_scope: dict[str, Any] | None = None,
) -> AgentState:
    selection_goal_text = str(intent.get("goal_text") or question).strip() or question
    selection = service.prepare_column_selection(
        selection_goal_text,
        context,
        feedback_history=list(feedback_history or []),
        previous_selection=previous_selection,
        intent_snapshot=_intent_snapshot(intent),
    )
    selection_payload = _serialize_column_selection(selection)
    selection_payload["question"] = selection_goal_text
    selection_payload["feedback_history"] = list(selection_payload.get("feedback_history") or feedback_history or [])
    selection_payload["status"] = "awaiting_review"
    selection_payload["population_scope"] = _normalize_population_scope(
        population_scope or rag_state.get("population_scope") or {}
    )

    review_prompt = _format_column_review_response("", selection_payload, revised=revised)
    selection_payload["review_prompt"] = review_prompt

    updated, artifact_id = _store_column_selection_artifact(
        state,
        selection=selection_payload,
        source_question=str(intent.get("source_question") or question),
        goal_text=selection_goal_text,
        intent_snapshot=_intent_snapshot(intent),
        retrieval_summary=_serialize_context_summary(context),
        retrieval_pool=_serialize_context_pool(context),
        summary="Proposed DB-RAG column selection awaiting human review.",
    )
    updated = _append_ai_response(updated, review_prompt)
    updated = _append_assistant_event(updated, updated["output"]["qa_response"])
    updated = _append_column_review_request_event(
        updated,
        artifact_id=artifact_id,
        text=review_prompt,
    )
    updated = _clear_output_error(updated)
    updated["meta"] = clear_clarification_meta(updated.get("meta") or {})

    rag_state["active_intent"] = intent
    rag_state["population_scope"] = selection_payload["population_scope"]
    rag_state["pending_extraction_opt_in"] = None
    rag_state["pending_column_review_artifact_id"] = artifact_id
    rag_state["approved_column_selection_artifact_id"] = None
    rag_state["pending_sql_candidate_artifact_id"] = None
    rag_state["pending_column_review"] = selection_payload
    rag_state.pop("pending_sql_candidate", None)
    rag_state["last_database_question"] = selection_goal_text
    rag_state["last_retrieval_context"] = _serialize_context_summary(context)
    rag_state["error"] = None
    rag_state["thread_status"] = "awaiting_column_review"
    return _finalize_rag_state(updated, rag_state, status="done", active_thread=True)


def _start_sql_review(
    state: AgentState,
    rag_state: dict[str, Any],
    *,
    service,
    selection_artifact_id: str,
    selection_payload: dict[str, Any],
) -> AgentState:
    approved_question = str(
        selection_payload.get("goal_text")
        or selection_payload.get("source_question")
        or rag_state.get("last_database_question")
        or ""
    ).strip()
    approved_selection = _deserialize_column_selection(
        {
            "selection_id": selection_payload.get("selection_id", ""),
            "question": approved_question,
            "tables": selection_payload.get("tables", []),
            "columns": selection_payload.get("columns", []),
            "rationale": selection_payload.get("rationale", ""),
            "feedback_history": selection_payload.get("feedback_history", []),
            "status": selection_payload.get("status", "approved"),
        }
    )

    try:
        prepared_candidate = service.prepare_sql_candidate(approved_question, approved_selection)
    except Exception as exc:
        if _is_missing_join_key_preflight_error(exc):
            return _reopen_column_review_for_sql_preflight(
                state,
                rag_state,
                selection_artifact_id=selection_artifact_id,
                selection_payload=selection_payload,
                error_message=str(exc),
            )
        error_payload = {
            "category": "db_rag_sql",
            "type": type(exc).__name__,
            "message": str(exc),
            "selection_artifact_id": selection_artifact_id,
        }
        return _ask_sql_preparation_recovery_clarification(
            state,
            rag_state,
            service=service,
            selection_artifact_id=selection_artifact_id,
            selection_payload=selection_payload,
            error_payload=error_payload,
        )

    candidate_payload = _serialize_prepared_sql_candidate(prepared_candidate)
    candidate_payload["question"] = approved_question
    candidate_payload["status"] = "prepared"
    candidate_payload["population_scope"] = _normalize_population_scope(
        selection_payload.get("population_scope") or rag_state.get("population_scope") or {}
    )

    intent = _serialize_intent(rag_state.get("active_intent") or {})
    updated, candidate_artifact_id = _store_sql_candidate_artifact(
        state,
        candidate=candidate_payload,
        selection_artifact_id=selection_artifact_id,
        source_question=str(selection_payload.get("source_question") or approved_question),
        goal_text=str(selection_payload.get("goal_text") or approved_question),
        intent_snapshot=selection_payload.get("intent_snapshot") or _intent_snapshot(intent),
        population_scope=candidate_payload["population_scope"],
    )
    updated = _append_ai_response(updated, _format_sql_candidate_response(candidate_payload))
    updated = _store_sql_candidate_output(updated, candidate_payload)
    updated = _append_assistant_event(updated, updated["output"]["qa_response"])
    updated = _append_sql_candidate_events(updated, candidate_payload, candidate_artifact_id)
    updated = _clear_output_error(updated)
    updated["meta"] = clear_clarification_meta(updated.get("meta") or {})

    rag_state["approved_column_selection_artifact_id"] = selection_artifact_id
    rag_state["pending_column_review_artifact_id"] = None
    rag_state["pending_sql_candidate_artifact_id"] = candidate_artifact_id
    rag_state["pending_column_review"] = selection_payload
    rag_state["pending_sql_candidate"] = candidate_payload
    rag_state["last_database_question"] = approved_question
    rag_state["error"] = None
    rag_state["thread_status"] = "awaiting_sql_review"
    return _finalize_rag_state(updated, rag_state, status="done", active_thread=True)


def _reprompt_pending_extraction(
    state: AgentState,
    rag_state: dict[str, Any],
    pending_extraction_opt_in: dict[str, Any],
) -> AgentState:
    prompt = str(pending_extraction_opt_in.get("prompt") or "").strip()
    goal_text = str(pending_extraction_opt_in.get("goal_text") or "").strip()
    reprompt = (
        "Please reply with 'yes' to proceed with table/column selection for extraction, or 'no' to skip it"
        + (f' for this request:\n\n"{goal_text}"' if goal_text else ".")
    )
    updated = _append_ai_response(state, reprompt)
    updated = _append_clarification_event(updated, prompt)
    updated = _clear_output_error(updated)
    updated["meta"] = set_clarification_meta(
        updated.get("meta", {}),
        return_node="rag_db_qa",
        kind=CLARIFICATION_KIND_RAG_DB_EXTRACTION_OPT_IN,
        pending_question=prompt or None,
        expected=build_expected(CLARIFICATION_KIND_RAG_DB_EXTRACTION_OPT_IN),
    )
    rag_state["thread_status"] = "awaiting_extraction_opt_in"
    rag_state["error"] = None
    return _finalize_rag_state(updated, rag_state, status="done", active_thread=True)


def _classify_pending_extraction_boundary(
    state: AgentState,
    *,
    service,
    question: str,
    pending_extraction_opt_in: dict[str, Any],
    active_intent: dict[str, Any],
) -> str:
    if hasattr(service, "classify_extraction_gate_message"):
        classified = service.classify_extraction_gate_message(
            pending_prompt=str(pending_extraction_opt_in.get("prompt") or "").strip(),
            user_message=question,
            recent_transcript=_render_db_rag_recent_turns(state, question),
            active_intent=active_intent or None,
        ) or {}
        label = str(classified.get("label") or "").strip().lower()
        if label in {"new_question", "reply_to_pending_gate", "unknown"}:
            return label
    return "unknown"


def _classify_pending_extraction_reply_label(
    state: AgentState,
    *,
    service,
    question: str,
    pending_extraction_opt_in: dict[str, Any],
) -> str:
    if hasattr(service, "classify_pending_reply"):
        classified = service.classify_pending_reply(
            pending_kind="rag_db_extraction_opt_in",
            pending_question=str(pending_extraction_opt_in.get("prompt") or "").strip(),
            user_reply=question,
            recent_transcript=_render_db_rag_recent_turns(state, question),
        ) or {}
        label = str(classified.get("label") or "").strip().lower()
        if label in {"yes", "no", "substantive_followup", "unknown"}:
            return label
    return "unknown"


def _handle_pending_sql_review_resume(
    state: AgentState,
    rag_state: dict[str, Any],
    *,
    question: str,
) -> AgentState:
    del question

    artifact_id = str(rag_state.get("pending_sql_candidate_artifact_id") or "").strip()
    candidate_payload = _artifact_content(state, artifact_id)
    if artifact_id and candidate_payload:
        updated = _append_ai_response(state, _format_sql_candidate_response(candidate_payload))
        updated = _store_sql_candidate_output(updated, candidate_payload)
        updated = _append_assistant_event(updated, updated["output"]["qa_response"])
        updated = _append_sql_candidate_events(updated, candidate_payload, artifact_id)
        updated = _clear_output_error(updated)
        updated["meta"] = clear_clarification_meta(updated.get("meta") or {})
        rag_state["pending_sql_candidate"] = candidate_payload
        rag_state["error"] = None
        rag_state["thread_status"] = "awaiting_sql_review"
        return _finalize_rag_state(updated, rag_state, status="done", active_thread=True)

    return _finish_terminal_response(
        state,
        {
            **rag_state,
            "approved_column_selection_artifact_id": None,
            "pending_sql_candidate_artifact_id": None,
            "pending_sql_candidate": None,
        },
        text="The pending SQL review state is missing its prepared candidate. Please restart the DB-RAG request.",
        thread_status="error",
        active_thread=False,
    )


def _handle_pending_column_review_resume(
    state: AgentState,
    rag_state: dict[str, Any],
    *,
    service,
    reranker_model: str | None,
    question: str,
) -> AgentState:
    del question

    artifact_id = str(rag_state.get("pending_column_review_artifact_id") or "").strip()
    selection_payload = _artifact_content(state, artifact_id)
    if selection_payload:
        status = str(selection_payload.get("status") or "").strip()
        if status == "awaiting_review":
            updated = _append_ai_response(state, _format_column_review_response("", selection_payload))
            updated = _append_assistant_event(updated, updated["output"]["qa_response"])
            updated = _append_column_review_request_event(
                updated,
                artifact_id=artifact_id,
                text="DB-RAG column selection is awaiting human review.",
            )
            updated = _clear_output_error(updated)
            updated["meta"] = clear_clarification_meta(updated.get("meta") or {})
            rag_state["pending_column_review"] = selection_payload
            rag_state["error"] = None
            rag_state["thread_status"] = "awaiting_column_review"
            return _finalize_rag_state(updated, rag_state, status="done", active_thread=True)
        if status == "needs_revision":
            active_intent = _serialize_intent(rag_state.get("active_intent") or {}) or _bootstrap_active_intent(rag_state, str(selection_payload.get("goal_text") or ""))
            feedback_history = list(selection_payload.get("feedback_history") or [])
            revised_intent = (
                _serialize_intent(service.update_intent_from_feedback(active_intent, feedback_history))
                if hasattr(service, "update_intent_from_feedback")
                else active_intent
            )
            revised_intent["status"] = "active"
            context = _retrieve_context_for_intent(service, revised_intent, reranker_model)
            return _start_column_review(
                state,
                rag_state,
                service=service,
                question=str(revised_intent.get("goal_text") or selection_payload.get("goal_text") or ""),
                intent=revised_intent,
                context=context,
                feedback_history=feedback_history,
                previous_selection=selection_payload,
                revised=True,
            )

    return _finish_terminal_response(
        state,
        {
            **rag_state,
            "pending_column_review_artifact_id": None,
            "pending_column_review": None,
        },
        text="The pending column review state is incomplete. Please restart the DB-RAG request.",
        thread_status="error",
        active_thread=False,
    )


def _handle_approved_column_selection_resume(
    state: AgentState,
    rag_state: dict[str, Any],
    *,
    service,
) -> AgentState:
    artifact_id = str(rag_state.get("approved_column_selection_artifact_id") or "").strip()
    selection_payload = _artifact_content(state, artifact_id)
    if artifact_id and selection_payload:
        return _start_sql_review(state, rag_state, service=service, selection_artifact_id=artifact_id, selection_payload=selection_payload)

    return _finish_terminal_response(
        state,
        {
            **rag_state,
            "approved_column_selection_artifact_id": None,
            "pending_column_review": None,
        },
        text="The approved column selection state is incomplete. Please restart the DB-RAG request.",
        thread_status="error",
        active_thread=False,
    )


def _handle_pending_extraction_reply(
    state: AgentState,
    rag_state: dict[str, Any],
    *,
    service,
    reranker_model: str | None,
    question: str,
) -> AgentState:
    pending_extraction_opt_in = dict(rag_state.get("pending_extraction_opt_in") or {})
    active_intent = _serialize_intent(rag_state.get("active_intent") or {})
    boundary = _classify_pending_extraction_boundary(
        state,
        service=service,
        question=question,
        pending_extraction_opt_in=pending_extraction_opt_in,
        active_intent=active_intent,
    )
    if boundary == "new_question":
        rag_state.pop("pending_extraction_opt_in", None)
        return _handle_fresh_db_rag_question(
            state,
            rag_state,
            service=service,
            reranker_model=reranker_model,
            question=question,
        )

    if boundary != "reply_to_pending_gate":
        return _reprompt_pending_extraction(state, rag_state, pending_extraction_opt_in)

    reply_label = _classify_pending_extraction_reply_label(
        state,
        service=service,
        question=question,
        pending_extraction_opt_in=pending_extraction_opt_in,
    )
    if reply_label == "yes":
        goal_text = str(active_intent.get("goal_text") or pending_extraction_opt_in.get("goal_text") or "").strip()
        if not goal_text:
            return _finish_terminal_response(
                state,
                rag_state,
                text="I couldn't recover the extraction goal from this thread. Please restate the subset request.",
                thread_status="answered_metadata",
                active_thread=True,
            )
        if not active_intent:
            active_intent = _fallback_extraction_intent(
                intent_id=str(pending_extraction_opt_in.get("intent_id") or f"intent:{goal_text}"),
                goal_text=goal_text,
            )
        active_intent["mode"] = "extraction"
        state = _upsert_db_rag_user_intent(
            state,
            active_intent=active_intent,
            status="awaiting_column_review",
        )
        context = _retrieve_context_for_intent(service, active_intent, reranker_model)
        return _start_column_review(
            state,
            rag_state,
            service=service,
            question=goal_text,
            intent=active_intent,
            context=context,
            revised=False,
        )

    if reply_label == "no":
        if active_intent:
            active_intent["status"] = "extraction_declined"
            rag_state["active_intent"] = active_intent
            state = _upsert_db_rag_user_intent(
                state,
                active_intent=active_intent,
                status="declined",
            )
        rag_state["pending_extraction_opt_in"] = None
        rag_state["pending_column_review_artifact_id"] = None
        rag_state["pending_sql_candidate_artifact_id"] = None
        rag_state.pop("pending_column_review", None)
        rag_state.pop("pending_sql_candidate", None)
        return _finish_terminal_response(
            state,
            rag_state,
            text="Understood. I won't prepare table/column selection for extraction. Ask a metadata question anytime.",
            thread_status="answered_metadata",
            active_thread=True,
        )

    if reply_label == "substantive_followup":
        context = service.retrieve_context(question, reranker_model=reranker_model)
        intent = _resolve_intent_for_question(
            service=service,
            question=question,
            context=context,
            prior_intent=active_intent,
            force_extraction=True,
        )
        state = _upsert_db_rag_user_intent(
            state,
            active_intent=intent,
            status="awaiting_column_review",
        )
        rag_state = _reset_active_workflow_for_new_question(
            rag_state,
            question=question,
            intent=intent,
            context_summary=_serialize_context_summary(context),
        )
        return _start_column_review(
            state,
            rag_state,
            service=service,
            question=question,
            intent=intent,
            context=context,
            revised=False,
        )

    return _reprompt_pending_extraction(state, rag_state, pending_extraction_opt_in)


def _completed_task(state: AgentState, task_id: str | None) -> dict[str, Any] | None:
    task_key = str(task_id or "").strip()
    if not task_key:
        return None
    memory = dict(state.get("memory") or {})
    completed = dict(memory.get("completed_tasks") or {})
    task = completed.get(task_key)
    return dict(task) if isinstance(task, dict) else None


def _terminal_resolved_reference_error(
    state: AgentState,
    rag_state: dict[str, Any],
    text: str,
) -> AgentState:
    updated = _append_ai_response(_clear_resolved_task_meta(state), text)
    updated = _append_assistant_event(updated, updated["output"]["qa_response"])
    updated = _clear_output_error(updated)
    rag_state["error"] = None
    rag_state["thread_status"] = "error"
    return _finalize_rag_state(updated, rag_state, status="done", active_thread=False)


def _sql_inspection_response(parent_task: dict[str, Any], sql_payload: dict[str, Any]) -> str:
    sql = str(sql_payload.get("sql") or "").strip()
    goal_text = str(
        sql_payload.get("goal_text")
        or parent_task.get("goal_text")
        or parent_task.get("source_question")
        or ""
    ).strip()
    tables = ", ".join(str(table) for table in list(sql_payload.get("tables") or []) if str(table).strip())
    columns = ", ".join(
        ".".join(
            part
            for part in (
                str(column.get("table") or "").strip(),
                str(column.get("column") or "").strip(),
            )
            if part
        )
        for column in list(sql_payload.get("columns") or [])
        if isinstance(column, dict) and str(column.get("column") or "").strip()
    )
    context_lines = []
    if goal_text:
        context_lines.append(f"Original extraction goal: {goal_text}")
    if tables:
        context_lines.append(f"Tables: {tables}")
    if columns:
        context_lines.append(f"Columns: {columns}")
    context = "\n".join(context_lines)
    return (
        f"Here is the SQL from the completed DB-RAG extraction:\n\n```sql\n{sql}\n```\n\n"
        f"{context}"
    ).strip()


def _handle_resolved_sql_inspection(
    state: AgentState,
    rag_state: dict[str, Any],
    *,
    question: str,
    parent_task_id: str,
    parent_task: dict[str, Any],
) -> AgentState:
    artifact_refs = dict(parent_task.get("artifact_refs") or {})
    sql_artifact_id = str(artifact_refs.get("sql_candidate_artifact_id") or "").strip()
    sql_payload = _artifact_content(state, sql_artifact_id)
    if not sql_payload:
        return _terminal_resolved_reference_error(
            state,
            rag_state,
            "I found the referenced DB-RAG task, but its SQL candidate artifact is missing. I cannot inspect it.",
        )

    response_text = _sql_inspection_response(parent_task, sql_payload)
    updated = _append_ai_response(_clear_resolved_task_meta(state), response_text)
    updated = _append_assistant_event(updated, updated["output"]["qa_response"])
    updated = _clear_output_error(updated)
    updated = complete_task(
        updated,
        kind="qa_answer",
        source_question=question,
        goal_text=f"Inspect SQL artifact for {parent_task.get('label') or parent_task_id}",
        label=f"QA answer: {question.strip()[:80]}",
        summary="Answered from a completed DB-RAG SQL candidate artifact.",
        artifact_refs={"sql_candidate_artifact_id": sql_artifact_id},
        parent_task_id=parent_task_id,
        relationship_to_parent="inspect_artifact",
        provenance={"producer_node": "rag_db_qa"},
    )
    rag_state["pending_column_review_artifact_id"] = None
    rag_state["approved_column_selection_artifact_id"] = None
    rag_state["pending_sql_candidate_artifact_id"] = None
    rag_state.pop("pending_column_review", None)
    rag_state.pop("pending_sql_candidate", None)
    rag_state["error"] = None
    rag_state["thread_status"] = "answered_artifact"
    return _finalize_rag_state(updated, rag_state, status="done", active_thread=False)


def _build_revision_question(question: str, parent_task: dict[str, Any], sql_payload: dict[str, Any]) -> str:
    parent_summary = str(parent_task.get("summary") or "").strip()
    parent_goal = str(parent_task.get("goal_text") or parent_task.get("source_question") or "").strip()
    sql = str(sql_payload.get("sql") or "").strip()
    parts = [f"User revision request: {question.strip()}"]
    if parent_goal:
        parts.append(f"Parent extraction goal: {parent_goal}")
    if parent_summary:
        parts.append(f"Parent task summary: {parent_summary}")
    if sql:
        parts.append(f"Prior SQL:\n{sql}")
    return "\n\n".join(parts)


def _clean_revision_goal(*, question: str, resolved_goal: str) -> str:
    goal = str(resolved_goal or "").strip()
    if not goal or any(marker in goal for marker in _REVISION_SYNTHETIC_MARKERS):
        return str(question or "").strip()
    return goal


def _contains_revision_synthetic_marker(text: str) -> bool:
    return any(marker in str(text or "") for marker in _REVISION_SYNTHETIC_MARKERS)


def _handle_resolved_sql_revision(
    state: AgentState,
    rag_state: dict[str, Any],
    *,
    service,
    reranker_model: str | None,
    question: str,
    parent_task_id: str,
    parent_task: dict[str, Any],
) -> AgentState:
    artifact_refs = dict(parent_task.get("artifact_refs") or {})
    selection_artifact_id = str(artifact_refs.get("selection_artifact_id") or "").strip()
    sql_artifact_id = str(artifact_refs.get("sql_candidate_artifact_id") or "").strip()
    selection_payload = _artifact_content(state, selection_artifact_id)
    sql_payload = _artifact_content(state, sql_artifact_id)
    if not selection_payload or not sql_payload:
        return _terminal_resolved_reference_error(
            state,
            rag_state,
            "I found the referenced DB-RAG task, but its selection or SQL artifact is missing. I cannot revise it.",
        )

    revision_question = _build_revision_question(question, parent_task, sql_payload)
    context = service.retrieve_context(revision_question, reranker_model=reranker_model)
    context_summary = _serialize_context_summary(context)
    prior_intent = _serialize_intent(sql_payload.get("intent_snapshot") or selection_payload.get("intent_snapshot") or {})
    intent = _resolve_intent_for_question(
        service=service,
        question=revision_question,
        context=context,
        prior_intent=prior_intent,
        force_extraction=True,
    )
    intent["source_question"] = question
    intent["goal_text"] = _clean_revision_goal(
        question=question,
        resolved_goal=str(intent.get("goal_text") or ""),
    )
    if _contains_revision_synthetic_marker(str(intent.get("intent_id") or "")):
        intent["intent_id"] = f"intent:{question}"
    rag_state = _reset_active_workflow_for_new_question(
        rag_state,
        question=question,
        intent=intent,
        context_summary=context_summary,
    )
    rag_state["active_task"] = {
        "parent_task_id": parent_task_id,
        "relationship_to_parent": "revision",
    }
    state_without_resolved_meta = _clear_resolved_task_meta(state)
    preserved_meta = dict(state_without_resolved_meta.get("meta") or {})
    updated = _start_column_review(
        state_without_resolved_meta,
        rag_state,
        service=service,
        question=question,
        intent=intent,
        context=context,
        revised=False,
    )
    meta = dict(preserved_meta)
    meta.update(dict(updated.get("meta") or {}))
    for key in _RESOLVED_TASK_META_KEYS:
        meta.pop(key, None)
    updated["meta"] = meta
    return updated


def _handle_resolved_memory_reference(
    state: AgentState,
    rag_state: dict[str, Any],
    *,
    service,
    reranker_model: str | None,
    question: str,
) -> AgentState | None:
    meta = dict(state.get("meta") or {})
    if meta.get(MetaKeys.RESOLVED_TASK_KIND) != "db_rag_sql_extraction":
        return None

    parent_task_id = str(meta.get(MetaKeys.RESOLVED_TASK_ID) or "").strip()
    parent_task = _completed_task(state, parent_task_id)
    if not parent_task:
        return _terminal_resolved_reference_error(
            state,
            rag_state,
            "I could not find the referenced completed DB-RAG SQL extraction task.",
        )

    relationship = str(meta.get(MetaKeys.RESOLVED_TASK_RELATIONSHIP) or "").strip()
    if relationship == "inspect_artifact":
        return _handle_resolved_sql_inspection(
            state,
            rag_state,
            question=question,
            parent_task_id=parent_task_id,
            parent_task=parent_task,
        )
    if relationship == "revision":
        return _handle_resolved_sql_revision(
            state,
            rag_state,
            service=service,
            reranker_model=reranker_model,
            question=question,
            parent_task_id=parent_task_id,
            parent_task=parent_task,
        )
    if relationship == "use_as_input":
        return _terminal_resolved_reference_error(
            state,
            rag_state,
            "This resolved dataset handoff should be routed to code generation, not DB-RAG.",
        )
    return _terminal_resolved_reference_error(
        state,
        rag_state,
        f"DB-RAG cannot handle the resolved DB-RAG SQL relationship '{relationship or 'unknown'}'.",
    )


def _handle_fresh_db_rag_question(
    state: AgentState,
    rag_state: dict[str, Any],
    *,
    service,
    reranker_model: str | None,
    question: str,
    force_new_intent: bool = True,
) -> AgentState:
    explicit_extraction = _is_explicit_extraction_question(question)
    context = service.retrieve_context(question, reranker_model=reranker_model)
    context_summary = _serialize_context_summary(context)
    intent = _resolve_intent_for_question(
        service=service,
        question=question,
        context=context,
        prior_intent=dict(rag_state.get("active_intent") or {}),
        force_extraction=explicit_extraction,
    )
    population_scope = _classify_population_scope_for_question(
        service,
        question=question,
        context=context,
        active_intent=intent,
        intent_snapshot=_intent_snapshot(intent),
    )
    rag_state = _reset_active_workflow_for_new_question(
        rag_state,
        question=question,
        intent=intent,
        context_summary=context_summary,
    )
    rag_state["population_scope"] = population_scope
    state = _upsert_db_rag_user_intent(
        state,
        active_intent=intent,
        status="awaiting_column_review" if explicit_extraction else "awaiting_extraction_opt_in",
        force_new=force_new_intent,
    )

    if explicit_extraction:
        return _start_column_review(
            state,
            rag_state,
            service=service,
            question=question,
            intent=intent,
            context=context,
            revised=False,
            population_scope=population_scope,
        )

    answer = service.answer_from_context(question, context)
    answer_text = str(_read_value(answer, "answer", "") or "").strip()
    pending_extraction_opt_in = _build_pending_extraction_opt_in(intent)
    prompt = str(pending_extraction_opt_in.get("prompt") or "").strip()
    response_text = answer_text if prompt.lower() in answer_text.lower() else (
        f"{answer_text}\n\n{prompt}" if answer_text else prompt
    )
    response_text = _with_population_scope_warning(response_text, population_scope, target="metadata")

    updated = _append_ai_response(state, response_text)
    updated = _append_assistant_event(updated, response_text)
    updated = _append_clarification_event(updated, prompt)
    updated = _clear_output_error(updated)
    updated["meta"] = set_clarification_meta(
        updated.get("meta", {}),
        return_node="rag_db_qa",
        kind=CLARIFICATION_KIND_RAG_DB_EXTRACTION_OPT_IN,
        pending_question=prompt or None,
        expected=build_expected(CLARIFICATION_KIND_RAG_DB_EXTRACTION_OPT_IN),
    )

    rag_state["pending_extraction_opt_in"] = pending_extraction_opt_in
    rag_state["thread_status"] = "awaiting_extraction_opt_in"
    rag_state["error"] = None
    return _finalize_rag_state(updated, rag_state, status="done", active_thread=True)


def _store_db_rag_timings(state: AgentState, records: list[dict[str, Any]]) -> AgentState:
    if not records:
        return state
    meta = dict(state.get("meta") or {})
    meta["db_rag_timing"] = {
        "node": "rag_db_qa",
        "stages": list(records)[-100:],
    }
    return {
        **state,
        "meta": meta,
    }


def _rag_db_qa_node_impl(
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
    question = str(question_override or latest_question)
    if question_override is not None:
        meta = dict(state.get("meta") or {})
        meta.pop(MetaKeys.RAG_DB_QUESTION_OVERRIDE, None)
        state = {
            **state,
            "meta": meta,
        }

    if provider not in _SUPPORTED_PROVIDERS:
        return _finish_terminal_response(
            state,
            rag_state,
            text="DB-RAG currently requires an OpenAI or Anthropic provider. Switch the model provider and try again.",
            thread_status="done",
            active_thread=False,
        )

    readiness = service.readiness() if service is not None else {"ready": False, "message": "DB-RAG service is unavailable."}
    if not readiness.get("ready"):
        return _finish_terminal_response(
            state,
            rag_state,
            text=str(readiness.get("message") or "DB-RAG assets are not ready."),
            thread_status="done",
            active_thread=False,
        )

    if question_override is None and (state.get("meta") or {}).get(MetaKeys.TURN_TYPE) == "workflow_reference":
        return _finish_terminal_response(
            state,
            rag_state,
            text=(
                "I do not have a resolved previous DB-RAG query for this turn. "
                "Please restate the database query you want to continue."
            ),
            thread_status="done",
            active_thread=False,
        )

    if question_override is not None:
        updated = _handle_fresh_db_rag_question(
            state,
            rag_state,
            service=service,
            reranker_model=reranker_model,
            question=question,
            force_new_intent=True,
        )
        updated_meta = dict(updated.get("meta") or {})
        updated_meta.pop(MetaKeys.RAG_DB_SOURCE_MESSAGE_HASH, None)
        return {**updated, "meta": updated_meta}

    if dict(rag_state.get("pending_recoverable_error") or {}).get("status") == "awaiting_reply":
        return _handle_pending_recoverable_error_reply(
            state,
            rag_state,
            service=service,
            reranker_model=reranker_model,
            question=question,
        )

    if rag_state.get("pending_sql_candidate_artifact_id"):
        return _handle_pending_sql_review_resume(state, rag_state, question=question)

    if rag_state.get("approved_column_selection_artifact_id"):
        return _handle_approved_column_selection_resume(
            state,
            rag_state,
            service=service,
        )

    if rag_state.get("pending_column_review_artifact_id"):
        return _handle_pending_column_review_resume(
            state,
            rag_state,
            service=service,
            reranker_model=reranker_model,
            question=question,
        )

    if dict(rag_state.get("pending_extraction_opt_in") or {}).get("status") == "awaiting_reply":
        return _handle_pending_extraction_reply(
            state,
            rag_state,
            service=service,
            reranker_model=reranker_model,
            question=question,
        )

    resolved = _handle_resolved_memory_reference(
        state,
        rag_state,
        service=service,
        reranker_model=reranker_model,
        question=question,
    )
    if resolved is not None:
        return resolved

    return _handle_fresh_db_rag_question(
        state,
        rag_state,
        service=service,
        reranker_model=reranker_model,
        question=question,
        force_new_intent=True,
    )


def rag_db_qa_node(
    state: AgentState,
    llm,
    *,
    provider: str,
    service,
    reranker_model: str | None = None,
    question_override: str | None = None,
) -> AgentState:
    with collect_timings() as records:
        with timing_stage("rag_db_qa.total"):
            updated = _rag_db_qa_node_impl(
                state,
                llm,
                provider=provider,
                service=service,
                reranker_model=reranker_model,
                question_override=question_override,
            )
    return _store_db_rag_timings(updated, records)
