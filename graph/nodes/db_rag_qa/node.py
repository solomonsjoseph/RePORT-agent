from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ...state import AgentState
from ..state_helpers import clear_clarification_meta, get_agent_state, set_clarification_meta
from ..tool_routing import latest_user_message
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
    _question_from_stale_qa_followup,
    _read_value,
    _render_db_rag_recent_turns,
    _reset_active_workflow_for_new_question,
    _serialize_column_selection,
    _serialize_context_summary,
    _serialize_intent,
    _serialize_prepared_sql_candidate,
    _store_column_selection_artifact,
    _store_sql_candidate_artifact,
    _store_sql_candidate_output,
)

_EXPLICIT_EXTRACTION_CUES = (
    "sql",
    "query",
    "extract",
    "subset",
    "filter",
    "count",
    "cohort",
    "rows",
    "generate the sql",
    "run the query",
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


def _is_explicit_extraction_question(question: str) -> bool:
    normalized = " ".join(str(question or "").strip().lower().split())
    if not normalized:
        return False
    return any(cue in normalized for cue in _EXPLICIT_EXTRACTION_CUES)


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
        "requested_fields": [],
        "filters": [],
        "required_tables": [],
        "required_columns": [],
        "excluded_tables": [],
        "excluded_columns": [],
        "feedback_history": [],
        "status": "active",
    }


def _retrieve_context_for_intent(service, intent: dict[str, Any], reranker_model: str | None) -> Any:
    if hasattr(service, "retrieve_context_for_intent"):
        return service.retrieve_context_for_intent(SimpleNamespace(**intent), reranker_model=reranker_model)
    return service.retrieve_context(str(intent.get("goal_text") or ""), reranker_model=reranker_model)


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

    updated, artifact_id = _store_column_selection_artifact(
        state,
        selection=selection_payload,
        source_question=str(intent.get("source_question") or question),
        goal_text=selection_goal_text,
        intent_snapshot=_intent_snapshot(intent),
        retrieval_summary=_serialize_context_summary(context),
        summary="Proposed DB-RAG column selection awaiting human review.",
    )
    updated = _append_ai_response(updated, _format_column_review_response("", selection_payload, revised=revised))
    updated = _append_assistant_event(updated, updated["output"]["qa_response"])
    updated = _append_column_review_request_event(
        updated,
        artifact_id=artifact_id,
        text="DB-RAG column selection is awaiting human review.",
    )
    updated = _clear_output_error(updated)
    updated["meta"] = clear_clarification_meta(updated.get("meta") or {})

    rag_state["active_intent"] = intent
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
        error_payload = {
            "category": "db_rag_sql",
            "type": type(exc).__name__,
            "message": str(exc),
            "selection_artifact_id": selection_artifact_id,
        }
        response_text = (
            "I could not prepare valid read-only SQL from the approved DB-RAG selection.\n\n"
            f"Details: {error_payload['type']}: {error_payload['message']}"
        )
        updated = _append_ai_response(state, response_text)
        updated = _append_assistant_event(updated, updated["output"]["qa_response"])
        output = dict(updated.get("output") or {})
        output["error"] = error_payload
        updated["output"] = output
        updated["meta"] = clear_clarification_meta(updated.get("meta") or {})
        rag_state["error"] = error_payload
        rag_state["approved_column_selection_artifact_id"] = None
        rag_state["pending_sql_candidate_artifact_id"] = None
        rag_state["thread_status"] = "error"
        return _finalize_rag_state(updated, rag_state, status="error", active_thread=True)

    candidate_payload = _serialize_prepared_sql_candidate(prepared_candidate)
    candidate_payload["question"] = approved_question
    candidate_payload["status"] = "prepared"

    intent = _serialize_intent(rag_state.get("active_intent") or {})
    updated, candidate_artifact_id = _store_sql_candidate_artifact(
        state,
        candidate=candidate_payload,
        selection_artifact_id=selection_artifact_id,
        source_question=str(selection_payload.get("source_question") or approved_question),
        goal_text=str(selection_payload.get("goal_text") or approved_question),
        intent_snapshot=selection_payload.get("intent_snapshot") or _intent_snapshot(intent),
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
    reprompt = (
        "Please reply with 'yes' to proceed with table/column selection for extraction, or 'no' to skip it."
    )
    updated = _append_ai_response(state, reprompt)
    updated = _append_clarification_event(updated, prompt)
    updated = _clear_output_error(updated)
    updated["meta"] = set_clarification_meta(
        updated.get("meta", {}),
        return_node="rag_db_qa",
        kind="rag_db_extraction_opt_in",
        pending_question=prompt or None,
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


def _handle_fresh_db_rag_question(
    state: AgentState,
    rag_state: dict[str, Any],
    *,
    service,
    reranker_model: str | None,
    question: str,
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
    rag_state = _reset_active_workflow_for_new_question(
        rag_state,
        question=question,
        intent=intent,
        context_summary=context_summary,
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
        )

    answer = service.answer_from_context(question, context)
    answer_text = str(_read_value(answer, "answer", "") or "").strip()
    pending_extraction_opt_in = _build_pending_extraction_opt_in(intent)
    prompt = str(pending_extraction_opt_in.get("prompt") or "").strip()
    response_text = answer_text if prompt.lower() in answer_text.lower() else (
        f"{answer_text}\n\n{prompt}" if answer_text else prompt
    )

    updated = _append_ai_response(state, response_text)
    updated = _append_assistant_event(updated, answer_text or response_text)
    updated = _append_clarification_event(updated, prompt)
    updated = _clear_output_error(updated)
    updated["meta"] = set_clarification_meta(
        updated.get("meta", {}),
        return_node="rag_db_qa",
        kind="rag_db_extraction_opt_in",
        pending_question=prompt or None,
    )

    rag_state["pending_extraction_opt_in"] = pending_extraction_opt_in
    rag_state["thread_status"] = "awaiting_extraction_opt_in"
    rag_state["error"] = None
    return _finalize_rag_state(updated, rag_state, status="done", active_thread=True)


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

    return _handle_fresh_db_rag_question(
        state,
        rag_state,
        service=service,
        reranker_model=reranker_model,
        question=question,
    )
