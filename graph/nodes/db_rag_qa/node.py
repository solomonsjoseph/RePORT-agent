from __future__ import annotations

from types import SimpleNamespace

from ...state import AgentState
from ..state_helpers import clear_clarification_meta, get_agent_state, set_clarification_meta
from ..tool_routing import latest_user_message
from .helpers import (
    _SUPPORTED_PROVIDERS,
    _append_ai_response,
    _bootstrap_active_intent,
    _classify_opt_in_reply,
    _clear_output_error,
    _deserialize_column_selection,
    _deserialize_prepared_sql_candidate,
    _finalize_rag_state,
    _format_column_review_response,
    _format_sql_candidate_response,
    _intent_snapshot,
    _is_bare_acknowledgement,
    _looks_like_substantive_db_followup,
    _is_non_informative_followup,
    _question_from_stale_qa_followup,
    _read_value,
    _render_db_rag_recent_turns,
    _serialize_column_selection,
    _serialize_context_summary,
    _serialize_intent,
    _serialize_prepared_sql_candidate,
    _store_sql_candidate_output,
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
    pending_extraction_opt_in = dict(rag_state.get("pending_extraction_opt_in") or {})

    if pending_extraction_opt_in.get("status") == "awaiting_reply":
        intent = _serialize_intent(rag_state.get("active_intent") or {})
        goal_text = str(intent.get("goal_text") or pending_extraction_opt_in.get("goal_text") or "").strip()
        opt_in_status = _classify_opt_in_reply(question)
        if opt_in_status not in {"yes", "no"} and hasattr(service, "classify_pending_reply"):
            classified = service.classify_pending_reply(
                pending_kind="rag_db_extraction_opt_in",
                pending_question=str(pending_extraction_opt_in.get("question") or "").strip(),
                user_reply=question,
                recent_transcript=_render_db_rag_recent_turns(state, question),
            ) or {}
            opt_in_status = str(classified.get("label") or "").strip().lower() or "unknown"
        if opt_in_status == "yes":
            if not goal_text:
                updated = _append_ai_response(
                    state,
                    "I couldn't recover the extraction goal from this thread. Please restate the subset request.",
                )
                updated = _clear_output_error(updated)
                updated["meta"] = clear_clarification_meta(updated.get("meta") or {})
                rag_state.pop("pending_extraction_opt_in", None)
                rag_state.pop("pending_column_review", None)
                rag_state.pop("pending_sql_candidate", None)
                rag_state["thread_status"] = "answered_metadata"
                rag_state["error"] = None
                return _finalize_rag_state(updated, rag_state, status="done", active_thread=True)

            context = service.retrieve_context_for_intent(SimpleNamespace(**intent), reranker_model=reranker_model)
            selection = service.prepare_column_selection(
                goal_text,
                context,
                feedback_history=[],
                previous_selection=None,
                intent_snapshot=_intent_snapshot(intent),
            )
            review_payload = _serialize_column_selection(selection)
            review_payload["question"] = goal_text
            review_payload["goal_text"] = goal_text
            review_payload["status"] = "awaiting_review"

            updated = _append_ai_response(state, _format_column_review_response("", review_payload, revised=True))
            updated = _clear_output_error(updated)
            updated["meta"] = clear_clarification_meta(updated.get("meta") or {})
            rag_state["pending_column_review"] = review_payload
            rag_state["last_database_question"] = goal_text
            rag_state["last_retrieval_context"] = _serialize_context_summary(context)
            rag_state.pop("pending_extraction_opt_in", None)
            rag_state.pop("pending_sql_candidate", None)
            rag_state["thread_status"] = "awaiting_column_review"
            rag_state["error"] = None
            return _finalize_rag_state(updated, rag_state, status="done", active_thread=True)

        if opt_in_status == "no":
            updated = _append_ai_response(
                state,
                "Understood. I won't prepare table/column selection for extraction. Ask a metadata question anytime.",
            )
            updated = _clear_output_error(updated)
            updated["meta"] = clear_clarification_meta(updated.get("meta") or {})
            rag_state.pop("pending_extraction_opt_in", None)
            rag_state.pop("pending_column_review", None)
            rag_state.pop("pending_sql_candidate", None)
            rag_state["extraction_opt_out_goal_text"] = goal_text
            rag_state["thread_status"] = "answered_metadata"
            rag_state["error"] = None
            return _finalize_rag_state(updated, rag_state, status="done", active_thread=True)

        # If the user provides substantive follow-up content while opt-in is pending,
        # treat it as a new DB-RAG question instead of forcing a yes/no reply.
        if _looks_like_substantive_db_followup(question) and not _is_non_informative_followup(question):
            rag_state.pop("pending_extraction_opt_in", None)
            pending_extraction_opt_in = {}

        if pending_extraction_opt_in:
            updated = _append_ai_response(
                state,
                "Please reply with 'yes' to proceed with table/column selection for extraction, or 'no' to skip it.",
            )
            updated = _clear_output_error(updated)
            updated["meta"] = set_clarification_meta(
                updated.get("meta", {}),
                return_node="rag_db_qa",
                kind="rag_db_extraction_opt_in",
                pending_question=str(pending_extraction_opt_in.get("question") or "").strip() or None,
            )
            rag_state["thread_status"] = "awaiting_extraction_opt_in"
            rag_state["error"] = None
            return _finalize_rag_state(updated, rag_state, status="done", active_thread=True)

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

    active_intent = _serialize_intent(rag_state.get("active_intent") or {})
    if (
        _is_bare_acknowledgement(question)
        and not pending_extraction_opt_in
        and not pending_column_review
        and not pending_sql_candidate
        and active_intent.get("mode") == "extraction"
        and str(active_intent.get("goal_text") or "").strip()
    ):
        goal_text = str(active_intent.get("goal_text") or "").strip()
        context = service.retrieve_context_for_intent(SimpleNamespace(**active_intent), reranker_model=reranker_model)
        selection = service.prepare_column_selection(
            goal_text,
            context,
            feedback_history=[],
            previous_selection=None,
            intent_snapshot=_intent_snapshot(active_intent),
        )
        review_payload = _serialize_column_selection(selection)
        review_payload["question"] = goal_text
        review_payload["goal_text"] = goal_text
        review_payload["status"] = "awaiting_review"

        updated = _append_ai_response(state, _format_column_review_response("", review_payload, revised=True))
        updated = _clear_output_error(updated)
        updated["meta"] = clear_clarification_meta(updated.get("meta") or {})
        rag_state["last_database_question"] = goal_text
        rag_state["last_retrieval_context"] = _serialize_context_summary(context)
        rag_state["pending_column_review"] = review_payload
        rag_state.pop("pending_extraction_opt_in", None)
        rag_state.pop("pending_sql_candidate", None)
        rag_state["error"] = None
        rag_state["thread_status"] = "awaiting_column_review"
        return _finalize_rag_state(updated, rag_state, status="done", active_thread=True)

    active_intent = _serialize_intent(rag_state.get("active_intent") or {})
    preserve_active_intent = bool(
        active_intent.get("intent_id")
        and active_intent.get("mode")
        and _is_non_informative_followup(question)
    )

    effective_question = (
        str(active_intent.get("goal_text") or rag_state.get("last_database_question") or question).strip()
        if preserve_active_intent
        else question
    )
    retrieval_question = _render_db_rag_recent_turns(state, effective_question)

    context = service.retrieve_context(retrieval_question, reranker_model=reranker_model)
    answer = service.answer_from_context(retrieval_question, context)
    context_summary = _serialize_context_summary(context)
    if preserve_active_intent:
        intent = active_intent
    else:
        intent = _serialize_intent(
            service.resolve_intent(
                retrieval_question,
                context,
                prior_intent=dict(rag_state.get("active_intent") or {}),
            )
        )
    opt_out_goal_text = str(rag_state.get("extraction_opt_out_goal_text") or "").strip()
    if opt_out_goal_text and str(intent.get("goal_text") or "").strip() != opt_out_goal_text:
        rag_state.pop("extraction_opt_out_goal_text", None)

    rag_state["active_intent"] = intent
    rag_state["last_database_question"] = effective_question
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
    if opt_out_goal_text and str(intent.get("goal_text") or "").strip() == opt_out_goal_text:
        updated = _append_ai_response(state, str(_read_value(answer, "answer", "") or "").strip())
        updated = _clear_output_error(updated)
        updated["meta"] = clear_clarification_meta(updated.get("meta") or {})
        rag_state.pop("pending_extraction_opt_in", None)
        rag_state.pop("pending_column_review", None)
        rag_state.pop("pending_sql_candidate", None)
        rag_state["thread_status"] = "answered_metadata"
        return _finalize_rag_state(updated, rag_state, status="done", active_thread=True)

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
