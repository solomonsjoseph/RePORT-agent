from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langgraph.types import interrupt

from ..conversation_events import append_conversation_event, build_review_decision_event
from ..state import MetaKeys
from .human_review_cancel import append_review_cancel_audit
from .rag_db_qa import _deserialize_prepared_sql_candidate, _execute_prepared_sql_candidate
from .state_helpers import update_agent_state

NODE_NAME = "human_review_rag_db_sql_execution"
NODE_CAPABILITY = (
    "Ask a human to review and explicitly approve the prepared DB-RAG SQL before execution."
)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _review_feedback_entry(action: str, feedback_text: str | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "timestamp": _utc_timestamp(),
        "action": action,
    }
    if feedback_text:
        entry["feedback"] = feedback_text
    return entry


def _artifact_content(state: dict[str, Any], artifact_id: str | None) -> dict[str, Any]:
    artifact_key = str(artifact_id or "").strip()
    if not artifact_key:
        return {}
    artifacts = dict(state.get("artifacts") or {})
    files = dict(artifacts.get("files") or {})
    artifact = dict(files.get(artifact_key) or {})
    content = artifact.get("content")
    return dict(content) if isinstance(content, dict) else {}


def _replace_artifact_content(
    state: dict[str, Any],
    *,
    artifact_id: str | None,
    content: dict[str, Any],
) -> dict[str, Any]:
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


def _review_artifact_error(
    state: dict[str, Any],
    rag_state: dict[str, Any],
    *,
    message: str,
) -> dict[str, Any]:
    output = dict(state.get("output") or {})
    error_payload = {
        "category": "db_rag_sql",
        "type": "MissingSqlReviewArtifact",
        "message": message,
    }
    output["qa_response"] = message
    output["error"] = error_payload
    updated_state = {
        **state,
        "output": output,
    }
    errored_rag_state = {
        **rag_state,
        "pending_sql_candidate_artifact_id": None,
        "pending_sql_candidate": None,
        "pending_column_review_artifact_id": None,
        "approved_column_selection_artifact_id": None,
        "pending_column_review": None,
        "error": error_payload,
        "thread_status": "error",
        "active_thread": False,
        "status": "error",
    }
    return update_agent_state(updated_state, "rag_db_qa", errored_rag_state)


def human_review_rag_db_sql_execution_node(state, service):
    rag_state = dict((state.get("agents") or {}).get("rag_db_qa") or {})
    candidate_artifact_id = str(rag_state.get("pending_sql_candidate_artifact_id") or "").strip()
    candidate_payload = _artifact_content(state, candidate_artifact_id)
    if not candidate_artifact_id or not candidate_payload:
        return _review_artifact_error(
            state,
            rag_state,
            message="The pending DB-RAG SQL review state is missing its prepared SQL artifact. Please restart the DB-RAG request.",
        )
    candidate = _deserialize_prepared_sql_candidate(candidate_payload)
    selection_artifact_id = str(
        candidate_payload.get("selection_artifact_id")
        or rag_state.get("approved_column_selection_artifact_id")
        or rag_state.get("pending_column_review_artifact_id")
        or ""
    ).strip()
    review = _artifact_content(state, selection_artifact_id)
    if not selection_artifact_id or not review:
        return _review_artifact_error(
            state,
            rag_state,
            message="The pending DB-RAG SQL review state is missing its reviewed column-selection artifact. Please restart the DB-RAG request.",
        )
    question = str(
        candidate_payload.get("question")
        or candidate_payload.get("goal_text")
        or candidate_payload.get("source_question")
        or review.get("goal_text")
        or review.get("source_question")
        or ""
    ).strip()

    feedback = interrupt(
        {
            "type": "human_review_rag_db_sql_execution",
            "artifact_id": candidate_artifact_id,
            "selection_artifact_id": selection_artifact_id,
            "question": question,
            "selection_id": getattr(candidate, "selection_id", None) or review.get("selection_id", ""),
            "tables": list(getattr(candidate, "tables", None) or review.get("tables") or []),
            "columns": list(getattr(candidate, "columns", None) or review.get("columns") or []),
            "rationale": review.get("rationale", ""),
            "sql": getattr(candidate, "sql", ""),
            "feedback_history": list(review.get("feedback_history") or []),
        }
    )

    action = str(feedback.get("action") or "").strip().lower()
    suggestion = str(feedback.get("suggestion") or feedback.get("feedback") or feedback.get("message") or "").strip()

    if action == "cancel":
        candidate_payload["status"] = "cancelled"
        updated_state = _replace_artifact_content(
            state,
            artifact_id=candidate_artifact_id,
            content=candidate_payload,
        )
        updated_state = append_review_cancel_audit(
            updated_state,
            actor="human_review_rag_db_sql_execution",
            review_kind="rag_db_sql_execution",
        )
        rag_state["pending_sql_candidate_artifact_id"] = None
        rag_state["pending_sql_candidate"] = None
        rag_state["approved_column_selection_artifact_id"] = None
        rag_state["pending_column_review_artifact_id"] = None
        rag_state["pending_column_review"] = None
        rag_state.pop("sql_review_approved_artifact_id", None)
        rag_state["thread_status"] = "cancelled"
        rag_state["active_thread"] = False
        agents = dict(updated_state.get("agents") or {})
        base_rag_state = dict(agents.get("rag_db_qa") or {})
        base_rag_state.pop("sql_review_approved_artifact_id", None)
        agents["rag_db_qa"] = base_rag_state
        updated_state = {
            **updated_state,
            "agents": agents,
        }
        return update_agent_state(updated_state, "rag_db_qa", rag_state)

    if action == "approve":
        updated_state = append_conversation_event(
            state,
            build_review_decision_event(
                actor="human_review_rag_db_sql_execution",
                user_turn_hash=str((state.get("meta") or {}).get(MetaKeys.LAST_USER_MESSAGE_HASH) or "").strip() or None,
                review_kind="rag_db_sql_execution",
                decision="approve",
                text="Approved prepared DB-RAG SQL for execution.",
                status="done",
            ),
        )
        execution_rag_state = {
            **rag_state,
            "pending_sql_candidate_artifact_id": candidate_artifact_id or None,
            "pending_sql_candidate": candidate_payload,
            "pending_column_review_artifact_id": selection_artifact_id or None,
            "approved_column_selection_artifact_id": selection_artifact_id or None,
            "pending_column_review": review,
            "sql_review_approved_artifact_id": candidate_artifact_id or None,
        }
        executed_state = _execute_prepared_sql_candidate(updated_state, execution_rag_state, candidate, service)
        executed_rag_state = dict((executed_state.get("agents") or {}).get("rag_db_qa") or {})
        executed_rag_state.pop("sql_review_approved_artifact_id", None)
        thread_status = str(executed_rag_state.get("thread_status") or "").strip()
        if thread_status in {"completed", "executed"}:
            executed_rag_state["thread_status"] = "completed"
            executed_rag_state["pending_sql_candidate_artifact_id"] = None
            executed_rag_state["pending_sql_candidate"] = None
            executed_rag_state["pending_column_review_artifact_id"] = None
            executed_rag_state["approved_column_selection_artifact_id"] = None
            executed_rag_state["pending_column_review"] = None
        elif thread_status == "error":
            executed_rag_state["pending_sql_candidate_artifact_id"] = None
            executed_rag_state["pending_sql_candidate"] = None
            executed_rag_state["pending_column_review_artifact_id"] = selection_artifact_id or None
            executed_rag_state["approved_column_selection_artifact_id"] = None
        return update_agent_state(executed_state, "rag_db_qa", executed_rag_state)

    explanation = (
        "Human requested SQL regeneration."
        if action == "regenerate"
        else f'Received unsupported action "{action}" during DB-RAG SQL review; treating it as regeneration.'
    )
    if suggestion:
        suggestion = f"{explanation} Additional context: {suggestion}"
    else:
        suggestion = explanation

    decision = (
        service.classify_sql_review_feedback(
            feedback_text=suggestion,
            sql=str(candidate_payload.get("sql") or ""),
        )
        if hasattr(service, "classify_sql_review_feedback")
        else {"label": "unknown", "confidence": 0.0}
    )
    label = str(decision.get("label") or "unknown").strip().lower()

    updated_state = state
    history = list(review.get("feedback_history") or [])
    history.append(_review_feedback_entry(action or "regenerate", suggestion))
    review["feedback_history"] = history
    if label in {"selection_revision", "unknown"}:
        review["status"] = "needs_revision"
        updated_state = _replace_artifact_content(
            updated_state,
            artifact_id=selection_artifact_id,
            content=review,
        )
        rag_state["approved_column_selection_artifact_id"] = None
        rag_state["pending_column_review_artifact_id"] = selection_artifact_id or None
        rag_state["pending_column_review"] = review
        rag_state["thread_status"] = "awaiting_column_review"
    else:
        review["status"] = "approved"
        updated_state = _replace_artifact_content(
            updated_state,
            artifact_id=selection_artifact_id,
            content=review,
        )
        rag_state["approved_column_selection_artifact_id"] = selection_artifact_id or None
        rag_state["pending_column_review_artifact_id"] = None
        rag_state["pending_column_review"] = review
        rag_state["thread_status"] = "awaiting_sql_generation"

    rag_state["pending_sql_candidate_artifact_id"] = None
    rag_state.pop("pending_sql_candidate", None)

    updated_state = append_conversation_event(
        updated_state,
        build_review_decision_event(
            actor="human_review_rag_db_sql_execution",
            user_turn_hash=str((state.get("meta") or {}).get(MetaKeys.LAST_USER_MESSAGE_HASH) or "").strip() or None,
            review_kind="rag_db_sql_execution",
            decision=action or "regenerate",
            text=suggestion,
            status="done",
        ),
    )
    return update_agent_state(updated_state, "rag_db_qa", rag_state)
