from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langgraph.types import interrupt

from ..conversation_events import append_conversation_event, build_review_decision_event
from ..state import MetaKeys
from .state_helpers import update_agent_state

NODE_NAME = "human_review_rag_db_column_selection"
NODE_CAPABILITY = (
    "Ask a human to review the DB-RAG selected tables and columns before SQL generation."
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
        "category": "db_rag_review",
        "type": "MissingColumnSelectionArtifact",
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
        "pending_column_review_artifact_id": None,
        "pending_column_review": None,
        "approved_column_selection_artifact_id": None,
        "pending_sql_candidate_artifact_id": None,
        "pending_sql_candidate": None,
        "error": error_payload,
        "thread_status": "error",
        "active_thread": False,
        "status": "error",
    }
    return update_agent_state(updated_state, "rag_db_qa", errored_rag_state)


def human_review_rag_db_column_selection_node(state):
    rag_state = dict((state.get("agents") or {}).get("rag_db_qa") or {})
    selection_artifact_id = str(rag_state.get("pending_column_review_artifact_id") or "").strip()
    review = _artifact_content(state, selection_artifact_id)
    if not selection_artifact_id or not review:
        return _review_artifact_error(
            state,
            rag_state,
            message="The pending DB-RAG column review state is missing its selection artifact. Please restart the DB-RAG request.",
        )
    feedback = interrupt(
        {
            "type": "human_review_rag_db_column_selection",
            "artifact_id": selection_artifact_id,
            "goal_text": review.get("goal_text", ""),
            "question": review.get("source_question", ""),
            "selection_id": review.get("selection_id", ""),
            "tables": list(review.get("tables") or []),
            "columns": list(review.get("columns") or []),
            "rationale": review.get("rationale", ""),
            "feedback_history": list(review.get("feedback_history") or []),
        }
    )

    action = str(feedback.get("action") or "").strip().lower()
    feedback_text = (
        str(feedback.get("feedback") or feedback.get("suggestion") or feedback.get("message") or "").strip()
    )

    if action == "approve":
        review["status"] = "approved"
        rag_state["approved_column_selection_artifact_id"] = selection_artifact_id
        rag_state["pending_column_review_artifact_id"] = None
        rag_state["pending_sql_candidate_artifact_id"] = None
        rag_state.pop("pending_sql_candidate", None)
        rag_state["thread_status"] = "awaiting_sql_generation"
    else:
        review["status"] = "needs_revision"
        explanation = (
            "Human requested regeneration."
            if action == "regenerate"
            else f'Received unsupported action "{action}" during DB-RAG column review; treating it as regeneration.'
        )
        if feedback_text:
            feedback_text = f"{explanation} Additional context: {feedback_text}"
        else:
            feedback_text = explanation
        history = list(review.get("feedback_history") or [])
        history.append(_review_feedback_entry(action or "regenerate", feedback_text))
        review["feedback_history"] = history
        rag_state["approved_column_selection_artifact_id"] = None
        rag_state["pending_column_review_artifact_id"] = selection_artifact_id or None
        rag_state["pending_sql_candidate_artifact_id"] = None
        rag_state.pop("pending_sql_candidate", None)
        rag_state["thread_status"] = "awaiting_column_review"

    updated_state = _replace_artifact_content(
        state,
        artifact_id=selection_artifact_id,
        content=review,
    )
    rag_state["pending_column_review"] = review
    meta = dict(updated_state.get("meta") or {})
    updated_state = append_conversation_event(
        updated_state,
        build_review_decision_event(
            actor="human_review_rag_db_column_selection",
            user_turn_hash=str(meta.get(MetaKeys.LAST_USER_MESSAGE_HASH) or "").strip() or None,
            review_kind="rag_db_column_selection",
            decision=action or "regenerate",
            text=feedback_text or ("Approved DB-RAG column selection." if action == "approve" else "Requested DB-RAG column selection revision."),
            status="done",
        ),
    )
    return update_agent_state(updated_state, "rag_db_qa", rag_state)
