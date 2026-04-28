from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langgraph.types import interrupt

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


def human_review_rag_db_column_selection_node(state):
    rag_state = dict((state.get("agents") or {}).get("rag_db_qa") or {})
    review = dict(rag_state.get("pending_column_review") or {})
    feedback = interrupt(
        {
            "type": "human_review_rag_db_column_selection",
            "question": review.get("question", ""),
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

    rag_state["pending_column_review"] = review
    updated_state = {
        **state,
        "agents": {
            **dict(state.get("agents") or {}),
            "rag_db_qa": rag_state,
        },
    }
    return update_agent_state(updated_state, "rag_db_qa", rag_state)
