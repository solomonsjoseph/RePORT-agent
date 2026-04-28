from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

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


def human_review_rag_db_sql_execution_node(state, service):
    rag_state = dict((state.get("agents") or {}).get("rag_db_qa") or {})
    review = dict(rag_state.get("pending_column_review") or {})
    candidate_payload = dict(rag_state.get("pending_sql_candidate") or {})
    candidate = _deserialize_prepared_sql_candidate(candidate_payload)

    feedback = interrupt(
        {
            "type": "human_review_rag_db_sql_execution",
            "question": candidate.question or review.get("question", ""),
            "selection_id": candidate.selection_id or review.get("selection_id", ""),
            "tables": list(candidate.tables or review.get("tables") or []),
            "columns": list(candidate.columns or review.get("columns") or []),
            "rationale": review.get("rationale", ""),
            "sql": candidate.sql,
            "feedback_history": list(review.get("feedback_history") or []),
        }
    )

    action = str(feedback.get("action") or "").strip().lower()
    suggestion = str(feedback.get("suggestion") or feedback.get("feedback") or feedback.get("message") or "").strip()

    if action == "approve":
        return _execute_prepared_sql_candidate(state, rag_state, candidate, service)

    explanation = (
        "Human requested SQL regeneration."
        if action == "regenerate"
        else f'Received unsupported action "{action}" during DB-RAG SQL review; treating it as regeneration.'
    )
    if suggestion:
        suggestion = f"{explanation} Additional context: {suggestion}"
    else:
        suggestion = explanation

    messages = list(state.get("messages", []))
    messages.append(HumanMessage(content=suggestion))

    history = list(review.get("feedback_history") or [])
    history.append(_review_feedback_entry(action or "regenerate", suggestion))
    review["status"] = "needs_revision"
    review["feedback_history"] = history
    rag_state["pending_column_review"] = review
    rag_state.pop("pending_sql_candidate", None)

    updated_state = {
        **state,
        "messages": messages,
        "agents": {
            **dict(state.get("agents") or {}),
            "rag_db_qa": rag_state,
        },
    }
    return update_agent_state(updated_state, "rag_db_qa", rag_state)
