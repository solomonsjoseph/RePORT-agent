from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage

from ..conversation_events import append_conversation_event, build_review_decision_event
from ..state import MetaKeys

CANCEL_REVIEW_MESSAGE = "Cancelled the pending review. You can start a new request when ready."


def append_review_cancel_audit(
    state: dict[str, Any],
    *,
    actor: str,
    review_kind: str,
) -> dict[str, Any]:
    messages = list(state.get("messages") or [])
    messages.append(AIMessage(content=CANCEL_REVIEW_MESSAGE))
    updated_state = {
        **state,
        "messages": messages,
    }
    meta = dict(updated_state.get("meta") or {})
    return append_conversation_event(
        updated_state,
        build_review_decision_event(
            actor=actor,
            user_turn_hash=str(meta.get(MetaKeys.LAST_USER_MESSAGE_HASH) or "").strip() or None,
            review_kind=review_kind,
            decision="cancel",
            text=CANCEL_REVIEW_MESSAGE,
            status="done",
        ),
    )
