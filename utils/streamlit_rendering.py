from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def should_render_conversation_controls(run_status: Mapping[str, Any] | None) -> bool:
    state = (run_status or {}).get("state")
    return state not in {"running", "error"}


def normalize_submitted_question(raw_text: object, *, blocked: bool) -> str | None:
    if blocked:
        return None
    text = str(raw_text or "").strip()
    return text or None


def conversation_history_with_pending_user(
    history: list[Any],
    pending_user_message: Any | None,
    *,
    welcome_message: str,
) -> list[Any]:
    messages = list(history)
    if pending_user_message is None:
        return messages
    messages = [
        msg
        for msg in messages
        if str(getattr(msg, "content", "") or "").strip() != welcome_message
    ]
    messages.append(pending_user_message)
    return messages
