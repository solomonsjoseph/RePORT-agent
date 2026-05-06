from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConversationControlsState:
    render: bool
    submission_blocked: bool
    run_in_progress: bool


def conversation_controls_state(
    run_status: Mapping[str, Any] | None,
    *,
    review_blocked: bool,
) -> ConversationControlsState:
    state = (run_status or {}).get("state")
    run_in_progress = state == "running"
    run_blocks_submission = state in {"running", "error"}
    return ConversationControlsState(
        render=True,
        submission_blocked=bool(review_blocked or run_blocks_submission),
        run_in_progress=run_in_progress,
    )


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


def canonicalize_conversation_history(
    history: list[Any],
    *,
    welcome_message: str,
) -> list[Any]:
    messages = list(history)
    has_human_turn = any(
        getattr(msg, "type", None) == "human"
        for msg in messages
    )
    if not has_human_turn:
        return messages
    return [
        msg
        for msg in messages
        if not (
            getattr(msg, "type", None) == "ai"
            and str(getattr(msg, "content", "") or "").strip() == welcome_message
        )
    ]
