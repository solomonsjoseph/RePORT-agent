from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConversationControlsState:
    render: bool
    submission_blocked: bool
    run_in_progress: bool


@dataclass(frozen=True)
class ConversationRenderState:
    messages: tuple[Any, ...]
    status_level: str | None
    status_text: str | None
    hide_secondary_sections: bool


def conversation_controls_state(
    run_status: Mapping[str, Any] | None,
    *,
    review_blocked: bool,
) -> ConversationControlsState:
    state = (run_status or {}).get("state")
    run_in_progress = state == "running"
    run_blocks_submission = state in {"running", "error"}
    return ConversationControlsState(
        render=not run_blocks_submission,
        submission_blocked=bool(review_blocked or run_blocks_submission),
        run_in_progress=run_in_progress,
    )


def should_render_compact_running_view(
    run_status: Mapping[str, Any] | None,
    *,
    has_review_interrupt: bool,
) -> bool:
    state = (run_status or {}).get("state")
    return state == "running" and not has_review_interrupt


def conversation_render_state(
    history: list[Any],
    *,
    run_status: Mapping[str, Any] | None,
    has_review_interrupt: bool,
) -> ConversationRenderState:
    compact_running = should_render_compact_running_view(
        run_status,
        has_review_interrupt=has_review_interrupt,
    )
    state = (run_status or {}).get("state")
    status_level = None
    status_text = None
    if state == "error":
        status_level = "error"
        status_text = f"Background workflow failed: {(run_status or {}).get('error') or 'unknown error'}"
    elif state == "running":
        status_level = "info"
        status_text = "⏳ Working in background..."
    return ConversationRenderState(
        messages=tuple(history),
        status_level=status_level,
        status_text=status_text,
        hide_secondary_sections=compact_running,
    )


def build_autoscroll_key(
    history: list[Any],
    *,
    ui_type: str | None,
    should_render_interrupt: bool,
) -> str:
    latest_content = str(getattr(history[-1], "content", "") if history else "")[:120]
    return (
        f"{len(history)}:"
        f"{latest_content}:"
        f"{ui_type or ''}:"
        f"{int(should_render_interrupt)}"
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
    pending_text = str(getattr(pending_user_message, "content", "") or "").strip()
    if pending_user_text_caught_up(messages, pending_text):
        return messages
    messages = [
        msg
        for msg in messages
        if str(getattr(msg, "content", "") or "").strip() != welcome_message
    ]
    messages.append(pending_user_message)
    return messages


def pending_user_text_caught_up(history: list[Any], pending_text: str | None) -> bool:
    normalized_pending = str(pending_text or "").strip()
    if not normalized_pending:
        return True
    return any(
        getattr(msg, "type", None) == "human"
        and str(getattr(msg, "content", "") or "").strip() == normalized_pending
        for msg in history
    )


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
