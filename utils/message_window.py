from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage


def _has_tool_call_metadata(message: BaseMessage) -> bool:
    direct_tool_calls = getattr(message, "tool_calls", None)
    if isinstance(direct_tool_calls, list) and bool(direct_tool_calls):
        return True

    invalid_tool_calls = getattr(message, "invalid_tool_calls", None)
    if isinstance(invalid_tool_calls, list) and bool(invalid_tool_calls):
        return True

    additional_kwargs = dict(getattr(message, "additional_kwargs", {}) or {})
    tool_calls = additional_kwargs.get("tool_calls")
    return isinstance(tool_calls, list) and bool(tool_calls)


def _is_tool_trace_message(message: BaseMessage) -> bool:
    if getattr(message, "type", None) == "tool":
        return True
    return getattr(message, "type", None) == "ai" and _has_tool_call_metadata(message)


def compact_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Drop blank assistant turns that should not influence later prompts/UI."""
    compacted: list[BaseMessage] = []
    for msg in messages or []:
        if (
            isinstance(msg, AIMessage)
            and not str(getattr(msg, "content", "") or "").strip()
            and not _has_tool_call_metadata(msg)
        ):
            continue
        compacted.append(msg)
    return compacted


def window_messages(
    messages: list[BaseMessage],
    max_turns: int = 10,
) -> list[BaseMessage]:
    """Return the last ``max_turns`` human+AI pairs from *messages*.

    The full message list in ``AgentState`` is never modified — this helper is
    only called when constructing LLM prompts so that token cost stays bounded
    regardless of conversation length.

    A "turn" is defined as one or more human messages followed by an AI reply.
    Any trailing human messages that have not yet received a reply are treated as
    one partial turn and are always included.

    Internal tool-call scaffolding from prior completed turns is stripped before
    windowing so downstream model calls receive only user-visible conversation
    history rather than provider-specific tool protocol messages.
    """
    messages = compact_messages(messages)
    messages = [msg for msg in messages if not _is_tool_trace_message(msg)]
    if not messages:
        return []

    # Group messages into turns. A turn is one or more human messages followed
    # by any assistant/tool exchange that answers them. Tool call chains must
    # stay attached to the prompting human turn.
    turns: list[list[BaseMessage]] = []
    current_turn: list[BaseMessage] = []
    for msg in messages:
        if (
            getattr(msg, "type", None) == "human"
            and current_turn
            and any(getattr(existing, "type", None) != "human" for existing in current_turn)
        ):
            turns.append(current_turn)
            current_turn = []
        current_turn.append(msg)

    if current_turn:
        turns.append(current_turn)

    selected = turns[-max_turns:]
    return [msg for turn in selected for msg in turn]
