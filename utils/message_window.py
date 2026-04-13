from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage


def compact_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Drop blank assistant turns that should not influence later prompts/UI."""
    compacted: list[BaseMessage] = []
    for msg in messages or []:
        if isinstance(msg, AIMessage) and not str(getattr(msg, "content", "") or "").strip():
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
    """
    messages = compact_messages(messages)
    if not messages:
        return []

    # Group messages into turns; a turn closes when an AI message is seen.
    turns: list[list[BaseMessage]] = []
    current_turn: list[BaseMessage] = []
    for msg in messages:
        current_turn.append(msg)
        if isinstance(msg, AIMessage):
            turns.append(current_turn)
            current_turn = []

    # Trailing human messages without an AI reply = one partial turn.
    if current_turn:
        turns.append(current_turn)

    selected = turns[-max_turns:]
    return [msg for turn in selected for msg in turn]
