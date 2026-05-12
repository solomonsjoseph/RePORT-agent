from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from graph.state_views import get_artifact_files, get_conversation_events
from utils.message_window import compact_messages


def _figure_paths_by_parent_event_id(state: dict) -> dict[str, str]:
    artifact_files = get_artifact_files(state)
    figure_paths: dict[str, str] = {}
    for event in get_conversation_events(state):
        if event.get("type") != "figure":
            continue
        parent_event_id = event.get("parent_event_id")
        artifact_id = event.get("artifact_id")
        if not isinstance(parent_event_id, str) or not parent_event_id.strip():
            continue
        if not isinstance(artifact_id, str) or artifact_id not in artifact_files:
            continue
        content = dict(artifact_files[artifact_id].get("content") or {})
        path_value = content.get("path")
        if isinstance(path_value, str) and path_value:
            figure_paths[parent_event_id] = path_value
    return figure_paths


def _contains_normalized_text(haystack: str, needle: str) -> bool:
    normalized_haystack = " ".join(str(haystack or "").split())
    normalized_needle = " ".join(str(needle or "").split())
    return bool(normalized_needle and normalized_needle in normalized_haystack)


def build_display_history(state: dict) -> list:
    events = get_conversation_events(state)
    if not events:
        return compact_messages(list(state.get("messages", [])))

    figure_paths = _figure_paths_by_parent_event_id(state)
    display_messages: list = []
    assistant_text_by_turn_hash: dict[str, str] = {}
    for event in events:
        event_type = event.get("type")
        text = str(event.get("text") or "").strip()
        if event_type == "user":
            if text:
                display_messages.append(HumanMessage(content=text))
            continue
        if event_type == "review_decision":
            if event.get("decision") == "cancel":
                continue
            if text:
                display_messages.append(HumanMessage(content=text))
            continue
        if event_type not in {"assistant", "clarification"}:
            continue
        user_turn_hash = event.get("user_turn_hash")
        if (
            event_type == "clarification"
            and isinstance(user_turn_hash, str)
            and _contains_normalized_text(
                assistant_text_by_turn_hash.get(user_turn_hash, ""),
                text,
            )
        ):
            continue
        additional_kwargs = {}
        event_id = event.get("event_id")
        if isinstance(event_id, str) and event_id in figure_paths:
            additional_kwargs["figure_path"] = figure_paths[event_id]
        display_messages.append(
            AIMessage(
                content=text,
                additional_kwargs=additional_kwargs,
            )
        )
        if event_type == "assistant" and isinstance(user_turn_hash, str):
            assistant_text_by_turn_hash[user_turn_hash] = text
    return display_messages


def serialize_display_history(messages: list) -> list[dict[str, object]]:
    serialized: list[dict[str, object]] = []
    for index, message in enumerate(messages, start=1):
        role = "assistant"
        if getattr(message, "type", None) == "human" or isinstance(message, HumanMessage):
            role = "user"
        serialized.append(
            {
                "index": index,
                "role": role,
                "content": str(getattr(message, "content", "") or ""),
                "additional_kwargs": dict(getattr(message, "additional_kwargs", {}) or {}),
            }
        )
    return serialized
