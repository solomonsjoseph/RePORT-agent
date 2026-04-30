from __future__ import annotations

import sys
from types import ModuleType


def _install_message_stubs() -> None:
    messages_mod = ModuleType("langchain_core.messages")

    class _HumanMessage:
        def __init__(self, content: str, additional_kwargs: dict | None = None):
            self.type = "human"
            self.content = content
            self.additional_kwargs = additional_kwargs or {}

    class _AIMessage:
        def __init__(self, content: str, additional_kwargs: dict | None = None):
            self.type = "ai"
            self.content = content
            self.additional_kwargs = additional_kwargs or {}

    messages_mod.BaseMessage = object
    messages_mod.HumanMessage = _HumanMessage
    messages_mod.AIMessage = _AIMessage
    sys.modules["langchain_core.messages"] = messages_mod


_install_message_stubs()

from utils.display_history import build_display_history, serialize_display_history


def test_build_display_history_projects_user_visible_events_only() -> None:
    state = {
        "messages": [],
        "artifacts": {
            "conversation_events": [
                {
                    "event_id": "e1",
                    "seq": 1,
                    "created_at": "2026-04-30T00:00:00Z",
                    "type": "user",
                    "actor": "human",
                    "actor_role": "user",
                    "user_turn_hash": "u1",
                    "text": "what's the weather?",
                },
                {
                    "event_id": "e2",
                    "seq": 2,
                    "created_at": "2026-04-30T00:00:01Z",
                    "type": "tool_call",
                    "actor": "qa",
                    "actor_role": "assistant",
                    "user_turn_hash": "u1",
                    "tool": "query_weather",
                    "args": {"city": "Boston"},
                },
                {
                    "event_id": "e3",
                    "seq": 3,
                    "created_at": "2026-04-30T00:00:02Z",
                    "type": "tool_result",
                    "actor": "qa",
                    "actor_role": "assistant",
                    "user_turn_hash": "u1",
                    "tool": "query_weather",
                    "artifact_id": None,
                    "text": "Boston weather payload",
                },
                {
                    "event_id": "e4",
                    "seq": 4,
                    "created_at": "2026-04-30T00:00:03Z",
                    "type": "assistant",
                    "actor": "qa",
                    "actor_role": "assistant",
                    "user_turn_hash": "u1",
                    "text": "Boston is cool today.",
                },
            ],
            "conversation_events_version": 1,
            "artifact_manifest_version": 1,
            "files": {},
        },
        "meta": {"next_event_seq": 5},
    }

    history = build_display_history(state)

    assert [message.type for message in history] == ["human", "ai"]
    assert history[0].content == "what's the weather?"
    assert history[1].content == "Boston is cool today."


def test_build_display_history_attaches_figure_to_parent_assistant_event() -> None:
    state = {
        "messages": [],
        "artifacts": {
            "conversation_events": [
                {
                    "event_id": "e1",
                    "seq": 1,
                    "created_at": "2026-04-30T00:00:00Z",
                    "type": "assistant",
                    "actor": "human_review_before_output",
                    "actor_role": "assistant",
                    "user_turn_hash": "u2",
                    "text": "Approved final output.",
                },
                {
                    "event_id": "e2",
                    "seq": 2,
                    "created_at": "2026-04-30T00:00:01Z",
                    "type": "figure",
                    "actor": "human_review_before_output",
                    "actor_role": "executor",
                    "user_turn_hash": "u2",
                    "artifact_id": "fig-1",
                    "text": "Approved figure output.",
                    "parent_event_id": "e1",
                },
            ],
            "conversation_events_version": 1,
            "artifact_manifest_version": 1,
            "files": {
                "fig-1": {
                    "artifact_id": "fig-1",
                    "kind": "figure",
                    "producer": "executor",
                    "mime": "image/png",
                    "summary": "figure",
                    "created_at": "2026-04-30T00:00:01Z",
                    "content": {"path": "/tmp/final.png"},
                }
            },
        },
        "meta": {"next_event_seq": 3},
    }

    history = build_display_history(state)

    assert len(history) == 1
    assert history[0].type == "ai"
    assert history[0].content == "Approved final output."
    assert history[0].additional_kwargs["figure_path"] == "/tmp/final.png"


def test_serialize_display_history_returns_user_visible_payload() -> None:
    messages = [
        sys.modules["langchain_core.messages"].HumanMessage(content="question"),
        sys.modules["langchain_core.messages"].AIMessage(
            content="answer",
            additional_kwargs={"figure_path": "/tmp/plot.png"},
        ),
    ]

    serialized = serialize_display_history(messages)

    assert serialized == [
        {
            "index": 1,
            "role": "user",
            "content": "question",
            "additional_kwargs": {},
        },
        {
            "index": 2,
            "role": "assistant",
            "content": "answer",
            "additional_kwargs": {"figure_path": "/tmp/plot.png"},
        },
    ]
