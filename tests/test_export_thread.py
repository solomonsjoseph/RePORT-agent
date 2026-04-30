from __future__ import annotations

import io
import json
import sys
import tempfile
import zipfile
from pathlib import Path
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

from langchain_core.messages import AIMessage, HumanMessage

from utils.export_thread import build_thread_export


def _read_zip(payload: bytes) -> tuple[set[str], dict[str, bytes]]:
    with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
        names = set(archive.namelist())
        contents = {name: archive.read(name) for name in names}
    return names, contents


def test_build_thread_export_uses_semantic_events_and_artifact_manifest() -> None:
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        handle.write(b"png-bytes")
        figure_path = handle.name

    state = {
        "messages": [
            HumanMessage(content="make a plot"),
            AIMessage(content="legacy runtime trace"),
        ],
        "output": {
            "generated_code": "print('stale output field')",
            "text": "stale text",
            "figure_artifact_id": "fig-1",
        },
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
                    "text": "make a plot",
                },
                {
                    "event_id": "e2",
                    "seq": 2,
                    "created_at": "2026-04-30T00:00:01Z",
                    "type": "assistant",
                    "actor": "human_review_before_output",
                    "actor_role": "assistant",
                    "user_turn_hash": "u1",
                    "text": (
                        "Summary.\n\n"
                        "Generated code:\n```python\nprint('hello')\n```\n\n"
                        "Output:\n```\nhello\n```"
                    ),
                },
                {
                    "event_id": "e3",
                    "seq": 3,
                    "created_at": "2026-04-30T00:00:02Z",
                    "type": "figure",
                    "actor": "human_review_before_output",
                    "actor_role": "executor",
                    "user_turn_hash": "u1",
                    "artifact_id": "fig-1",
                    "text": "Approved figure output.",
                    "parent_event_id": "e2",
                },
            ],
            "conversation_events_version": 1,
            "artifact_manifest_version": 1,
            "files": {
                "code-1": {
                    "artifact_id": "code-1",
                    "kind": "code",
                    "producer": "generate_code",
                    "mime": "text/x-python",
                    "summary": "Generated code",
                    "created_at": "2026-04-30T00:00:00Z",
                    "content": "print('hello')\n",
                },
                "text-1": {
                    "artifact_id": "text-1",
                    "kind": "text",
                    "producer": "executor",
                    "mime": "text/plain",
                    "summary": "Execution output",
                    "created_at": "2026-04-30T00:00:00Z",
                    "content": "hello\n",
                },
                "fig-1": {
                    "artifact_id": "fig-1",
                    "kind": "figure",
                    "producer": "executor",
                    "mime": "image/png",
                    "summary": "Figure",
                    "created_at": "2026-04-30T00:00:00Z",
                    "content": {"path": figure_path},
                },
            },
        },
        "meta": {"next_event_seq": 4},
    }

    payload = build_thread_export(
        thread_id="thread-1",
        provider="openai",
        model_name="gpt-test",
        state=state,
    )

    names, contents = _read_zip(payload)

    assert "conversation.json" in names
    assert "conversation.md" in names
    assert "artifacts.json" in names
    assert "artifacts/code-1.py" in names
    assert "artifacts/text-1.txt" in names
    assert "artifacts/fig-1.png" in names
    assert contents["artifacts/code-1.py"] == b"print('hello')\n"
    assert contents["artifacts/text-1.txt"] == b"hello\n"
    assert contents["artifacts/fig-1.png"] == b"png-bytes"

    manifest = json.loads(contents["artifacts.json"].decode("utf-8"))
    assert manifest == [
        {
            "artifact_id": "code-1",
            "kind": "code",
            "producer": "generate_code",
            "mime": "text/x-python",
            "summary": "Generated code",
            "created_at": "2026-04-30T00:00:00Z",
            "filename": "artifacts/code-1.py",
            "content_source": "artifacts.files",
        },
        {
            "artifact_id": "text-1",
            "kind": "text",
            "producer": "executor",
            "mime": "text/plain",
            "summary": "Execution output",
            "created_at": "2026-04-30T00:00:00Z",
            "filename": "artifacts/text-1.txt",
            "content_source": "artifacts.files",
        },
        {
            "artifact_id": "fig-1",
            "kind": "figure",
            "producer": "executor",
            "mime": "image/png",
            "summary": "Figure",
            "created_at": "2026-04-30T00:00:00Z",
            "filename": "artifacts/fig-1.png",
            "content_source": "artifacts.files",
        },
    ]

    conversation = json.loads(contents["conversation.json"].decode("utf-8"))
    assert conversation["conversation"] == [
        {
            "index": 1,
            "role": "human",
            "content": "make a plot",
            "additional_kwargs": {},
        },
        {
            "index": 2,
            "role": "ai",
            "content": (
                "Summary.\n\n"
                "Generated code:\n```python\nprint('hello')\n```\n\n"
                "Output:\n```\nhello\n```"
            ),
            "additional_kwargs": {"figure_path": figure_path},
        },
    ]
    assert len(conversation["conversation_events"]) == 3
    assert len(conversation["runtime_messages"]) == 2
    Path(figure_path).unlink(missing_ok=True)


def test_build_thread_export_keeps_conversation_exports_without_artifacts() -> None:
    state = {
        "messages": [
            HumanMessage(content="run analysis"),
            AIMessage(content="legacy runtime trace"),
        ],
        "output": {
            "generated_code": "print('not approved')",
            "text": "not in chat history",
            "figure_artifact_id": "fig-2",
        },
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
                    "text": "run analysis",
                },
                {
                    "event_id": "e2",
                    "seq": 2,
                    "created_at": "2026-04-30T00:00:01Z",
                    "type": "assistant",
                    "actor": "qa",
                    "actor_role": "assistant",
                    "user_turn_hash": "u1",
                    "text": "Analysis finished without attached artifacts.",
                },
            ],
            "conversation_events_version": 1,
            "artifact_manifest_version": 1,
            "files": {},
        },
        "meta": {"next_event_seq": 3},
    }

    payload = build_thread_export(
        thread_id="thread-2",
        provider="openai",
        model_name="gpt-test",
        state=state,
    )

    names, contents = _read_zip(payload)

    assert "conversation.json" in names
    assert "conversation.md" in names
    assert "artifacts.json" in names
    assert not any(name.startswith("artifacts/") and name != "artifacts.json" for name in names)

    manifest = json.loads(contents["artifacts.json"].decode("utf-8"))
    assert manifest == []

    conversation = json.loads(contents["conversation.json"].decode("utf-8"))
    assert conversation["thread_id"] == "thread-2"
    assert conversation["conversation"] == [
        {
            "index": 1,
            "role": "human",
            "content": "run analysis",
            "additional_kwargs": {},
        },
        {
            "index": 2,
            "role": "ai",
            "content": "Analysis finished without attached artifacts.",
            "additional_kwargs": {},
        },
    ]
    assert conversation["output"] == {
        "text": "not in chat history",
        "generated_code": "print('not approved')",
        "has_figure_png": True,
    }
