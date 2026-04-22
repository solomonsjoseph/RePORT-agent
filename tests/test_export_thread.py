from __future__ import annotations

import io
import json
import sys
import zipfile
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


def test_build_thread_export_collects_assistant_message_artifacts() -> None:
    messages = [
        HumanMessage(content="make a plot"),
        AIMessage(
            content=(
                "Summary.\n\n"
                "Generated code:\n```python\nprint('hello')\n```\n\n"
                "Output:\n```\nhello\n```"
            ),
            additional_kwargs={"figure_png": b"png-bytes"},
        ),
    ]

    payload = build_thread_export(
        thread_id="thread-1",
        provider="openai",
        model_name="gpt-test",
        messages=messages,
        output={
            "generated_code": "print('stale output field')",
            "text": "stale text",
            "figure_png": b"stale-figure",
        },
    )

    names, contents = _read_zip(payload)

    assert "conversation.json" in names
    assert "conversation.md" in names
    assert "artifacts.json" in names
    assert "artifacts/message_002_generated_code.py" in names
    assert "artifacts/message_002_output.txt" in names
    assert "artifacts/message_002_figure.png" in names
    assert contents["artifacts/message_002_generated_code.py"] == b"print('hello')\n"
    assert contents["artifacts/message_002_output.txt"] == b"hello\n"
    assert contents["artifacts/message_002_figure.png"] == b"png-bytes"

    manifest = json.loads(contents["artifacts.json"].decode("utf-8"))
    assert manifest == [
        {
            "message_index": 2,
            "artifact_type": "generated_code",
            "filename": "artifacts/message_002_generated_code.py",
            "role": "ai",
            "content_source": "assistant_message.content",
            "label": "Generated code",
        },
        {
            "message_index": 2,
            "artifact_type": "output_text",
            "filename": "artifacts/message_002_output.txt",
            "role": "ai",
            "content_source": "assistant_message.content",
            "label": "Output",
        },
        {
            "message_index": 2,
            "artifact_type": "figure",
            "filename": "artifacts/message_002_figure.png",
            "role": "ai",
            "content_source": "assistant_message.additional_kwargs.figure_png",
            "label": "Figure",
        },
    ]


def test_build_thread_export_keeps_conversation_exports_and_skips_non_message_output() -> None:
    messages = [
        HumanMessage(content="run analysis"),
        AIMessage(content="Analysis finished without attached artifacts."),
    ]

    payload = build_thread_export(
        thread_id="thread-2",
        provider="openai",
        model_name="gpt-test",
        messages=messages,
        output={
            "generated_code": "print('not approved')",
            "text": "not in chat history",
            "figure_png": b"not-in-history",
        },
    )

    names, contents = _read_zip(payload)

    assert "conversation.json" in names
    assert "conversation.md" in names
    assert "artifacts.json" in names
    assert "generated_code.py" not in names
    assert "output.txt" not in names
    assert "figure.png" not in names
    assert not any(name.startswith("artifacts/message_") for name in names)

    manifest = json.loads(contents["artifacts.json"].decode("utf-8"))
    assert manifest == []

    conversation = json.loads(contents["conversation.json"].decode("utf-8"))
    assert conversation["thread_id"] == "thread-2"
    assert conversation["output"] == {
        "text": "not in chat history",
        "generated_code": "print('not approved')",
        "has_figure_png": True,
    }
