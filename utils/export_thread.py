from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from graph.state_views import get_artifact_files, get_conversation_events
from utils.display_history import build_display_history


def _json_safe(value):
    if isinstance(value, bytes):
        return {
            "type": "bytes",
            "size": len(value),
        }
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _message_role(message):
    message_type = getattr(message, "type", None)
    if message_type == "human":
        return "human"
    if message_type == "ai":
        return "ai"
    if isinstance(message, HumanMessage):
        return "human"
    if isinstance(message, AIMessage):
        return "ai"
    return type(message).__name__


def serialize_messages(messages):
    serialized = []
    for index, message in enumerate(messages, start=1):
        serialized.append(
            {
                "index": index,
                "role": _message_role(message),
                "content": message.content,
                "additional_kwargs": _json_safe(
                    dict(getattr(message, "additional_kwargs", {}) or {})
                ),
            }
        )
    return serialized


def _artifact_extension(kind: str, mime: str | None) -> str:
    if kind == "figure" or mime == "image/png":
        return ".png"
    if kind == "code":
        return ".py"
    if kind == "text":
        return ".txt"
    if kind == "sql":
        return ".sql"
    return ".json"


def _artifact_bytes(record: dict) -> bytes:
    content = record.get("content")
    kind = str(record.get("kind") or "")
    mime = record.get("mime")
    if kind == "figure" or mime == "image/png":
        path_value = dict(content or {}).get("path") if isinstance(content, dict) else None
        if isinstance(path_value, str) and path_value:
            return Path(path_value).read_bytes()
        raise ValueError("figure artifact is missing a readable path")
    if isinstance(content, str):
        return content.encode("utf-8")
    return json.dumps(content, indent=2, ensure_ascii=False).encode("utf-8")


def _artifact_manifest_and_files(state: dict) -> tuple[list[dict], list[tuple[str, bytes]]]:
    artifact_files = get_artifact_files(state)
    manifest: list[dict] = []
    archive_files: list[tuple[str, bytes]] = []
    for artifact_id, record in artifact_files.items():
        kind = str(record.get("kind") or "artifact")
        mime = record.get("mime")
        extension = _artifact_extension(kind, mime if isinstance(mime, str) or mime is None else None)
        filename = f"artifacts/{artifact_id}{extension}"
        manifest.append(
            {
                "artifact_id": artifact_id,
                "kind": kind,
                "producer": record.get("producer"),
                "mime": mime,
                "summary": record.get("summary"),
                "created_at": record.get("created_at"),
                "filename": filename,
                "content_source": "artifacts.files",
            }
        )
        archive_files.append((filename, _artifact_bytes(record)))
    return manifest, archive_files


def build_thread_export(thread_id, provider, model_name, state):
    state = dict(state or {})
    display_messages = build_display_history(state)
    runtime_messages = serialize_messages(list(state.get("messages", [])))
    display_conversation = serialize_messages(display_messages)
    conversation_events = get_conversation_events(state)
    artifacts_manifest, artifact_files = _artifact_manifest_and_files(state)
    output = dict(state.get("output") or {})
    export_payload = {
        "thread_id": thread_id,
        "provider": provider,
        "model_name": model_name,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "conversation": display_conversation,
        "conversation_events": conversation_events,
        "runtime_messages": runtime_messages,
        "artifacts": artifacts_manifest,
        "output": {
            "text": output.get("text", ""),
            "generated_code": output.get("generated_code", ""),
            "has_figure_png": bool(output.get("figure_artifact_id")),
        },
    }

    transcript_lines = []
    for entry in display_conversation:
        role = entry["role"].upper()
        transcript_lines.append(f"## {role}\n\n{entry['content']}\n")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("conversation.json", json.dumps(export_payload, indent=2, ensure_ascii=False))
        archive.writestr("conversation.md", "\n".join(transcript_lines).strip() + "\n")
        archive.writestr("artifacts.json", json.dumps(artifacts_manifest, indent=2, ensure_ascii=False))
        for filename, data in artifact_files:
            archive.writestr(filename, data)

    return zip_buffer.getvalue()
