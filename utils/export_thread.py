import io
import json
import re
import zipfile
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage


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


_FENCED_BLOCK_RE = re.compile(r"```(?P<lang>[^\n`]*)\n(?P<body>.*?)\n```", re.DOTALL)
_CODE_LABEL_HINTS = ("generated code", "code", "python")
_OUTPUT_LABEL_HINTS = ("output", "result", "stdout")


def _artifact_entry(
    message_index: int,
    artifact_type: str,
    filename: str,
    content_source: str,
    label: str,
):
    return {
        "message_index": message_index,
        "artifact_type": artifact_type,
        "filename": filename,
        "role": "ai",
        "content_source": content_source,
        "label": label,
    }


def _nearest_label_before(content: str, start: int) -> str:
    context = content[:start]
    lines = [line.strip().lower() for line in context.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _extract_text_artifacts(message_index: int, content: str):
    artifacts = []
    text = str(content or "")

    for match in _FENCED_BLOCK_RE.finditer(text):
        body = match.group("body")
        language = str(match.group("lang") or "").strip().lower()
        label = _nearest_label_before(text, match.start())

        if (
            language == "python"
            or any(hint in label for hint in _CODE_LABEL_HINTS)
        ):
            artifacts.append(
                (
                    _artifact_entry(
                        message_index=message_index,
                        artifact_type="generated_code",
                        filename=f"artifacts/message_{message_index:03d}_generated_code.py",
                        content_source="assistant_message.content",
                        label="Generated code",
                    ),
                    body.rstrip() + "\n",
                )
            )
            continue

        if any(hint in label for hint in _OUTPUT_LABEL_HINTS):
            artifacts.append(
                (
                    _artifact_entry(
                        message_index=message_index,
                        artifact_type="output_text",
                        filename=f"artifacts/message_{message_index:03d}_output.txt",
                        content_source="assistant_message.content",
                        label="Output",
                    ),
                    body.rstrip() + "\n",
                )
            )

    return artifacts


def _extract_message_artifacts(messages):
    manifest = []
    files = []

    for message_index, message in enumerate(messages, start=1):
        if _message_role(message) != "ai":
            continue

        content = getattr(message, "content", "") or ""
        additional_kwargs = dict(getattr(message, "additional_kwargs", {}) or {})

        for artifact_entry, artifact_text in _extract_text_artifacts(message_index, content):
            manifest.append(artifact_entry)
            files.append((artifact_entry["filename"], artifact_text.encode("utf-8")))

        figure_png = additional_kwargs.get("figure_png")
        if isinstance(figure_png, bytes) and figure_png:
            artifact_entry = _artifact_entry(
                message_index=message_index,
                artifact_type="figure",
                filename=f"artifacts/message_{message_index:03d}_figure.png",
                content_source="assistant_message.additional_kwargs.figure_png",
                label="Figure",
            )
            manifest.append(artifact_entry)
            files.append((artifact_entry["filename"], figure_png))

    return manifest, files


def build_thread_export(thread_id, provider, model_name, messages, output):
    serialized_messages = serialize_messages(messages)
    artifacts_manifest, artifact_files = _extract_message_artifacts(messages)
    export_payload = {
        "thread_id": thread_id,
        "provider": provider,
        "model_name": model_name,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "conversation": serialized_messages,
        "artifacts": artifacts_manifest,
        "output": {
            "text": output.get("text", ""),
            "generated_code": output.get("generated_code", ""),
            "has_figure_png": bool(output.get("figure_png")),
        },
    }

    transcript_lines = []
    for entry in serialized_messages:
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
