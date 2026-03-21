import io
import json
import zipfile
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage


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
                "additional_kwargs": dict(getattr(message, "additional_kwargs", {}) or {}),
            }
        )
    return serialized


def build_thread_export(thread_id, provider, model_name, messages, output):
    serialized_messages = serialize_messages(messages)
    export_payload = {
        "thread_id": thread_id,
        "provider": provider,
        "model_name": model_name,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "conversation": serialized_messages,
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

        generated_code = output.get("generated_code")
        if generated_code:
            archive.writestr("generated_code.py", generated_code)

        output_text = output.get("text")
        if output_text:
            archive.writestr("output.txt", output_text)

        figure_png = output.get("figure_png")
        if figure_png:
            archive.writestr("figure.png", figure_png)

    return zip_buffer.getvalue()
