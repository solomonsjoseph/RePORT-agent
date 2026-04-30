from __future__ import annotations

from typing import Any


def build_user_message_attachments(uploaded_artifact: dict[str, Any] | None) -> list[dict[str, str]]:
    if not uploaded_artifact:
        return []

    attachments = [
        {
            "artifact_id": str(uploaded_artifact["id"]),
            "kind": "dataset",
            "role": "primary",
        }
    ]
    schema_path = str(uploaded_artifact.get("schema_path") or "").strip()
    if schema_path:
        attachments.append(
            {
                "artifact_id": f'{uploaded_artifact["id"]}.schema',
                "kind": "schema",
                "role": "supporting",
            }
        )
    return attachments


def latest_message_attachments(state: dict[str, Any]) -> list[dict[str, str]]:
    for message in reversed(list(state.get("messages", []))):
        if getattr(message, "type", None) != "human":
            continue
        additional_kwargs = dict(getattr(message, "additional_kwargs", {}) or {})
        attachments = list(additional_kwargs.get("attachments") or [])
        return [
            {
                "artifact_id": str(item.get("artifact_id") or ""),
                "kind": str(item.get("kind") or ""),
                "role": str(item.get("role") or ""),
            }
            for item in attachments
            if isinstance(item, dict) and str(item.get("artifact_id") or "").strip()
        ]
    return []


def latest_attachment_summary(state: dict[str, Any]) -> dict[str, object]:
    attachments = latest_message_attachments(state)
    return {
        "has_attachments": bool(attachments),
        "attachment_kinds": [attachment["kind"] for attachment in attachments],
        "attachments": attachments,
    }
