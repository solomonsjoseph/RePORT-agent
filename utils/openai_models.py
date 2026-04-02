from __future__ import annotations

from typing import Any

import requests


def is_openai_gpt5_family(model_name: str) -> bool:
    normalized = (model_name or "").strip().lower()
    return normalized.startswith("gpt-5")


def is_openai_chat_candidate(model_name: str) -> bool:
    normalized = (model_name or "").strip().lower()
    if not (normalized.startswith("gpt-4") or normalized.startswith("gpt-5")):
        return False

    unsupported_markers = (
        "audio",
        "realtime",
        "search",
        "transcribe",
        "tts",
        "moderation",
        "image",
        "codex",
        "pro",
    )
    return not any(marker in normalized for marker in unsupported_markers)


def openai_chat_probe_payload(model_name: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
    }
    if is_openai_gpt5_family(model_name):
        payload["max_completion_tokens"] = 8
    else:
        payload["max_tokens"] = 8
    return payload


def list_supported_openai_chat_models(
    api_key: str,
    http_client=requests,
) -> list[str]:
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = http_client.get("https://api.openai.com/v1/models", headers=headers, timeout=10)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    model_ids = sorted({item.get("id") for item in data if item.get("id")})

    supported: list[str] = []
    for model_id in model_ids:
        if not is_openai_chat_candidate(model_id):
            continue

        probe = http_client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={**headers, "Content-Type": "application/json"},
            json=openai_chat_probe_payload(model_id),
            timeout=15,
        )
        if probe.ok:
            supported.append(model_id)

    return supported
