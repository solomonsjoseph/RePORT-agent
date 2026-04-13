from __future__ import annotations

from typing import Any

import requests


class OpenAIModelProbeError(RuntimeError):
    """Raised when OpenAI model discovery finds candidates but none pass a chat probe."""


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


def _extract_openai_error_message(response: Any) -> str:
    try:
        payload = response.json()
    except Exception:
        payload = None

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if message:
                return str(message)

    status_code = getattr(response, "status_code", "unknown")
    return f"HTTP {status_code}"


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
    candidate_failures: list[str] = []
    saw_candidate = False
    for model_id in model_ids:
        if not is_openai_chat_candidate(model_id):
            continue
        saw_candidate = True

        try:
            probe = http_client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={**headers, "Content-Type": "application/json"},
                json=openai_chat_probe_payload(model_id),
                timeout=15,
            )
        except Exception as exc:
            candidate_failures.append(f"{model_id}: {exc}")
            continue

        if probe.ok:
            supported.append(model_id)
            continue

        candidate_failures.append(f"{model_id}: {_extract_openai_error_message(probe)}")

    if not supported and saw_candidate and candidate_failures:
        raise OpenAIModelProbeError(
            "No supported OpenAI chat models passed the availability probe: "
            + "; ".join(candidate_failures)
        )

    return supported
