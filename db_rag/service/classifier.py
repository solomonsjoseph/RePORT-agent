from __future__ import annotations

import os
from typing import Any

from utils.llm_response import coerce_text_content

from db_rag.config import resolve_db_rag_reply_classifier_model
from db_rag.generation import parse_json_object


def classify_pending_reply(
    *,
    pending_question: str,
    user_reply: str,
    recent_transcript: str,
    resolve_model=resolve_db_rag_reply_classifier_model,
) -> dict[str, Any]:
    model = resolve_model()
    if not model:
        return {"label": "unknown", "confidence": 0.0}

    api_key = str(os.getenv("DB_RAG_REPLY_CLASSIFIER_API_KEY", "") or "").strip()
    if not api_key:
        api_key = str(os.getenv("OPENAI_API_KEY", "") or "").strip()
    if not api_key:
        return {"label": "unknown", "confidence": 0.0}

    try:
        from openai import OpenAI
    except ModuleNotFoundError:
        return {"label": "unknown", "confidence": 0.0}

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = str(os.getenv("DB_RAG_REPLY_CLASSIFIER_BASE_URL", "") or "").strip()
    if base_url:
        client_kwargs["base_url"] = base_url

    try:
        client = OpenAI(**client_kwargs)
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You classify the latest user reply for a pending RePORT DB-RAG conversational gate. "
                        "Return JSON only with keys label and confidence. "
                        "Allowed labels: yes, no, substantive_followup, unknown."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Pending question:\n{pending_question}\n\n"
                        f"Recent transcript:\n{recent_transcript}\n\n"
                        f"Latest user reply:\n{user_reply}\n\n"
                        "Classify whether the user is agreeing to proceed, declining, asking a new substantive "
                        "DB-RAG follow-up instead of answering yes/no, or remaining unclear."
                    ),
                },
            ],
        )
        content = coerce_text_content(getattr(response.choices[0].message, "content", ""))
        parsed = parse_json_object(content) or {}
    except Exception:
        return {"label": "unknown", "confidence": 0.0}

    label = str(parsed.get("label") or "").strip().lower()
    if label not in {"yes", "no", "substantive_followup", "unknown"}:
        return {"label": "unknown", "confidence": 0.0}

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return {"label": label, "confidence": confidence}
