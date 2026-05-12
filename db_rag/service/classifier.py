from __future__ import annotations

import json
import math
import os
from typing import Any

from utils.llm_response import coerce_text_content

from db_rag.config import resolve_db_rag_reply_classifier_model
from db_rag.generation import parse_json_object

_NEW_HELPER_CONFIDENCE_THRESHOLD = 0.5


def _unknown_result() -> dict[str, Any]:
    return {"label": "unknown", "confidence": 0.0}


def _normalize_classifier_result(
    parsed: dict[str, Any],
    *,
    allowed_labels: set[str],
) -> dict[str, Any]:
    label = str(parsed.get("label") or "").strip().lower()
    if label not in allowed_labels:
        return _unknown_result()

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if not math.isfinite(confidence):
        return _unknown_result()
    confidence = max(0.0, min(1.0, confidence))
    return {"label": label, "confidence": confidence}


def _resolve_openai_client() -> Any | None:
    try:
        from openai import OpenAI as client_cls
    except ModuleNotFoundError:
        return None
    return client_cls


def _classify_new_helper(
    *,
    system_text: str,
    user_text: str,
    allowed_labels: set[str],
    resolve_model=resolve_db_rag_reply_classifier_model,
) -> dict[str, Any]:
    model = resolve_model()
    if not model:
        return _unknown_result()

    api_key = str(os.getenv("DB_RAG_REPLY_CLASSIFIER_API_KEY", "") or "").strip()
    if not api_key:
        api_key = str(os.getenv("OPENAI_API_KEY", "") or "").strip()
    if not api_key:
        return _unknown_result()

    openai_client = _resolve_openai_client()
    if openai_client is None:
        return _unknown_result()

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = str(os.getenv("DB_RAG_REPLY_CLASSIFIER_BASE_URL", "") or "").strip()
    if base_url:
        client_kwargs["base_url"] = base_url

    try:
        client = openai_client(**client_kwargs)
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
        )
        content = coerce_text_content(getattr(response.choices[0].message, "content", ""))
        parsed = parse_json_object(content) or {}
    except Exception:
        return _unknown_result()

    result = _normalize_classifier_result(parsed, allowed_labels=allowed_labels)
    if result["confidence"] < _NEW_HELPER_CONFIDENCE_THRESHOLD:
        return _unknown_result()
    return result


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


def classify_database_source_intent(
    *,
    pending_question: str,
    user_reply: str,
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

    openai_client = _resolve_openai_client()
    if openai_client is None:
        return {"label": "unknown", "confidence": 0.0}

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = str(os.getenv("DB_RAG_REPLY_CLASSIFIER_BASE_URL", "") or "").strip()
    if base_url:
        client_kwargs["base_url"] = base_url

    try:
        client = openai_client(**client_kwargs)
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You classify whether a RePORT user request should use the DB-RAG database. "
                        "Return JSON only with keys label and confidence. "
                        "Allowed labels: database, unknown. "
                        "Use database when the user means the RePORT, study, database, or DB source. "
                        "Use unknown when ambiguous."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original task:\n{pending_question}\n\n"
                        f"Latest user reply:\n{user_reply}"
                    ),
                },
            ],
        )
        content = coerce_text_content(getattr(response.choices[0].message, "content", ""))
        parsed = parse_json_object(content) or {}
    except Exception:
        return {"label": "unknown", "confidence": 0.0}

    result = _normalize_classifier_result(
        parsed,
        allowed_labels={"database", "unknown"},
    )
    if result["confidence"] < _NEW_HELPER_CONFIDENCE_THRESHOLD:
        return {"label": "unknown", "confidence": 0.0}
    return result


def _unknown_clarification_result() -> dict[str, Any]:
    return {
        "decision": "unclear",
        "intent": "unknown",
        "normalized_value": None,
        "target_node": None,
        "confidence": 0.0,
        "reason": "",
    }


def classify_clarification_reply(
    *,
    clarification_kind: str,
    clarification_return_node: str,
    pending_question: str,
    reply: str,
    expected: dict[str, Any],
    allowed_reroute_intents: list[str],
    allowed_target_nodes: list[str],
    resolve_model=resolve_db_rag_reply_classifier_model,
) -> dict[str, Any]:
    model = resolve_model()
    if not model:
        return _unknown_clarification_result()

    api_key = str(os.getenv("DB_RAG_REPLY_CLASSIFIER_API_KEY", "") or "").strip()
    if not api_key:
        api_key = str(os.getenv("OPENAI_API_KEY", "") or "").strip()
    if not api_key:
        return _unknown_clarification_result()

    openai_client = _resolve_openai_client()
    if openai_client is None:
        return _unknown_clarification_result()

    allowed_intents = [
        str(item).strip()
        for item in allowed_reroute_intents
        if str(item).strip()
    ]
    allowed_targets = [
        str(item).strip()
        for item in allowed_target_nodes
        if str(item).strip()
    ]
    try:
        expected_text = json.dumps(expected or {}, sort_keys=True)
    except TypeError:
        return _unknown_clarification_result()
    intent_descriptions = {
        "database_source": (
            "The user wants to use the RePORT/study database or DB-RAG source "
            "instead of answering the current uploaded-dataset or code-generation clarification."
        ),
        "uploaded_dataset_analysis": (
            "The user wants to analyze an uploaded/local dataset instead of the "
            "current database or QA workflow."
        ),
        "general_question": (
            "The user is asking a general explanatory question rather than continuing "
            "the current database or code workflow."
        ),
    }
    active_intent_descriptions = {
        intent: intent_descriptions[intent]
        for intent in allowed_intents
        if intent in intent_descriptions
    }

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = str(os.getenv("DB_RAG_REPLY_CLASSIFIER_BASE_URL", "") or "").strip()
    if base_url:
        client_kwargs["base_url"] = base_url

    try:
        client = openai_client(**client_kwargs)
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify a user's reply to an active RePORT clarification. "
                        "Return JSON only with keys decision, intent, normalized_value, "
                        "target_node, confidence, reason. "
                        "Allowed decisions: valid, reroute, unclear. "
                        f"Allowed reroute intents: {json.dumps(allowed_intents)}. "
                        f"Reroute intent meanings: {json.dumps(active_intent_descriptions)}. "
                        f"Allowed target nodes: {json.dumps(allowed_targets)}. "
                        "Do not invent dataset IDs, node names, or workflow options. "
                        "Use unclear when the reply does not satisfy the expected format "
                        "or an allowed reroute."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Clarification kind: {clarification_kind}\n"
                        f"Return node: {clarification_return_node}\n"
                        f"Expected answer contract: {expected_text}\n"
                        f"Original pending question: {pending_question}\n"
                        f"Latest user reply: {reply}"
                    ),
                },
            ],
        )
        content = coerce_text_content(getattr(response.choices[0].message, "content", ""))
        parsed = parse_json_object(content) or {}
    except Exception:
        return _unknown_clarification_result()

    decision = str(parsed.get("decision") or "").strip().lower()
    if decision not in {"valid", "reroute", "unclear"}:
        return _unknown_clarification_result()

    target_node = str(parsed.get("target_node") or "").strip()
    if target_node and target_node not in allowed_targets:
        return _unknown_clarification_result()

    intent = str(parsed.get("intent") or "unknown").strip() or "unknown"
    if decision == "reroute" and intent not in allowed_intents:
        return _unknown_clarification_result()

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if not math.isfinite(confidence):
        return _unknown_clarification_result()
    confidence = max(0.0, min(1.0, confidence))
    if confidence < _NEW_HELPER_CONFIDENCE_THRESHOLD:
        return _unknown_clarification_result()

    return {
        "decision": decision,
        "intent": intent,
        "normalized_value": parsed.get("normalized_value"),
        "target_node": target_node or None,
        "confidence": confidence,
        "reason": str(parsed.get("reason") or ""),
    }


def classify_extraction_gate_message(
    *,
    pending_prompt: str,
    user_message: str,
    recent_transcript: str,
    active_intent: dict[str, Any] | None,
    resolve_model=resolve_db_rag_reply_classifier_model,
) -> dict[str, Any]:
    try:
        active_intent_text = json.dumps(active_intent or {}, indent=2, sort_keys=True)
    except Exception:
        return _unknown_result()

    return _classify_new_helper(
        system_text=(
            "You classify a free-text reply under a pending DB-RAG extraction clarification. "
            "Return JSON only with keys label and confidence. "
            "Allowed labels: new_question, reply_to_pending_gate, unknown."
        ),
        user_text=(
            f"Pending prompt:\n{pending_prompt}\n\n"
            f"Latest user message:\n{user_message}\n\n"
            f"Recent transcript:\n{recent_transcript}\n\n"
            f"Active intent:\n{active_intent_text}\n\n"
            "Classify whether the latest message is a fresh self-contained DB-RAG question, "
            "a reply to the pending extraction prompt, or unclear."
        ),
        allowed_labels={"new_question", "reply_to_pending_gate", "unknown"},
        resolve_model=resolve_model,
    )


def classify_sql_review_feedback(
    *,
    feedback_text: str,
    sql: str,
    resolve_model=resolve_db_rag_reply_classifier_model,
) -> dict[str, Any]:
    return _classify_new_helper(
        system_text=(
            "You classify DB-RAG SQL review rejection feedback. "
            "Return JSON only with keys label and confidence. "
            "Allowed labels: sql_only_revision, selection_revision, unknown."
        ),
        user_text=(
            f"SQL under review:\n{sql}\n\n"
            f"Reviewer feedback:\n{feedback_text}\n\n"
            "Classify whether the feedback asks to regenerate SQL from the same approved selection, "
            "or whether it actually requires column/table selection revision."
        ),
        allowed_labels={"sql_only_revision", "selection_revision", "unknown"},
        resolve_model=resolve_model,
    )


def classify_population_scope(
    *,
    question: str,
    table_context: str,
    column_context: str,
    active_intent: dict[str, Any] | None,
    intent_snapshot: dict[str, Any] | None,
    referenced_artifacts: dict[str, Any] | None,
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

    openai_client = _resolve_openai_client()
    if openai_client is None:
        return {"label": "unknown", "confidence": 0.0}

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = str(os.getenv("DB_RAG_REPLY_CLASSIFIER_BASE_URL", "") or "").strip()
    if base_url:
        client_kwargs["base_url"] = base_url

    try:
        client = openai_client(**client_kwargs)
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You classify whether a RePORT database question has explicit population scope. "
                        "The database has index cases and household contacts. Return JSON only with keys "
                        "label, population, confidence, and reason. Allowed labels: specified, ambiguous, "
                        "not_population_scoped, unknown. Allowed population values: index_case, "
                        "household_contact, both, null. Mark ambiguous when a participant-level question could "
                        "refer to either index cases or household contacts and the request/prior intent does not "
                        "resolve the population. Mark not_population_scoped for schema-only questions."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question:\n{question}\n\n"
                        f"Active intent:\n{json.dumps(active_intent or {}, indent=2, sort_keys=True)}\n\n"
                        f"Intent snapshot:\n{json.dumps(intent_snapshot or {}, indent=2, sort_keys=True)}\n\n"
                        f"Referenced artifacts:\n{json.dumps(referenced_artifacts or {}, indent=2, sort_keys=True)}\n\n"
                        f"Table context:\n{table_context or 'none'}\n\n"
                        f"Column context:\n{column_context or 'none'}"
                    ),
                },
            ],
        )
        content = coerce_text_content(getattr(response.choices[0].message, "content", ""))
        parsed = parse_json_object(content) or {}
    except Exception:
        return {"label": "unknown", "confidence": 0.0}

    result = _normalize_classifier_result(
        parsed,
        allowed_labels={"specified", "ambiguous", "not_population_scoped", "unknown"},
    )
    if result["confidence"] < _NEW_HELPER_CONFIDENCE_THRESHOLD:
        return {"label": "unknown", "confidence": 0.0}
    population = str(parsed.get("population") or "").strip().lower()
    if population not in {"index_case", "household_contact", "both"}:
        population = None
    result["population"] = population
    result["reason"] = str(parsed.get("reason") or "").strip()
    return result


def build_recoverable_error_clarification(
    *,
    workflow: str,
    error_payload: dict[str, Any],
    context: dict[str, Any],
    resolve_model=resolve_db_rag_reply_classifier_model,
) -> dict[str, str]:
    model = resolve_model()
    if not model:
        return {}

    api_key = str(os.getenv("DB_RAG_REPLY_CLASSIFIER_API_KEY", "") or "").strip()
    if not api_key:
        api_key = str(os.getenv("OPENAI_API_KEY", "") or "").strip()
    if not api_key:
        return {}

    openai_client = _resolve_openai_client()
    if openai_client is None:
        return {}

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = str(os.getenv("DB_RAG_REPLY_CLASSIFIER_BASE_URL", "") or "").strip()
    if base_url:
        client_kwargs["base_url"] = base_url

    try:
        client = openai_client(**client_kwargs)
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You convert a recoverable DB-RAG workflow error into one concise user clarification. "
                        "Return JSON only with key clarification_question. "
                        "Ask for the missing decision or constraint needed to continue; do not apologize, "
                        "do not expose stack traces, and do not tell the user to restart."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Workflow:\n{workflow}\n\n"
                        f"Error payload:\n{json.dumps(error_payload, indent=2, sort_keys=True)}\n\n"
                        f"DB-RAG context:\n{json.dumps(context, indent=2, sort_keys=True)}"
                    ),
                },
            ],
        )
        content = coerce_text_content(getattr(response.choices[0].message, "content", ""))
        parsed = parse_json_object(content) or {}
    except Exception:
        return {}

    question = str(parsed.get("clarification_question") or "").strip()
    if not question:
        return {}
    return {"clarification_question": question}
