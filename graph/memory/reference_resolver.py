from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any

from graph.state import AgentState

from .task_store import (
    cache_reference_resolution,
    ensure_memory_state,
    get_cached_reference_resolution,
    latest_task_cards,
)
from .validation import validate_reference_resolution


TASK_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9_])(task_[0-9a-fA-F]+)(?![A-Za-z0-9_])")
DISPLAY_ORDINAL_PATTERN = re.compile(r"(?<![A-Za-z0-9_])Task\s+([1-9][0-9]*)(?![A-Za-z0-9_])")

RESOLVER_INSTRUCTIONS = (
    "Classify whether the latest user message references one completed task.\n"
    "Use only the task cards provided.\n"
    "Return JSON with label, task_id, relationship, intended_action, confidence, needs_reference, reason.\n"
    "Allowed labels: resolved, new_task, ambiguous, unknown.\n"
    "Allowed relationships: revision, rerun, explain, inspect_artifact, use_as_input, compare.\n"
    "Do not invent relationship names such as follow_up_to_completed_task.\n"
    "If the user says a completed result is wrong, incomplete, needs different fields, "
    "or should use the right dataset, use revision."
)


def _direct_task_ids_from_text(memory: dict[str, Any], user_message: str) -> list[str]:
    completed = memory["completed_tasks"]
    direct_ids: list[str] = []

    for match in TASK_ID_PATTERN.finditer(user_message):
        task_id = match.group(1)
        if task_id in completed and task_id not in direct_ids:
            direct_ids.append(task_id)

    ordinal_to_task_id = {
        card.get("display_ordinal"): task_id
        for task_id, card in completed.items()
        if isinstance(card, dict)
    }
    for match in DISPLAY_ORDINAL_PATTERN.finditer(user_message):
        task_id = ordinal_to_task_id.get(int(match.group(1)))
        if isinstance(task_id, str) and task_id not in direct_ids:
            direct_ids.append(task_id)

    for task_id, card in completed.items():
        artifact_refs = card.get("artifact_refs") if isinstance(card, dict) else {}
        if not isinstance(artifact_refs, dict):
            continue
        for artifact_id in artifact_refs.values():
            if (
                isinstance(artifact_id, str)
                and artifact_id
                and _contains_exact_token(user_message, artifact_id)
                and task_id not in direct_ids
            ):
                direct_ids.append(task_id)

    return direct_ids


def _contains_exact_token(text: str, token: str) -> bool:
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])")
    return bool(pattern.search(text))


def build_resolver_payload(
    state: AgentState,
    user_message: str,
    user_message_hash: str,
    limit: int = 12,
) -> dict[str, Any]:
    _state, memory = ensure_memory_state(state)
    direct_task_ids = _direct_task_ids_from_text(memory, user_message)
    return {
        "user_message": user_message,
        "user_message_hash": user_message_hash,
        "latest_task_id": memory.get("last_task_id"),
        "completed_tasks": latest_task_cards(state, limit=limit, extra_task_ids=direct_task_ids),
    }


def _relationship_for_direct_match(card: dict[str, Any], user_message: str) -> str:
    artifact_refs = card.get("artifact_refs") if isinstance(card.get("artifact_refs"), dict) else {}
    dataset_id = artifact_refs.get("dataset_artifact_id")
    if isinstance(dataset_id, str) and _contains_exact_token(user_message, dataset_id):
        return "use_as_input"
    if card.get("kind") != "db_rag_sql_extraction":
        return "explain"
    return "inspect_artifact"


def _direct_resolution(memory: dict[str, Any], user_message: str) -> dict[str, Any] | None:
    direct_task_ids = _direct_task_ids_from_text(memory, user_message)
    if not direct_task_ids:
        return None
    if len(direct_task_ids) > 1:
        return {
            "label": "ambiguous",
            "task_id": None,
            "relationship": None,
            "intended_action": None,
            "confidence": "high",
            "needs_reference": True,
            "reason": "The user message contains exact references to multiple completed tasks.",
        }
    task_id = direct_task_ids[0]
    card = memory["completed_tasks"][task_id]
    relationship = _relationship_for_direct_match(card, user_message)
    return {
        "label": "resolved",
        "task_id": task_id,
        "relationship": relationship,
        "intended_action": None,
        "confidence": "high",
        "needs_reference": False,
        "reason": "The user message contains an exact task or artifact identifier.",
    }


def _parse_llm_response(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    content = getattr(response, "content", response)
    if isinstance(content, dict):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                text_parts.append(item["text"])
            elif isinstance(item, str):
                text_parts.append(item)
            else:
                text_parts.append(str(item))
        text = "".join(text_parts)
    else:
        text = str(content)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM reference resolution response must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("LLM reference resolution response must be a JSON object")
    return parsed


def _invoke_llm(llm: Any, payload: dict[str, Any]) -> dict[str, Any]:
    prompt = f"{RESOLVER_INSTRUCTIONS}\n\nTask cards payload:\n{json.dumps(payload, sort_keys=True)}"
    if not hasattr(llm, "invoke"):
        raise ValueError("Reference resolver llm must expose invoke")
    return _parse_llm_response(llm.invoke(prompt))


def _cache_and_return(
    state: AgentState,
    user_message_hash: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    cache_reference_resolution(state, user_message_hash, result)
    return deepcopy(result)


def resolve_reference_for_turn(
    state: AgentState,
    llm: Any,
    user_message: str,
    user_message_hash: str,
) -> dict[str, Any]:
    cached = get_cached_reference_resolution(state, user_message_hash)
    if cached is not None:
        return cached

    _state, memory = ensure_memory_state(state)
    if not memory["completed_tasks"]:
        result = validate_reference_resolution(
            state,
            {"label": "new_task", "needs_reference": False, "reason": "No completed tasks exist."},
        )
        return _cache_and_return(state, user_message_hash, result)

    payload = build_resolver_payload(state, user_message, user_message_hash)
    direct_result = _direct_resolution(memory, user_message)
    if direct_result is not None:
        result = validate_reference_resolution(
            state,
            direct_result,
            candidate_cards=payload["completed_tasks"],
        )
        return _cache_and_return(state, user_message_hash, result)

    raw_result = _invoke_llm(llm, payload)
    result = validate_reference_resolution(
        state,
        raw_result,
        candidate_cards=payload["completed_tasks"],
    )
    return _cache_and_return(state, user_message_hash, result)
