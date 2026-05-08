from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from graph.state import AgentState

from .schema import ALLOWED_USER_INTENT_KINDS, ALLOWED_USER_INTENT_STATUSES, require_json_safe
from .task_store import ensure_memory_state


LIVE_USER_INTENT_STATUSES = {
    "active",
    "awaiting_extraction_opt_in",
    "awaiting_column_review",
    "awaiting_sql_review",
}

ALLOWED_USER_INTENT_REFERENCE_TARGETS = {
    "new_user_intent",
    "existing_user_intent",
    "completed_task",
    "ambiguous",
}

ALLOWED_USER_INTENT_RELATIONSHIPS = {
    "continue",
    "refine",
    "inspect_result",
    "new_request",
}

CLASSIFIER_INSTRUCTIONS = (
    "Classify whether the latest user message refers to a prior DB-RAG user intent, "
    "a completed task/result, a fresh request, or an ambiguous reference. "
    "Choose only IDs present in candidate_user_intents or candidate_completed_tasks. "
    "Return JSON with target, target_id, relationship, confidence, reason."
)

COMPACT_COMPLETED_TASK_CLASSIFIER_FIELDS = (
    "task_id",
    "kind",
    "source_question",
    "goal_text",
    "label",
    "summary",
    "dataset_id",
    "created_at",
    "updated_at",
)

COMPACT_USER_INTENT_FIELDS = (
    "intent_id",
    "display_ordinal",
    "kind",
    "agent",
    "source_question",
    "goal_text",
    "status",
    "source_message_hash",
    "active_intent_id",
    "completed_task_id",
    "continued_from_intent_id",
    "created_at",
    "updated_at",
)

INSPECTABLE_COMPLETED_TASK_ARTIFACT_KEYS = {
    "dataset_artifact_id",
    "dataset_id",
    "sql_candidate_artifact_id",
    "selection_artifact_id",
}


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _new_intent_id(memory: dict[str, Any]) -> str:
    intent_id = f"intent_{uuid4().hex[:8]}"
    while intent_id in memory["user_intents"]:
        intent_id = f"intent_{uuid4().hex[:8]}"
    return intent_id


def _validate_user_intent_kind(kind: str) -> None:
    if kind not in ALLOWED_USER_INTENT_KINDS:
        raise ValueError(f"Unknown user intent kind: {kind}")


def _validate_user_intent_status(status: str) -> None:
    if status not in ALLOWED_USER_INTENT_STATUSES:
        raise ValueError(f"Unknown user intent status: {status}")


def _validate_text(value: Any, field_name: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")
    require_json_safe(value, field_name)


def _validate_optional_text(value: Any, field_name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{field_name} must be str | None")
    require_json_safe(value, field_name)


def _validate_db_rag_active_intent(active_intent: Any) -> dict[str, Any]:
    if not isinstance(active_intent, dict):
        raise ValueError("active_intent must be a dict")
    require_json_safe(active_intent, "active_intent")
    active_intent_id = active_intent.get("intent_id")
    source_question = active_intent.get("source_question")
    goal_text = active_intent.get("goal_text")
    _validate_optional_text(active_intent_id, "active_intent.intent_id")
    _validate_text(source_question, "active_intent.source_question")
    _validate_text(goal_text, "active_intent.goal_text")
    return active_intent


def _matching_intent_id(memory: dict[str, Any], active_intent_id: str | None) -> str | None:
    if active_intent_id is None:
        return None
    for intent_id in memory["intent_order"]:
        card = memory["user_intents"].get(intent_id)
        if isinstance(card, dict) and card.get("active_intent_id") == active_intent_id:
            return intent_id
    return None


def _latest_live_intent_id(memory: dict[str, Any], *, kind: str) -> str | None:
    _validate_user_intent_kind(kind)
    intent_id = memory["last_user_intent_id_by_kind"].get(kind)
    if not isinstance(intent_id, str):
        return None
    card = memory["user_intents"].get(intent_id)
    if (
        isinstance(card, dict)
        and card.get("kind") == kind
        and card.get("agent") == "rag_db_qa"
        and card.get("status") in LIVE_USER_INTENT_STATUSES
    ):
        return intent_id
    return None


def _compact_card(card: dict[str, Any]) -> dict[str, Any]:
    return {field: deepcopy(card.get(field)) for field in COMPACT_USER_INTENT_FIELDS}


def _compact_completed_task_classifier_card(card: dict[str, Any]) -> dict[str, Any]:
    return {
        field: deepcopy(card.get(field))
        for field in COMPACT_COMPLETED_TASK_CLASSIFIER_FIELDS
    }


def compact_user_intent_cards(
    state: AgentState,
    *,
    limit: int = 12,
    kind: str | None = None,
) -> list[dict[str, Any]]:
    _state, memory = ensure_memory_state(state)
    if kind is not None:
        _validate_user_intent_kind(kind)
    capped_limit = max(limit, 0)
    if capped_limit == 0:
        return []
    selected: list[dict[str, Any]] = []
    for intent_id in reversed(memory["intent_order"]):
        card = memory["user_intents"].get(intent_id)
        if not isinstance(card, dict):
            continue
        if kind is not None and card.get("kind") != kind:
            continue
        selected.append(_compact_card(card))
        if len(selected) >= capped_limit:
            break
    return selected


def build_user_intent_classifier_payload(
    state: AgentState,
    *,
    user_message: str,
    user_message_hash: str,
    recent_turns: list[dict[str, str]] | None = None,
    completed_task_cards: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    agents = dict(state.get("agents") or {})
    rag_state = dict(agents.get("rag_db_qa") or {})
    candidate_completed_tasks = [
        _compact_completed_task_classifier_card(card)
        for card in list(completed_task_cards or [])[:8]
        if isinstance(card, dict)
    ]
    return {
        "latest_user_message": str(user_message or ""),
        "user_message_hash": user_message_hash,
        "recent_turns": list(recent_turns or [])[-4:],
        "active_workflow": {
            "rag_db_thread_status": rag_state.get("thread_status"),
            "rag_db_active_thread": bool(rag_state.get("active_thread")),
            "pending_review": bool(agents.get("human_review")),
            "pending_clarification": bool(
                (state.get("meta") or {}).get("awaiting_user_clarification")
            ),
        },
        "candidate_user_intents": compact_user_intent_cards(
            state,
            kind="db_rag_query",
            limit=8,
        ),
        "candidate_completed_tasks": candidate_completed_tasks,
    }


def _content_blocks_to_text(blocks: list[Any]) -> str:
    text_parts: list[str] = []
    for block in blocks:
        if isinstance(block, str):
            text_parts.append(block)
            continue
        if isinstance(block, dict):
            text = block.get("text")
            if isinstance(text, str):
                text_parts.append(text)
                continue
            content = block.get("content")
            if isinstance(content, str):
                text_parts.append(content)
                continue
        text = getattr(block, "text", None)
        if isinstance(text, str):
            text_parts.append(text)
    return "".join(text_parts)


def _parse_classifier_response(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    content = getattr(response, "content", response)
    if isinstance(content, dict):
        return content
    text = _content_blocks_to_text(content) if isinstance(content, list) else str(content)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("User-intent classifier response must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("User-intent classifier response must be a JSON object")
    return parsed


def _confidence_is_low(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "low"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value < 0.5
    return value is None


def _ambiguous_result(reason: str) -> dict[str, Any]:
    return {
        "target": "ambiguous",
        "target_id": None,
        "relationship": None,
        "confidence": "low",
        "needs_clarification": True,
        "reason": reason,
    }


def validate_user_intent_reference(
    state: AgentState,
    raw_result: dict[str, Any],
    *,
    candidate_user_intent_ids: set[str] | None = None,
    candidate_completed_task_ids: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(raw_result, dict):
        raise ValueError("User-intent reference result must be a dict")
    require_json_safe(raw_result, "user_intent_reference")

    target = raw_result.get("target")
    relationship = raw_result.get("relationship")
    if target not in ALLOWED_USER_INTENT_REFERENCE_TARGETS:
        raise ValueError(f"Unknown user-intent reference target: {target}")
    if relationship is not None and relationship not in ALLOWED_USER_INTENT_RELATIONSHIPS:
        raise ValueError(f"Unknown user-intent relationship: {relationship}")
    if target == "ambiguous" or _confidence_is_low(raw_result.get("confidence")):
        return _ambiguous_result(str(raw_result.get("reason") or "The reference is ambiguous."))
    if target == "new_user_intent":
        return {
            "target": "new_user_intent",
            "target_id": None,
            "relationship": "new_request",
            "confidence": raw_result.get("confidence"),
            "needs_clarification": False,
            "reason": str(raw_result.get("reason") or ""),
        }

    _state, memory = ensure_memory_state(state)
    target_id = raw_result.get("target_id")
    if not isinstance(target_id, str) or not target_id:
        return _ambiguous_result("The classifier did not select a target ID.")

    if target == "existing_user_intent":
        if candidate_user_intent_ids is not None and target_id not in candidate_user_intent_ids:
            return _ambiguous_result("The classifier selected an unavailable user intent.")
        card = memory["user_intents"].get(target_id)
        if not isinstance(card, dict):
            raise ValueError(f"Resolved user intent does not exist: {target_id}")
        if card.get("kind") != "db_rag_query":
            raise ValueError(f"Resolved user intent has unsupported kind: {card.get('kind')}")
        source_question = str(card.get("source_question") or "").strip()
        if not source_question:
            raise ValueError("Resolved user intent has no source_question")
        if relationship not in {"continue", "refine"}:
            return _ambiguous_result(
                "The requested relationship is not supported for user intents."
            )
        return {
            "target": "existing_user_intent",
            "target_id": target_id,
            "kind": "db_rag_query",
            "relationship": relationship,
            "source_question": source_question,
            "goal_text": str(card.get("goal_text") or source_question),
            "confidence": raw_result.get("confidence"),
            "needs_clarification": False,
            "reason": str(raw_result.get("reason") or ""),
        }

    if target == "completed_task":
        if (
            candidate_completed_task_ids is not None
            and target_id not in candidate_completed_task_ids
        ):
            return _ambiguous_result("The classifier selected an unavailable completed task.")
        if relationship != "inspect_result":
            raise ValueError("completed task target has unsupported relationship")
        completed = memory["completed_tasks"]
        task = completed.get(target_id)
        if not isinstance(task, dict):
            raise ValueError(f"Resolved completed task does not exist: {target_id}")
        artifact_refs = task.get("artifact_refs")
        if not isinstance(artifact_refs, dict) or not any(
            artifact_refs.get(key) for key in INSPECTABLE_COMPLETED_TASK_ARTIFACT_KEYS
        ):
            raise ValueError("Resolved completed task has no inspectable artifact ref")
        return {
            "target": "completed_task",
            "target_id": target_id,
            "relationship": relationship,
            "confidence": raw_result.get("confidence"),
            "needs_clarification": False,
            "reason": str(raw_result.get("reason") or ""),
        }

    return _ambiguous_result("The classifier result is ambiguous.")


def classify_user_intent_reference(
    state: AgentState,
    classifier: Any,
    *,
    user_message: str,
    user_message_hash: str,
    recent_turns: list[dict[str, str]] | None = None,
    completed_task_cards: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = build_user_intent_classifier_payload(
        state,
        user_message=user_message,
        user_message_hash=user_message_hash,
        recent_turns=recent_turns,
        completed_task_cards=completed_task_cards,
    )
    if not payload["candidate_user_intents"] and not payload["candidate_completed_tasks"]:
        return {
            "target": "new_user_intent",
            "target_id": None,
            "relationship": "new_request",
            "confidence": "high",
            "needs_clarification": False,
            "reason": "No user intents exist.",
        }
    if not hasattr(classifier, "invoke"):
        raise ValueError("User-intent classifier must expose invoke")
    prompt = f"{CLASSIFIER_INSTRUCTIONS}\n\nPayload:\n{json.dumps(payload, sort_keys=True)}"
    raw_result = _parse_classifier_response(classifier.invoke(prompt))
    candidate_user_intent_ids = {
        card["intent_id"]
        for card in payload["candidate_user_intents"]
        if isinstance(card.get("intent_id"), str)
    }
    candidate_completed_task_ids = {
        card["task_id"]
        for card in payload["candidate_completed_tasks"]
        if isinstance(card.get("task_id"), str)
    }
    return validate_user_intent_reference(
        state,
        raw_result,
        candidate_user_intent_ids=candidate_user_intent_ids,
        candidate_completed_task_ids=candidate_completed_task_ids,
    )


def latest_user_intent(state: AgentState, *, kind: str) -> dict[str, Any] | None:
    _state, memory = ensure_memory_state(state)
    _validate_user_intent_kind(kind)
    intent_id = memory["last_user_intent_id_by_kind"].get(kind)
    if isinstance(intent_id, str):
        card = memory["user_intents"].get(intent_id)
        if isinstance(card, dict) and card.get("kind") == kind:
            return deepcopy(card)

    for fallback_intent_id in reversed(memory["intent_order"]):
        card = memory["user_intents"].get(fallback_intent_id)
        if isinstance(card, dict) and card.get("kind") == kind:
            return deepcopy(card)
    return None


def upsert_user_intent_from_db_rag_intent(
    state: AgentState,
    *,
    active_intent: dict[str, Any],
    source_message_hash: str | None,
    status: str,
    continued_from_intent_id: str | None = None,
    force_new: bool = False,
) -> AgentState:
    state, memory = ensure_memory_state(state)
    active_intent = _validate_db_rag_active_intent(active_intent)
    _validate_optional_text(source_message_hash, "source_message_hash")
    _validate_user_intent_status(status)
    _validate_optional_text(continued_from_intent_id, "continued_from_intent_id")
    if (
        continued_from_intent_id is not None
        and continued_from_intent_id not in memory["user_intents"]
    ):
        raise ValueError(f"Unknown continued_from_intent_id: {continued_from_intent_id}")

    active_intent_id = active_intent.get("intent_id")
    intent_id = None if force_new else _matching_intent_id(memory, active_intent_id)
    if intent_id is None and not force_new:
        intent_id = _latest_live_intent_id(memory, kind="db_rag_query")

    timestamp = _now_iso()
    if intent_id is not None:
        card = memory["user_intents"][intent_id]
        card["goal_text"] = active_intent["goal_text"]
        card["status"] = status
        card["source_message_hash"] = source_message_hash
        card["active_intent_id"] = active_intent_id
        card["updated_at"] = timestamp
        memory["last_user_intent_id"] = intent_id
        memory["last_user_intent_id_by_kind"]["db_rag_query"] = intent_id
        return state

    intent_id = _new_intent_id(memory)
    card = {
        "intent_id": intent_id,
        "display_ordinal": len(memory["intent_order"]) + 1,
        "kind": "db_rag_query",
        "agent": "rag_db_qa",
        "source_question": active_intent["source_question"],
        "goal_text": active_intent["goal_text"],
        "status": status,
        "source_message_hash": source_message_hash,
        "active_intent_id": active_intent_id,
        "completed_task_id": None,
        "continued_from_intent_id": continued_from_intent_id,
        "created_at": timestamp,
        "updated_at": timestamp,
    }
    memory["user_intents"][intent_id] = card
    memory["intent_order"].append(intent_id)
    memory["last_user_intent_id"] = intent_id
    memory["last_user_intent_id_by_kind"]["db_rag_query"] = intent_id
    return state


def update_user_intent_status(
    state: AgentState,
    *,
    status: str,
    intent_id: str | None = None,
    active_intent_id: str | None = None,
) -> AgentState:
    state, memory = ensure_memory_state(state)
    _validate_user_intent_status(status)
    _validate_optional_text(intent_id, "intent_id")
    _validate_optional_text(active_intent_id, "active_intent_id")
    if intent_id is None:
        intent_id = _matching_intent_id(memory, active_intent_id)
    if intent_id is None or intent_id not in memory["user_intents"]:
        return state

    card = memory["user_intents"][intent_id]
    card["status"] = status
    card["updated_at"] = _now_iso()
    return state


def link_user_intent_completed_task(
    state: AgentState,
    *,
    intent_id: str,
    task_id: str,
) -> AgentState:
    state, memory = ensure_memory_state(state)
    _validate_text(intent_id, "intent_id")
    _validate_text(task_id, "task_id")
    if intent_id not in memory["user_intents"]:
        raise ValueError(f"Unknown user intent: {intent_id}")
    task = memory["completed_tasks"].get(task_id)
    if not isinstance(task, dict):
        raise ValueError(f"Unknown task_id: {task_id}")
    if task.get("kind") != "db_rag_sql_extraction":
        raise ValueError("completed task kind must be db_rag_sql_extraction")

    provenance = task.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
        task["provenance"] = provenance
    try:
        require_json_safe(provenance, "provenance")
    except ValueError:
        provenance = {}
        task["provenance"] = provenance
    provenance["originating_user_intent_id"] = intent_id
    require_json_safe(provenance, "provenance")

    card = memory["user_intents"][intent_id]
    card["status"] = "completed"
    card["completed_task_id"] = task_id
    card["updated_at"] = _now_iso()
    return state
