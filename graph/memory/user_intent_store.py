from __future__ import annotations

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
    for intent_id in reversed(memory["intent_order"]):
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


def latest_user_intent(state: AgentState, *, kind: str) -> dict[str, Any] | None:
    cards = compact_user_intent_cards(state, limit=1, kind=kind)
    return deepcopy(cards[0]) if cards else None


def upsert_user_intent_from_db_rag_intent(
    state: AgentState,
    *,
    active_intent: dict[str, Any],
    source_message_hash: str,
    status: str,
    continued_from_intent_id: str | None = None,
    force_new: bool = False,
) -> AgentState:
    state, memory = ensure_memory_state(state)
    active_intent = _validate_db_rag_active_intent(active_intent)
    _validate_text(source_message_hash, "source_message_hash")
    _validate_user_intent_status(status)
    _validate_optional_text(continued_from_intent_id, "continued_from_intent_id")

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
        raise ValueError("Unknown user intent")

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

    card = memory["user_intents"][intent_id]
    card["status"] = "completed"
    card["completed_task_id"] = task_id
    card["updated_at"] = _now_iso()
    return state
