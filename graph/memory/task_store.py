from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from graph.state import AgentState

from .schema import (
    ALLOWED_RELATIONSHIPS,
    ALLOWED_TASK_KINDS,
    EDITABLE_TASK_FIELDS,
    MemoryState,
    require_json_safe,
)


EMPTY_MEMORY_STATE: MemoryState = {
    "completed_tasks": {},
    "failed_tasks": {},
    "task_order": [],
    "last_task_id": None,
    "last_task_id_by_kind": {},
    "last_failed_task_id_by_kind": {},
    "last_reference_resolution": None,
    "pending_reference_clarification": None,
    "user_intents": {},
    "intent_order": [],
    "last_user_intent_id": None,
    "last_user_intent_id_by_kind": {},
}

MEMORY_KEY_TYPES = {
    "completed_tasks": dict,
    "failed_tasks": dict,
    "task_order": list,
    "last_task_id": (str, type(None)),
    "last_task_id_by_kind": dict,
    "last_failed_task_id_by_kind": dict,
    "last_reference_resolution": (dict, type(None)),
    "pending_reference_clarification": (dict, type(None)),
    "user_intents": dict,
    "intent_order": list,
    "last_user_intent_id": (str, type(None)),
    "last_user_intent_id_by_kind": dict,
}

COMPACT_CARD_FIELDS = (
    "task_id",
    "display_ordinal",
    "kind",
    "label",
    "source_question",
    "goal_text",
    "summary",
    "artifact_refs",
    "parent_task_id",
    "relationship_to_parent",
    "status",
    "created_at",
    "completed_at",
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _empty_memory() -> dict[str, Any]:
    return deepcopy(EMPTY_MEMORY_STATE)


def ensure_memory_state(state: AgentState) -> tuple[AgentState, dict[str, Any]]:
    memory = state.get("memory")
    if not isinstance(memory, dict):
        memory = _empty_memory()
        state["memory"] = memory
        return state, memory

    defaults = _empty_memory()
    for key, value in defaults.items():
        if key not in memory or not isinstance(memory[key], MEMORY_KEY_TYPES[key]):
            memory[key] = value
    completed_tasks = memory["completed_tasks"]
    task_order = [
        task_id
        for task_id in memory["task_order"]
        if isinstance(task_id, str) and task_id in completed_tasks
    ]
    memory["task_order"] = task_order
    if memory["last_task_id"] not in completed_tasks:
        memory["last_task_id"] = task_order[-1] if task_order else None
    memory["last_task_id_by_kind"] = {
        kind: task_id
        for kind, task_id in memory["last_task_id_by_kind"].items()
        if isinstance(kind, str) and isinstance(task_id, str) and task_id in completed_tasks
    }
    user_intents = memory["user_intents"]
    intent_order = [
        intent_id
        for intent_id in memory["intent_order"]
        if isinstance(intent_id, str) and intent_id in user_intents
    ]
    memory["intent_order"] = intent_order
    if memory["last_user_intent_id"] not in user_intents:
        memory["last_user_intent_id"] = intent_order[-1] if intent_order else None
    memory["last_user_intent_id_by_kind"] = {
        kind: intent_id
        for kind, intent_id in memory["last_user_intent_id_by_kind"].items()
        if isinstance(kind, str) and isinstance(intent_id, str) and intent_id in user_intents
    }
    return state, memory


def _validate_parent(memory: dict[str, Any], parent_task_id: str | None) -> None:
    if parent_task_id is None:
        return
    if parent_task_id not in memory["completed_tasks"]:
        raise ValueError(f"Unknown parent_task_id: {parent_task_id}")


def _validate_task_kind(kind: str) -> None:
    if kind not in ALLOWED_TASK_KINDS:
        raise ValueError(f"Unknown task kind: {kind}")


def _validate_relationship(relationship_to_parent: str | None) -> None:
    if relationship_to_parent is not None and relationship_to_parent not in ALLOWED_RELATIONSHIPS:
        raise ValueError(f"Unknown relationship_to_parent: {relationship_to_parent}")


def _validate_string_refs(refs: dict[str, Any], field_name: str) -> None:
    require_json_safe(refs, field_name)
    if not all(isinstance(value, str) for value in refs.values()):
        raise ValueError(f"{field_name} must be dict[str, str]")


def _validate_text(value: Any, field_name: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    require_json_safe(value, field_name)


def _validate_tags(value: Any) -> None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("tags must be list[str]")
    require_json_safe(value, "tags")


def _validate_optional_text(value: Any, field_name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{field_name} must be str | None")
    require_json_safe(value, field_name)


def complete_task(
    state: AgentState,
    *,
    kind: str,
    source_question: str,
    goal_text: str,
    label: str,
    summary: str,
    artifact_refs: dict[str, str] | None = None,
    event_refs: dict[str, str] | None = None,
    parent_task_id: str | None = None,
    relationship_to_parent: str | None = None,
    provenance: dict[str, Any] | None = None,
    tags: list[Any] | None = None,
) -> AgentState:
    state, memory = ensure_memory_state(state)
    _validate_task_kind(kind)
    _validate_parent(memory, parent_task_id)
    _validate_relationship(relationship_to_parent)

    artifact_refs = dict(artifact_refs or {})
    event_refs = dict(event_refs or {})
    provenance = dict(provenance or {})
    _validate_string_refs(artifact_refs, "artifact_refs")
    _validate_string_refs(event_refs, "event_refs")
    require_json_safe(provenance, "provenance")
    _validate_text(source_question, "source_question")
    _validate_text(goal_text, "goal_text")
    _validate_text(label, "label")
    _validate_text(summary, "summary")
    if tags is not None:
        _validate_tags(tags)

    task_id = f"task_{uuid4().hex[:8]}"
    while task_id in memory["completed_tasks"]:
        task_id = f"task_{uuid4().hex[:8]}"
    timestamp = _now_iso()
    card: dict[str, Any] = {
        "task_id": task_id,
        "display_ordinal": len(memory["task_order"]) + 1,
        "kind": kind,
        "label": label,
        "source_question": source_question,
        "goal_text": goal_text,
        "summary": summary,
        "artifact_refs": deepcopy(artifact_refs),
        "event_refs": deepcopy(event_refs),
        "parent_task_id": parent_task_id,
        "relationship_to_parent": relationship_to_parent,
        "provenance": deepcopy(provenance),
        "status": "completed",
        "created_at": timestamp,
        "completed_at": timestamp,
        "last_enriched_at": None,
    }
    if tags is not None:
        card["tags"] = deepcopy(tags)

    memory["completed_tasks"][task_id] = card
    memory["task_order"].append(task_id)
    memory["last_task_id"] = task_id
    memory["last_task_id_by_kind"][kind] = task_id
    return state


def update_task_enrichment(state: AgentState, task_id: str, **fields: Any) -> AgentState:
    state, memory = ensure_memory_state(state)
    unknown_fields = set(fields) - EDITABLE_TASK_FIELDS
    if unknown_fields:
        raise ValueError(f"Cannot update immutable task fields: {sorted(unknown_fields)}")
    try:
        card = memory["completed_tasks"][task_id]
    except KeyError as exc:
        raise ValueError(f"Unknown task_id: {task_id}") from exc

    for key, value in fields.items():
        if key in {"label", "summary"}:
            _validate_text(value, key)
        elif key == "tags":
            _validate_tags(value)
        elif key == "last_enriched_at":
            _validate_optional_text(value, key)
        else:
            require_json_safe(value, key)
        card[key] = deepcopy(value)
    return state


def cache_reference_resolution(
    state: AgentState,
    user_message_hash: str,
    result: dict[str, Any],
) -> AgentState:
    state, memory = ensure_memory_state(state)
    require_json_safe(result, "result")
    memory["last_reference_resolution"] = {
        "user_message_hash": user_message_hash,
        "result": deepcopy(result),
    }
    return state


def get_cached_reference_resolution(
    state: AgentState,
    user_message_hash: str,
) -> dict[str, Any] | None:
    _state, memory = ensure_memory_state(state)
    cached = memory.get("last_reference_resolution")
    if not isinstance(cached, dict):
        return None
    if cached.get("user_message_hash") != user_message_hash:
        return None
    result = cached.get("result")
    return deepcopy(result) if isinstance(result, dict) else None


def _compact_card(card: dict[str, Any]) -> dict[str, Any]:
    return {field: deepcopy(card.get(field)) for field in COMPACT_CARD_FIELDS}


def latest_task_cards(
    state: AgentState,
    limit: int = 12,
    extra_task_ids: list[str] | set[str] | tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    _state, memory = ensure_memory_state(state)
    completed = memory["completed_tasks"]
    ordered_ids = list(reversed(memory["task_order"]))
    selected_ids = ordered_ids[: max(limit, 0)]

    for task_id in extra_task_ids or []:
        if task_id in completed and task_id not in selected_ids:
            selected_ids.append(task_id)

    return [_compact_card(completed[task_id]) for task_id in selected_ids if task_id in completed]
