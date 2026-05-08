from __future__ import annotations

from typing import Any, TypedDict


ALLOWED_TASK_KINDS = {
    "qa_answer",
    "db_rag_metadata_answer",
    "db_rag_sql_extraction",
    "code_analysis",
}
ALLOWED_REFERENCE_LABELS = {"resolved", "new_task", "ambiguous", "unknown"}
ALLOWED_RELATIONSHIPS = {"revision", "rerun", "explain", "inspect_artifact", "use_as_input", "compare"}
EDITABLE_TASK_FIELDS = {"label", "summary", "tags", "last_enriched_at"}
ALLOWED_USER_INTENT_KINDS = {"db_rag_query"}
ALLOWED_USER_INTENT_STATUSES = {
    "active",
    "awaiting_extraction_opt_in",
    "awaiting_column_review",
    "awaiting_sql_review",
    "cancelled",
    "completed",
    "superseded",
    "declined",
}


class TaskCard(TypedDict, total=False):
    task_id: str
    display_ordinal: int
    kind: str
    label: str
    source_question: str
    goal_text: str
    summary: str
    tags: list[str]
    artifact_refs: dict[str, str]
    event_refs: dict[str, str]
    parent_task_id: str | None
    relationship_to_parent: str | None
    provenance: dict[str, Any]
    status: str
    created_at: str
    completed_at: str
    last_enriched_at: str | None


class UserIntentCard(TypedDict, total=False):
    intent_id: str
    display_ordinal: int
    kind: str
    agent: str
    source_question: str
    goal_text: str
    status: str
    source_message_hash: str | None
    active_intent_id: str | None
    completed_task_id: str | None
    continued_from_intent_id: str | None
    created_at: str
    updated_at: str


class ReferenceResolution(TypedDict, total=False):
    label: str
    task_id: str | None
    relationship: str | None
    intended_action: str | None
    confidence: str | float | int | None
    needs_reference: bool
    reason: str


class MemoryState(TypedDict):
    completed_tasks: dict[str, TaskCard]
    failed_tasks: dict[str, TaskCard]
    task_order: list[str]
    last_task_id: str | None
    last_task_id_by_kind: dict[str, str]
    last_failed_task_id_by_kind: dict[str, str]
    last_reference_resolution: dict[str, Any] | None
    pending_reference_clarification: dict[str, Any] | None
    user_intents: dict[str, UserIntentCard]
    intent_order: list[str]
    last_user_intent_id: str | None
    last_user_intent_id_by_kind: dict[str, str]


def is_json_safe(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, list):
        return all(is_json_safe(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and is_json_safe(item) for key, item in value.items())
    return False


def require_json_safe(value: Any, field_name: str) -> None:
    if not is_json_safe(value):
        raise ValueError(f"{field_name} must be JSON-safe")
