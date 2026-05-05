from __future__ import annotations

from copy import deepcopy
from typing import Any

from graph.state import AgentState

from .schema import ALLOWED_REFERENCE_LABELS, ALLOWED_RELATIONSHIPS, require_json_safe
from .task_store import ensure_memory_state


REQUIRED_ARTIFACT_REFS = {
    ("db_rag_sql_extraction", "inspect_artifact"): {"sql_candidate_artifact_id"},
    ("db_rag_sql_extraction", "revision"): {
        "selection_artifact_id",
        "sql_candidate_artifact_id",
    },
    ("db_rag_sql_extraction", "use_as_input"): {"dataset_artifact_id"},
}

EXPECTED_FILE_ARTIFACT_KINDS = {
    "selection_artifact_id": "db_rag_column_selection",
    "sql_candidate_artifact_id": "db_rag_sql_candidate",
}

EXPECTED_DATASET_ARTIFACT_KINDS = {
    "dataset_artifact_id": "subset",
}

ALLOWED_RELATIONSHIPS_BY_TASK_KIND = {
    "qa_answer": {"revision", "rerun", "explain", "compare"},
    "db_rag_metadata_answer": {"revision", "rerun", "explain", "inspect_artifact", "compare"},
    "db_rag_sql_extraction": ALLOWED_RELATIONSHIPS,
    "code_analysis": ALLOWED_RELATIONSHIPS,
}


def _base_result(raw_result: dict[str, Any], *, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "task_id": None,
        "relationship": None,
        "intended_action": raw_result.get("intended_action"),
        "confidence": raw_result.get("confidence"),
        "needs_reference": raw_result.get("needs_reference", False),
        "reason": raw_result.get("reason") or "",
    }


def _validate_scalar_fields(raw_result: dict[str, Any]) -> None:
    if "needs_reference" in raw_result and not isinstance(raw_result["needs_reference"], bool):
        raise ValueError("needs_reference must be a bool")

    if "reason" in raw_result and not isinstance(raw_result["reason"], str):
        raise ValueError("reason must be a string")

    intended_action = raw_result.get("intended_action")
    if intended_action is not None and not isinstance(intended_action, str):
        raise ValueError("intended_action must be str | None")

    confidence = raw_result.get("confidence")
    if isinstance(confidence, bool) or (
        confidence is not None and not isinstance(confidence, (str, int, float))
    ):
        raise ValueError("confidence must be str | int | float | None")


def _clarification_result(
    raw_result: dict[str, Any],
    *,
    label: str,
    candidate_cards: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    result = _base_result(raw_result, label=label)
    result["needs_reference"] = True
    result["candidate_cards"] = deepcopy(candidate_cards or [])
    return result


def _artifact_store_for_ref(state: AgentState, ref_name: str) -> dict[str, Any]:
    artifacts = state.get("artifacts") if isinstance(state.get("artifacts"), dict) else {}
    store_key = "datasets" if ref_name == "dataset_artifact_id" else "files"
    store = artifacts.get(store_key) if isinstance(artifacts, dict) else {}
    return store if isinstance(store, dict) else {}


def _validate_artifact_kind(ref_name: str, artifact: Any) -> None:
    expected_kind = EXPECTED_FILE_ARTIFACT_KINDS.get(ref_name)
    if expected_kind is None:
        expected_kind = EXPECTED_DATASET_ARTIFACT_KINDS.get(ref_name)
    if expected_kind is None:
        return
    actual_kind = artifact.get("kind") if isinstance(artifact, dict) else None
    if actual_kind != expected_kind:
        raise ValueError(
            f"Required artifact ref has wrong kind: {ref_name} "
            f"expected {expected_kind}, got {actual_kind}"
        )


def _validate_required_artifacts(
    state: AgentState,
    card: dict[str, Any],
    relationship: str,
) -> None:
    required_refs = REQUIRED_ARTIFACT_REFS.get((card.get("kind"), relationship), set())
    artifact_refs = card.get("artifact_refs") if isinstance(card.get("artifact_refs"), dict) else {}
    for ref_name in sorted(required_refs):
        artifact_id = artifact_refs.get(ref_name)
        if not isinstance(artifact_id, str) or not artifact_id:
            raise ValueError(f"Missing required artifact ref: {ref_name}")
        artifact_store = _artifact_store_for_ref(state, ref_name)
        if artifact_id not in artifact_store:
            raise ValueError(f"Required artifact ref does not exist: {ref_name}")
        _validate_artifact_kind(ref_name, artifact_store[artifact_id])


def validate_reference_resolution(
    state: AgentState,
    raw_result: dict[str, Any],
    candidate_cards: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not isinstance(raw_result, dict):
        raise ValueError("Reference resolution result must be a dict")
    require_json_safe(raw_result, "raw_result")
    _validate_scalar_fields(raw_result)

    _state, memory = ensure_memory_state(state)
    label = raw_result.get("label")
    if label not in ALLOWED_REFERENCE_LABELS:
        raise ValueError(f"Unknown reference resolution label: {label}")

    if label == "new_task":
        result = _base_result(raw_result, label="new_task")
        result["needs_reference"] = False
        return result

    if label == "unknown":
        if raw_result.get("needs_reference", False):
            return _clarification_result(raw_result, label="unknown", candidate_cards=candidate_cards)
        result = _base_result(raw_result, label="new_task")
        result["needs_reference"] = False
        return result

    if label == "ambiguous":
        return _clarification_result(raw_result, label="ambiguous", candidate_cards=candidate_cards)

    task_id = raw_result.get("task_id")
    if not isinstance(task_id, str) or task_id not in memory["completed_tasks"]:
        raise ValueError(f"Resolved task_id does not exist: {task_id}")

    card = memory["completed_tasks"][task_id]
    relationship = raw_result.get("relationship")
    if relationship not in ALLOWED_RELATIONSHIPS:
        raise ValueError(f"Unknown relationship: {relationship}")
    allowed_for_kind = ALLOWED_RELATIONSHIPS_BY_TASK_KIND.get(card.get("kind"), set())
    if relationship not in allowed_for_kind:
        raise ValueError(f"Relationship {relationship} is not allowed for task kind {card.get('kind')}")

    _validate_required_artifacts(state, card, relationship)

    return {
        "label": "resolved",
        "task_id": task_id,
        "task_kind": card.get("kind"),
        "relationship": relationship,
        "intended_action": raw_result.get("intended_action"),
        "confidence": raw_result.get("confidence"),
        "needs_reference": False,
        "reason": raw_result.get("reason") or "",
    }
