from __future__ import annotations

import hashlib

from ...state_views import get_artifacts, get_node_data

PROGRESS_SNAPSHOT = "progress_snapshot"
PROGRESS_MADE_LAST_STEP = "progress_made_last_step"
STAGNATION_COUNT = "stagnation_count"
REPEATED_FAILURE_SIGNATURE = "repeated_failure_signature"
REVIEW_STATUS_FIELDS = (
    "final_decision",
    "before_run_decision",
    "after_error_decision",
)


def _hash_text(value: object) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _count_tool_results(section: object) -> int:
    if not isinstance(section, dict):
        return 0

    total = 0
    for node_state in section.values():
        if not isinstance(node_state, dict):
            continue
        tool_results = node_state.get("tool_results")
        if isinstance(tool_results, list):
            total += len(tool_results)
    return total


def build_progress_snapshot(state: dict) -> dict:
    artifacts = get_artifacts(state)
    executor = get_node_data(state, "executor")
    human_review = get_node_data(state, "human_review")
    meta = dict(state.get("meta") or {})
    error = dict(artifacts.get("error") or {})
    return {
        "last_action": state.get("last_action"),
        "generated_code_hash": _hash_text(artifacts.get("generated_code")),
        "execution_status": executor.get("run_status"),
        "error_signature": f"{error.get('category')}:{error.get('message')}" if error else None,
        "tool_result_count": _count_tool_results(state.get("agents"))
        + _count_tool_results(state.get("node_data")),
        "clarification_pending": bool(meta.get("awaiting_user_clarification")),
        "review_status": {field: human_review.get(field) for field in REVIEW_STATUS_FIELDS},
        "user_visible_output_hash": _hash_text(artifacts.get("execution_output")),
    }


def classify_progress(previous: dict | None, current: dict) -> bool:
    if not previous:
        return True

    meaningful_keys = (
        "execution_status",
        "error_signature",
        "tool_result_count",
        "clarification_pending",
        "review_status",
        "user_visible_output_hash",
    )
    if any(previous.get(key) != current.get(key) for key in meaningful_keys):
        return True

    return previous.get("generated_code_hash") != current.get("generated_code_hash")


def update_progress_tracking(state: dict) -> dict:
    meta = dict(state.get("meta") or {})
    current = build_progress_snapshot(state)
    previous = dict(meta.get(PROGRESS_SNAPSHOT) or {})
    progressed = classify_progress(previous or None, current)
    meta[PROGRESS_SNAPSHOT] = current
    meta[PROGRESS_MADE_LAST_STEP] = progressed
    meta[STAGNATION_COUNT] = 0 if progressed else int(meta.get(STAGNATION_COUNT, 0)) + 1
    meta[REPEATED_FAILURE_SIGNATURE] = current.get("error_signature")
    return {**state, "meta": meta}
