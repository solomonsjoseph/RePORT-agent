from __future__ import annotations

import hashlib

from ...state import MetaKeys
from ...state_views import get_artifacts, get_node_data


def _hash_text(value: object) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return hashlib.sha256(text.encode()).hexdigest()[:12]


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
        "tool_result_count": len(list(artifacts.get("tool_results") or [])),
        "clarification_pending": bool(meta.get("awaiting_user_clarification")),
        "review_status": (
            human_review.get("final_decision")
            or human_review.get("before_run_decision")
            or human_review.get("after_error_decision")
        ),
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

    code_changed = previous.get("generated_code_hash") != current.get("generated_code_hash")
    repeated_error = (
        previous.get("execution_status") == "error"
        and current.get("execution_status") == "error"
        and previous.get("error_signature")
        == current.get("error_signature")
        == current.get("error_signature")
    )
    return code_changed and not repeated_error


def update_progress_tracking(state: dict) -> dict:
    meta = dict(state.get("meta") or {})
    current = build_progress_snapshot(state)
    previous = dict(meta.get(MetaKeys.PROGRESS_SNAPSHOT) or {})
    progressed = classify_progress(previous or None, current)
    meta[MetaKeys.PROGRESS_SNAPSHOT] = current
    meta[MetaKeys.PROGRESS_MADE_LAST_STEP] = progressed
    meta[MetaKeys.STAGNATION_COUNT] = 0 if progressed else int(meta.get(MetaKeys.STAGNATION_COUNT, 0)) + 1
    meta[MetaKeys.REPEATED_FAILURE_SIGNATURE] = current.get("error_signature")
    return {**state, "meta": meta}
