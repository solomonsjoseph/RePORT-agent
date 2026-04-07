from __future__ import annotations

import hashlib

from ...state import AgentState, MetaKeys
from ...state_views import get_artifacts
from .state_logic import _user_message_hash

MAX_STAGNATION_STEPS = 4
MAX_WEAK_PROGRESS_STEPS = 4


def _hash_text(value: object) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def build_progress_snapshot(state: AgentState) -> dict:
    meta = dict(state.get("meta") or {})
    artifacts = get_artifacts(state)
    error = dict(artifacts.get("error") or {})
    return {
        "milestone": meta.get(MetaKeys.WORKFLOW_MILESTONE),
        "completion_status": meta.get(MetaKeys.COMPLETION_STATUS),
        "blocker_signature": meta.get(MetaKeys.BLOCKER_SIGNATURE),
        "generated_code_hash": _hash_text(artifacts.get("generated_code")),
        "error_signature": (
            f"{error.get('category')}:{error.get('type')}:{_hash_text(error.get('message'))}"
            if error
            else None
        ),
        "selected_action": state.get("last_action"),
        "user_turn_hash": meta.get(MetaKeys.LAST_USER_MESSAGE_HASH) or _user_message_hash(state),
    }


def classify_progress(previous: dict | None, current: dict) -> str:
    if not previous:
        return "hard"

    if current.get("user_turn_hash") and previous.get("user_turn_hash") != current.get("user_turn_hash"):
        return "hard"

    if previous.get("milestone") != current.get("milestone"):
        return "hard"

    if previous.get("completion_status") != current.get("completion_status"):
        return "hard"

    if previous.get("blocker_signature") != current.get("blocker_signature"):
        return "hard"

    if previous.get("generated_code_hash") != current.get("generated_code_hash"):
        return "weak"

    if previous.get("error_signature") != current.get("error_signature"):
        return "weak"

    return "none"


def update_recurrence_state(state: AgentState) -> AgentState:
    meta = dict(state.get("meta") or {})
    current = build_progress_snapshot(state)
    previous = dict(meta.get(MetaKeys.PROGRESS_SNAPSHOT) or {})
    progress_class = classify_progress(previous or None, current)

    stagnation_count = int(meta.get(MetaKeys.STAGNATION_COUNT, 0))
    weak_progress_count = int(meta.get(MetaKeys.WEAK_PROGRESS_COUNT, 0))

    if progress_class == "hard":
        stagnation_count = 0
        weak_progress_count = 0
    elif progress_class == "weak":
        weak_progress_count += 1
    else:
        stagnation_count += 1

    meta[MetaKeys.PROGRESS_SNAPSHOT] = current
    meta[MetaKeys.PROGRESS_CLASS] = progress_class
    meta[MetaKeys.PROGRESS_MADE_LAST_STEP] = progress_class != "none"
    meta[MetaKeys.STAGNATION_COUNT] = stagnation_count
    meta[MetaKeys.WEAK_PROGRESS_COUNT] = weak_progress_count
    meta[MetaKeys.REPEATED_FAILURE_SIGNATURE] = current.get("error_signature")

    return {**state, "meta": meta}


def apply_recurrence_guard(
    next_action: str,
    state: AgentState,
    observations: list[str],
) -> tuple[str, list[str], bool]:
    if next_action == "end":
        return next_action, observations, False

    meta = dict(state.get("meta") or {})
    obs = list(observations)
    bypass_actions = list(meta.get(MetaKeys.LOOP_GUARD_BYPASS_ACTIONS, []))
    if next_action in bypass_actions:
        obs.append(f"orchestrator [recurrence]: bypass for human-requested action '{next_action}'")
        return next_action, obs, False

    milestone = meta.get(MetaKeys.WORKFLOW_MILESTONE) or "unknown"
    blocker = meta.get(MetaKeys.BLOCKER_SIGNATURE) or "none"

    if int(meta.get(MetaKeys.STAGNATION_COUNT, 0)) >= MAX_STAGNATION_STEPS:
        obs.append(
            "orchestrator [recurrence]: stagnating "
            f"(milestone={milestone}, blocker={blocker}); overriding '{next_action}' -> 'end'"
        )
        return "end", obs, True

    if int(meta.get(MetaKeys.WEAK_PROGRESS_COUNT, 0)) >= MAX_WEAK_PROGRESS_STEPS:
        obs.append(
            "orchestrator [recurrence]: weak-progress churn "
            f"(milestone={milestone}, blocker={blocker}); overriding '{next_action}' -> 'end'"
        )
        return "end", obs, True

    return next_action, obs, False
