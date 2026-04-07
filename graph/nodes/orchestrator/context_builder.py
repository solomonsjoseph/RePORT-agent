from __future__ import annotations

import json

from ...state import MetaKeys
from ...state_views import get_artifacts, get_node_data, get_planner_state, merge_state_patch
from ..action_metadata import ACTION_CAPABILITIES
from .state_logic import _latest_user_message


def build_planner_runtime_state(state: dict, available_actions: list[str]) -> dict:
    node_names = set(available_actions)
    node_names.update({"executor", "human_review"})
    node_data = {name: get_node_data(state, name) for name in sorted(node_names)}
    planner_state = get_planner_state(state)

    return merge_state_patch(
        state,
        {
            "artifacts": get_artifacts(state),
            "node_data": node_data,
            "planner": planner_state,
        },
    )


def build_planner_context(state: dict, available_actions: list[str]) -> dict:
    artifacts = get_artifacts(state)
    planner_state = get_planner_state(state)
    meta = dict(state.get("meta") or {})
    observations = list(state.get("observations") or [])
    workflow_trace = list(meta.get(MetaKeys.WORKFLOW_TRACE, []))
    executor = get_node_data(state, "executor")
    review = get_node_data(state, "human_review")
    summary_lines = [
        f"latest_user_message={_latest_user_message(state)}",
        f"last_action={state.get('last_action')}",
        f"workflow_trace_tail={workflow_trace[-8:]}",
        f"recent_observations={observations[-6:]}",
        f"generated_code_present={bool(artifacts.get('generated_code'))}",
        f"executor_run_status={executor.get('run_status')}",
        f"review_state={json.dumps(review, default=str, sort_keys=True)}",
        f"workflow_milestone={meta.get(MetaKeys.WORKFLOW_MILESTONE)}",
        f"completion_status={meta.get(MetaKeys.COMPLETION_STATUS)}",
        f"blocker_signature={meta.get(MetaKeys.BLOCKER_SIGNATURE)}",
        f"progress_class={meta.get(MetaKeys.PROGRESS_CLASS)}",
        f"progress_made_last_step={meta.get(MetaKeys.PROGRESS_MADE_LAST_STEP)}",
        f"stagnation_count={meta.get(MetaKeys.STAGNATION_COUNT, 0)}",
        f"weak_progress_count={meta.get(MetaKeys.WEAK_PROGRESS_COUNT, 0)}",
        f"repeated_failure_signature={meta.get(MetaKeys.REPEATED_FAILURE_SIGNATURE)}",
    ]
    node_caps = "\n".join(
        f"- {name}: {ACTION_CAPABILITIES[name]}"
        for name in available_actions
        if name in ACTION_CAPABILITIES
    )
    return {
        "environment_summary": "\n".join(summary_lines),
        "node_capabilities": node_caps,
        "recent_observations": observations[-6:],
        "planner_decision_trace": list(planner_state.get("decision_trace") or [])[-5:],
        "artifacts": artifacts,
    }
