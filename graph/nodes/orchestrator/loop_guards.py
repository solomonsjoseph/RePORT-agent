from __future__ import annotations

from ...state import AgentState, MetaKeys
from .constants import ACTION_LOOKBACK_ACTIONS, LOOP_GUARD_LOOKBACK, MAX_ACTION_REPEATS


def _action_trace(trace: list[str], lookback_actions: int = ACTION_LOOKBACK_ACTIONS) -> list[str]:
    actions = [step for step in trace if step != "orchestrator"]
    return actions[-lookback_actions:]


def _is_expected_retry_pair(a: str, b: str) -> bool:
    return {a, b} == {"execute_code", "error_handler"}


def _detect_two_node_cycle(trace: list[str], lookback_actions: int = ACTION_LOOKBACK_ACTIONS) -> bool:
    action_window = _action_trace(trace, lookback_actions=lookback_actions)
    if len(action_window) < 4:
        return False
    for i in range(len(action_window) - 3):
        a, b, c, d = (
            action_window[i],
            action_window[i + 1],
            action_window[i + 2],
            action_window[i + 3],
        )
        if a == c and b == d and a != b and not _is_expected_retry_pair(a, b):
            return True
    return False


def _count_action_in_recent_trace(
    action: str,
    trace: list[str],
    lookback_actions: int = ACTION_LOOKBACK_ACTIONS,
) -> int:
    return _action_trace(trace, lookback_actions=lookback_actions).count(action)


def _is_loop_guard_bypassed_for_action(state: AgentState, next_action: str) -> bool:
    bypass_actions = list((state.get("meta") or {}).get(MetaKeys.LOOP_GUARD_BYPASS_ACTIONS, []))
    return next_action in bypass_actions


def _apply_loop_guards(
    next_action: str,
    state: AgentState,
    observations: list[str],
) -> tuple[str, list[str], bool]:
    if next_action == "end":
        return next_action, observations, False

    trace = list((state.get("meta") or {}).get(MetaKeys.WORKFLOW_TRACE, []))
    obs = list(observations)

    if _is_loop_guard_bypassed_for_action(state, next_action):
        obs.append(
            f"orchestrator [loop_guard]: bypass for human-requested action '{next_action}'"
        )
        return next_action, obs, False

    if _detect_two_node_cycle(trace):
        obs.append(
            f"orchestrator [loop_guard]: two-node cycle detected "
            f"(tail={trace[-LOOP_GUARD_LOOKBACK:]}); overriding '{next_action}' → 'end'"
        )
        return "end", obs, True

    repeat_count = _count_action_in_recent_trace(next_action, trace)
    if repeat_count >= MAX_ACTION_REPEATS:
        obs.append(
            f"orchestrator [loop_guard]: '{next_action}' appeared {repeat_count}× "
            f"in the last {ACTION_LOOKBACK_ACTIONS} action steps (max={MAX_ACTION_REPEATS}); "
            "overriding → 'end'"
        )
        return "end", obs, True

    return next_action, obs, False
