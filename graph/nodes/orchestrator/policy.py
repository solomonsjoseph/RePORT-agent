from __future__ import annotations

from typing import Iterable

from ...state import AgentState, MetaKeys
from ..node_registry import NODE_REGISTRY_MAP
from ..state_helpers import get_agent_state
from .state_logic import _should_end_now

DETERMINISTIC_CONTROL_ACTIONS = (
    "tool_handler",
    "error_handler",
    "terminal_execution_error",
    "human_review_after_error",
    "human_review_before_run",
    "execute_code",
    "human_review_final",
)


def _tool_request_queue(state: AgentState) -> list[str]:
    return list((state.get("meta") or {}).get(MetaKeys.TOOL_REQUEST_QUEUE, []))


def _next_tool_requester(state: AgentState) -> str | None:
    for requester in _tool_request_queue(state):
        agent_state = get_agent_state(state, requester)
        if agent_state.get("tool_results"):
            return requester
    return None


def choose_invariant_action(state: AgentState, available_actions: Iterable[str]) -> str | None:
    """Return a deterministic action for state-machine invariants only.

    This layer should own transitions that are unambiguous from current state,
    regardless of planner quality.
    """
    available = set(available_actions)

    if _should_end_now(state):
        return "end"

    for action_name in DETERMINISTIC_CONTROL_ACTIONS:
        node = NODE_REGISTRY_MAP.get(action_name)
        if node and action_name in available and node.is_ready(state):
            return action_name

    generate_node = NODE_REGISTRY_MAP.get("generate_code")
    if (
        generate_node
        and "generate_code" in available
        and generate_node.is_ready(state)
        and "generate_code" in list((state.get("meta") or {}).get(MetaKeys.LOOP_GUARD_BYPASS_ACTIONS, []))
    ):
        return "generate_code"

    return None
