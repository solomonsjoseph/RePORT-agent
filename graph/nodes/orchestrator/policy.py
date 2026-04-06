from __future__ import annotations

from typing import Iterable

from ...state import AgentState, MetaKeys
from ..node_registry import NODE_REGISTRY
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
        node = next((nd for nd in NODE_REGISTRY if nd.name == action_name), None)
        if node and action_name in available and node.is_ready(state):
            return action_name

    generate_node = next((nd for nd in NODE_REGISTRY if nd.name == "generate_code"), None)
    if (
        generate_node
        and "generate_code" in available
        and generate_node.is_ready(state)
        and "generate_code" in list((state.get("meta") or {}).get(MetaKeys.LOOP_GUARD_BYPASS_ACTIONS, []))
    ):
        return "generate_code"

    if get_agent_state(state, "human_review").get("final_decision") == "approve":
        return "end"

    return None


def choose_next_action(state: AgentState, available_actions: Iterable[str]) -> str:
    """Return the last-resort fallback action when the planner cannot help.

    Semantic routing should come from the planner. This helper exists to keep
    deterministic invariants intact and to provide a stable fallback for
    malformed or empty planner responses.
    """
    available = set(available_actions)

    invariant_action = choose_invariant_action(state, available_actions)
    if invariant_action:
        return invariant_action

    qa_node = next((nd for nd in NODE_REGISTRY if nd.name == "qa"), None)

    # Last-resort deterministic fallback: choose among semantic nodes only.
    if qa_node and "qa" in available and qa_node.is_ready(state):
        return "qa"
    generate_node = next((nd for nd in NODE_REGISTRY if nd.name == "generate_code"), None)
    if generate_node and "generate_code" in available and generate_node.is_ready(state):
        return "generate_code"

    return "end"


def should_prefer_policy_action(
    state: AgentState,
    llm_action: str,
    policy_action: str,
) -> bool:
    """Return True when deterministic routing should override the planner.

    The planner chooses semantic actions, but it should not overrule
    deterministic system transitions.
    """
    if not policy_action or policy_action == llm_action:
        return False

    invariant_action = choose_invariant_action(state, {llm_action, policy_action})
    if invariant_action == policy_action:
        return True

    return False
