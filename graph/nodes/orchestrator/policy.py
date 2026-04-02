from __future__ import annotations

from typing import Iterable

from ...state import AgentState, MetaKeys
from ..node_registry import NODE_REGISTRY
from ..state_helpers import get_agent_state
from ..tool_routing import is_tool_requested
from .intent import infer_intent_from_latest_user
from .state_logic import _should_end_now


def _tool_request_queue(state: AgentState) -> list[str]:
    return list((state.get("meta") or {}).get(MetaKeys.TOOL_REQUEST_QUEUE, []))


def _next_tool_requester(state: AgentState) -> str | None:
    for requester in _tool_request_queue(state):
        agent_state = get_agent_state(state, requester)
        if agent_state.get("tool_results"):
            return requester
    return None


def _current_intent(state: AgentState) -> str:
    return (
        (state.get("meta") or {}).get(MetaKeys.INTENT)
        or infer_intent_from_latest_user(state)
        or ""
    ).strip()


def choose_invariant_action(state: AgentState, available_actions: Iterable[str]) -> str | None:
    """Return a deterministic action for state-machine invariants only.

    This layer should own transitions that are unambiguous from current state,
    regardless of planner quality.
    """
    available = set(available_actions)

    if _should_end_now(state):
        return "end"

    invariant_actions = (
        "tool_handler",
        "error_handler",
        "human_review_after_error",
        "human_review_before_run",
        "execute_code",
        "human_review_final",
    )
    for action_name in invariant_actions:
        node = next((nd for nd in NODE_REGISTRY if nd.name == action_name), None)
        if node and action_name in available and node.is_ready(state):
            return action_name

    if get_agent_state(state, "human_review").get("final_decision") == "approve":
        return "end"

    return None


def choose_next_action(state: AgentState, available_actions: Iterable[str]) -> str:
    available = set(available_actions)

    invariant_action = choose_invariant_action(state, available_actions)
    if invariant_action:
        return invariant_action

    intent = _current_intent(state)
    qa_node = next((nd for nd in NODE_REGISTRY if nd.name == "qa"), None)
    generate_node = next((nd for nd in NODE_REGISTRY if nd.name == "generate_code"), None)

    if (
        intent == "qa"
        and "qa" in available
        and qa_node is not None
        and qa_node.is_ready(state)
    ):
        return "qa"
    if (
        intent == "code"
        and "generate_code" in available
        and generate_node is not None
        and generate_node.is_ready(state)
    ):
        return "generate_code"

    # Last-resort deterministic fallback: choose among semantic nodes only.
    for action_name in ("qa", "generate_code"):
        node = next((nd for nd in NODE_REGISTRY if nd.name == action_name), None)
        if node and action_name in available and node.is_ready(state):
            return action_name

    return "end"


def should_prefer_policy_action(
    state: AgentState,
    llm_action: str,
    policy_action: str,
) -> bool:
    """Return True when deterministic routing should override the planner.

    The planner is useful for ambiguous cases, but it should not overrule
    high-confidence intent routing or deterministic system transitions.
    """
    if not policy_action or policy_action == llm_action:
        return False

    invariant_action = choose_invariant_action(state, {llm_action, policy_action})
    if invariant_action == policy_action:
        return True

    intent = _current_intent(state)
    if intent == "qa" and policy_action == "qa":
        return True

    return False
