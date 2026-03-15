from __future__ import annotations

from typing import Iterable

from ...state import AgentState, MetaKeys
from ..state_helpers import get_agent_state
from ..tool_routing import is_tool_requested
from ..node_registry import NODE_REGISTRY
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


def choose_next_action(state: AgentState, available_actions: Iterable[str]) -> str:
    intent = (
        (state.get("meta") or {}).get(MetaKeys.INTENT)
        or infer_intent_from_latest_user(state)
        or ""
    ).strip()
    available = set(available_actions)

    if _should_end_now(state):
        return "end"

    if intent == "qa" and "qa" in available and not is_tool_requested(state):
        return "qa"

    sorted_nodes = sorted(NODE_REGISTRY, key=lambda nd: nd.priority)
    for nd in sorted_nodes:
        if nd.name in available and nd.is_ready(state):
            return nd.name

    if get_agent_state(state, "human_review").get("final_decision") == "approve":
        return "end"

    return "end"
