from __future__ import annotations

from ...state import AgentState, MetaKeys
from ..state_helpers import get_agent_state

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
