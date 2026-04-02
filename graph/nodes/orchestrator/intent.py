from __future__ import annotations

from ...state import AgentState
from ..state_helpers import get_agent_state
from ..tool_routing import should_route_tools
from .constants import (
    ANALYSIS_CUES,
    CODE_REQUEST_CUES,
    DATA_OPERATION_CUES,
    EXECUTION_CUES,
    INFO_CODE_CUES,
    OWN_DATA_CUES,
    PROTOTYPE_CUES,
)
from .state_logic import _latest_user_message


def infer_intent_from_latest_user(state: AgentState) -> str | None:
    user_message = _latest_user_message(state).lower()
    if not user_message:
        return None

    qa_state = get_agent_state(state, "qa")
    if qa_state.get("awaiting_tool_clarification"):
        return "qa"
    if should_route_tools(user_message):
        return "qa"

    has_code_request = any(token in user_message for token in CODE_REQUEST_CUES)
    has_info_code_request = any(token in user_message for token in INFO_CODE_CUES)
    has_prototype_request = any(token in user_message for token in PROTOTYPE_CUES)
    has_data_context = any(token in user_message for token in DATA_OPERATION_CUES)
    has_own_data = any(token in user_message for token in OWN_DATA_CUES)
    has_analysis_action = any(token in user_message for token in ANALYSIS_CUES)
    has_execution_request = (
        any(token in user_message for token in EXECUTION_CUES)
        or has_data_context
        or has_own_data
    )

    if has_code_request and has_info_code_request and not has_execution_request:
        return "qa"

    if has_prototype_request and not has_own_data and not has_data_context:
        return "qa"

    if has_own_data:
        return "code"

    if has_analysis_action and has_data_context:
        return "code"

    code_score = 0
    if has_code_request:
        code_score += 1
    if has_data_context:
        code_score += 1

    if code_score >= 1:
        return "code"
    return None
