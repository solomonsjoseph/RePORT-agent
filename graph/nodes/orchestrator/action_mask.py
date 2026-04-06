from __future__ import annotations

from ...state import MetaKeys
from ...state_views import get_artifacts


def _tool_request_active(state: dict) -> bool:
    agents = dict(state.get("agents") or {})
    return any(agent_state.get("tool_requests") for agent_state in agents.values())


def _clarification_loop_active(state: dict) -> bool:
    meta = dict(state.get("meta") or {})
    return bool(meta.get(MetaKeys.AWAITING_USER_CLARIFICATION))


def _generated_code_present(state: dict) -> bool:
    return bool(get_artifacts(state).get("generated_code"))


CONTROL_ACTION_GATES = {
    "tool_handler": (_tool_request_active, "requires active tool request"),
    "clarification": (_clarification_loop_active, "requires active clarification loop"),
    "execute_code": (_generated_code_present, "requires generated code"),
}


def mask_actions(state: dict, available_actions: list[str]) -> tuple[list[str], dict[str, str]]:
    meta = dict(state.get("meta") or {})
    blocked: dict[str, str] = {}

    if meta.get(MetaKeys.AWAITING_USER_CLARIFICATION) and meta.get(MetaKeys.CLARIFICATION_KIND) == "qa_tool":
        allowed = ["clarification"] if "clarification" in available_actions else []
        for action in available_actions:
            if action != "clarification":
                blocked[action] = "clarification loop active"
        return allowed, blocked

    allowed: list[str] = []
    for action in available_actions:
        gate = CONTROL_ACTION_GATES.get(action)
        if gate and not gate[0](state):
            blocked[action] = gate[1]
            continue
        allowed.append(action)

    return allowed, blocked
