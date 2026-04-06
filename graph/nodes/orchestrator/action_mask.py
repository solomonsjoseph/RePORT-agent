from __future__ import annotations

from ...state import MetaKeys
from ..node_registry import NODE_REGISTRY_MAP


CONTROL_ACTION_REASONS = {
    "tool_handler": "requires active tool request",
    "clarification": "requires active clarification loop",
    "error_handler": "requires retryable execution error",
    "terminal_execution_error": "requires terminal execution error",
    "human_review_after_error": "requires exhausted retryable execution error awaiting review",
    "human_review_before_run": "requires generated code awaiting run approval",
    "execute_code": "requires approved generated code ready to run",
    "human_review_final": "requires successful execution awaiting final review",
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
        node = NODE_REGISTRY_MAP.get(action)
        if action in CONTROL_ACTION_REASONS and node and not node.is_ready(state):
            blocked[action] = CONTROL_ACTION_REASONS[action]
            continue
        allowed.append(action)

    return allowed, blocked
