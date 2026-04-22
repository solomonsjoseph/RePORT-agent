from __future__ import annotations

from ...state import MetaKeys
from ..node_registry import NODE_REGISTRY_MAP
from .policy import DETERMINISTIC_CONTROL_ACTIONS


def _static_reason(text: str):
    return lambda _state: text


def _execute_code_block_reason(state: dict) -> str:
    final_review = NODE_REGISTRY_MAP.get("human_review_before_output")
    before_run = NODE_REGISTRY_MAP.get("human_review_before_run")
    if final_review and final_review.is_ready(state):
        return "already succeeded; move to human_review_before_output"
    if before_run and before_run.is_ready(state):
        return "requires generated code awaiting run approval"
    return "requires approved generated code ready to run"


CONTROL_ACTION_REASON_FACTORIES = {
    "tool_handler": _static_reason("requires active tool request"),
    "clarification": _static_reason("requires active clarification loop"),
    "error_handler": _static_reason("requires retryable execution error"),
    "terminal_execution_error": _static_reason("requires terminal execution error"),
    "human_review_after_error": _static_reason("requires exhausted retryable execution error awaiting review"),
    "human_review_before_run": _static_reason("requires generated code awaiting run approval"),
    "execute_code": _execute_code_block_reason,
    "human_review_before_output": _static_reason(
        "requires successful execution awaiting final review"
    ),
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
        if action == "clarification" and node and not node.is_ready(state):
            blocked[action] = CONTROL_ACTION_REASON_FACTORIES[action](state)
            continue
        if action in DETERMINISTIC_CONTROL_ACTIONS and node and not node.is_ready(state):
            blocked[action] = CONTROL_ACTION_REASON_FACTORIES[action](state)
            continue
        allowed.append(action)

    return allowed, blocked
