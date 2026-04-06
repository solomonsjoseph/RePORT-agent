from __future__ import annotations

from ...state import MetaKeys
from ...state_views import get_artifacts


def mask_actions(state: dict, available_actions: list[str]) -> tuple[list[str], dict[str, str]]:
    meta = dict(state.get("meta") or {})
    artifacts = get_artifacts(state)
    blocked: dict[str, str] = {}

    if meta.get(MetaKeys.AWAITING_USER_CLARIFICATION) and meta.get(MetaKeys.CLARIFICATION_KIND) == "qa_tool":
        allowed = ["clarification"] if "clarification" in available_actions else []
        for action in available_actions:
            if action != "clarification":
                blocked[action] = "clarification loop active"
        return allowed, blocked

    allowed: list[str] = []
    for action in available_actions:
        if action == "execute_code" and not artifacts.get("generated_code"):
            blocked[action] = "requires generated code"
            continue
        if action == "tool_handler" and not list(meta.get(MetaKeys.TOOL_REQUEST_QUEUE, [])):
            blocked[action] = "requires pending tool requests"
            continue
        allowed.append(action)

    return allowed, blocked
