from __future__ import annotations

from typing import Any
from ..state import MetaKeys


def get_agent_state(state: dict, agent_name: str) -> dict[str, Any]:
    agents = dict(state.get("agents", {}))
    agent_state = dict(agents.get(agent_name, {}))
    agent_state.setdefault("status", "idle")
    agent_state.setdefault("run_status", "idle")
    agent_state.setdefault("tool_requests", [])
    agent_state.setdefault("tool_results", [])
    agent_state.setdefault("notes", [])
    agents[agent_name] = agent_state
    state["agents"] = agents
    return agent_state


def update_agent_state(state: dict, agent_name: str, updates: dict[str, Any]) -> dict[str, Any]:
    agent_state = get_agent_state(state, agent_name)
    agent_state.update(updates)
    agents = dict(state.get("agents", {}))
    agents[agent_name] = agent_state
    return {
        **state,
        "agents": agents,
    }


def enqueue_tool_requester(state: dict, agent_name: str) -> dict[str, Any]:
    meta = dict(state.get("meta", {}))
    queue = list(meta.get("tool_request_queue", []))
    if agent_name not in queue:
        queue.append(agent_name)
    meta["tool_request_queue"] = queue
    return {
        **state,
        "meta": meta,
    }


def clear_clarification_meta(meta: dict[str, Any] | None) -> dict[str, Any]:
    updated = dict(meta or {})
    updated.pop(MetaKeys.AWAITING_USER_CLARIFICATION, None)
    updated.pop(MetaKeys.PENDING_QUESTION, None)
    updated.pop(MetaKeys.CLARIFICATION_RETURN_NODE, None)
    updated.pop(MetaKeys.CLARIFICATION_KIND, None)
    updated.pop(MetaKeys.SEMANTIC_LAST_ACTION, None)
    return updated


def set_clarification_meta(
    meta: dict[str, Any] | None,
    *,
    return_node: str,
    kind: str,
    pending_question: str | None = None,
) -> dict[str, Any]:
    updated = dict(meta or {})
    updated[MetaKeys.AWAITING_USER_CLARIFICATION] = True
    updated[MetaKeys.CLARIFICATION_RETURN_NODE] = return_node
    updated[MetaKeys.CLARIFICATION_KIND] = kind
    if pending_question is not None:
        updated[MetaKeys.PENDING_QUESTION] = pending_question
    return updated
