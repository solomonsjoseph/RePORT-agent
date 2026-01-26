from __future__ import annotations

from typing import Any


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
