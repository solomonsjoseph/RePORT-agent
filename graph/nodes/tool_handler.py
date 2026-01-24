from __future__ import annotations

from typing import Any

from ..state import AgentState
from tools.mcp_tools import run_mcp_tool
from .state_helpers import get_agent_state, update_agent_state


def tool_handler_node(state: AgentState) -> AgentState:
    tool_results: list[dict[str, Any]] = []
    agents = dict(state.get("agents", {}))
    for agent_name in list(agents.keys()):
        agent_state = get_agent_state(state, agent_name)
        requests = list(agent_state.get("tool_requests", []))
        if not requests:
            continue
        agent_results: list[dict[str, Any]] = []
        for request in requests:
            tool_name = request.get("tool_name")
            payload = request.get("payload", {})
            if not tool_name:
                continue
            result = {
                "agent": agent_name,
                "tool_name": tool_name,
                "result": run_mcp_tool(tool_name, payload),
            }
            tool_results.append(result)
            agent_results.append(result)
        agent_state["tool_requests"] = []
        agent_state.setdefault("tool_results", []).extend(agent_results)
        agents[agent_name] = agent_state

    if not tool_results:
        return state

    observations = list(state.get("observations", []))
    observations.append(f"tool_handler: executed {len(tool_results)} tools")

    updated_state = {
        **state,
        "agents": agents,
        "observations": observations,
    }
    return update_agent_state(
        updated_state,
        "tool_handler",
        {
            "status": "done",
            "results": tool_results,
        },
    )
