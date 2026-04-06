from __future__ import annotations

import traceback
from typing import Any
from tools.mcp_pool import call_mcp_tool_sync

from ..state import AgentState
from tools.mcp_tools import make_mcp_tool
from .state_helpers import get_agent_state, update_agent_state

NODE_NAME = "tool_handler"
NODE_CAPABILITY = (
    "Execute already-requested external tools and store results back to the requesting agent. "
    "Do not use for routing decisions or direct user replies."
)


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

            result = make_mcp_tool(tool_name, payload)

            # If MCP tool is queued → execute it
            if result.get("status") == "queued":
                server_name = result["server"]

                try:
                    output = call_mcp_tool_sync(server_name, tool_name, payload)

                    result = {
                        "status": "done",
                        "server": server_name,
                        "tool_name": tool_name,
                        "payload": payload,
                        "output": output,
                    }
                except Exception as e:
                    error_message = str(e).strip() or f"{type(e).__name__} (empty error message)"
                    result = {
                        "status": "error",
                        "server": server_name,
                        "tool_name": tool_name,
                        "payload": payload,
                        "message": error_message,
                        "error_type": type(e).__name__,
                        "traceback": traceback.format_exc(),
                    }

            tool_result = {
                "agent": agent_name,
                "tool_name": tool_name,
                "result": result,
            }

            tool_results.append(tool_result)
            agent_results.append(tool_result)

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
