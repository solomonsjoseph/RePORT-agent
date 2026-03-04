from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_servers_config(config_path: str | None = None) -> dict[str, Any]:
    if config_path is None:
        config_path = str(Path(__file__).resolve().parent / "servers_config.json")
    path = Path(config_path)
    if not path.exists():
        return {"mcpServers": {}}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_server_config(server_name: str) -> dict[str, Any] | None:
    config = load_servers_config()
    return config.get("mcpServers", {}).get(server_name)


def make_mcp_tool(tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    server_name = payload.get("server")
    if not server_name:
        return {
            "status": "error",
            "message": "Missing MCP server name in payload.",
        }
    server_config = get_server_config(server_name)
    if not server_config:
        return {"status": "error", "message": f"Unknown MCP server: {server_name}"}
    return {
        "status": "queued",
        "server": server_name,
        "server_config": server_config,
        "tool_name": tool_name,
        "payload": payload,
    }