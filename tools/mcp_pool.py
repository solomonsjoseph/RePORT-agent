import asyncio
from mcp.client.stdio import stdio_client
from tools.mcp_tools import get_server_config

_POOL = {}
_LOCK = asyncio.Lock()


async def get_mcp_client(server_name: str):
    async with _LOCK:
        if server_name in _POOL:
            return _POOL[server_name]

        cfg = get_server_config(server_name)

        cmd = [cfg["command"]] + cfg.get("args", [])

        client = await stdio_client(cmd).__aenter__()

        _POOL[server_name] = client
        return client


async def call_mcp_tool(server_name: str, tool_name: str, payload: dict):
    client = await get_mcp_client(server_name)
    return await client.call_tool(tool_name, payload)