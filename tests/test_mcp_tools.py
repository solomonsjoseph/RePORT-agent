import json
from pathlib import Path

import pytest

from tools import mcp_tools


def test_load_servers_config_returns_default_when_missing(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"

    result = mcp_tools.load_servers_config(str(missing_path))

    assert result == {"mcpServers": {}}


def test_load_servers_config_reads_file(tmp_path: Path) -> None:
    config = {"mcpServers": {"demo": {"command": "python"}}}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    result = mcp_tools.load_servers_config(str(config_path))

    assert result == {
        "mcpServers": {
            "demo": {
                "command": mcp_tools.sys.executable,
            }
        }
    }


def test_make_mcp_tool_requires_server_name() -> None:
    result = mcp_tools.make_mcp_tool("demo", {})

    assert result == {
        "status": "error",
        "message": "Missing MCP server name in payload.",
    }


def test_make_mcp_tool_reports_unknown_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_tools, "load_servers_config", lambda: {"mcpServers": {}})

    result = mcp_tools.make_mcp_tool("demo", {"server": "unknown"})

    assert result == {
        "status": "error",
        "message": "Unknown MCP server: unknown",
    }


def test_make_mcp_tool_returns_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        mcp_tools,
        "load_servers_config",
        lambda: {"mcpServers": {"demo": {"command": "python"}}},
    )

    payload = {"server": "demo", "query": "hello"}
    result = mcp_tools.make_mcp_tool("do", payload)

    assert result == {
        "status": "queued",
        "server": "demo",
        "server_config": {"command": "python"},
        "tool_name": "do",
        "payload": payload,
    }
