import pytest

pytest.importorskip("mcp")

from tools import mcp_pool


@pytest.mark.asyncio
async def test_call_mcp_tool_retries_once_after_closed_resource(monkeypatch: pytest.MonkeyPatch) -> None:
    class ClosedResourceError(Exception):
        pass

    class FlakyClient:
        def __init__(self) -> None:
            self.calls = 0

        async def call_tool(self, tool_name: str, payload: dict):
            self.calls += 1
            if self.calls == 1:
                raise ClosedResourceError
            return {"ok": True, "tool": tool_name, "payload": payload}

    flaky_client = FlakyClient()
    fresh_client = FlakyClient()
    fresh_client.calls = 1

    get_calls: list[str] = []

    async def fake_get(server_name: str):
        get_calls.append(server_name)
        return flaky_client if len(get_calls) == 1 else fresh_client

    reset_calls: list[str] = []

    async def fake_reset(server_name: str) -> None:
        reset_calls.append(server_name)

    monkeypatch.setattr(mcp_pool, "get_mcp_client", fake_get)
    monkeypatch.setattr(mcp_pool, "reset_mcp_client", fake_reset)

    result = await mcp_pool.call_mcp_tool("weather", "query_weather", {"city": "Boston"})

    assert result == {"ok": True, "tool": "query_weather", "payload": {"city": "Boston"}}
    assert get_calls == ["weather", "weather"]
    assert reset_calls == ["weather"]


@pytest.mark.asyncio
async def test_reset_mcp_client_closes_stack_and_removes_client() -> None:
    class DummyStack:
        def __init__(self) -> None:
            self.closed = False

        async def aclose(self) -> None:
            self.closed = True

    stack = DummyStack()
    mcp_pool._POOL["demo"] = {"client": object(), "stack": stack}

    await mcp_pool.reset_mcp_client("demo")

    assert stack.closed is True
    assert "demo" not in mcp_pool._POOL
