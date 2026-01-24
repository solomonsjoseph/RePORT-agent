from __future__ import annotations

import asyncio
import os

import pytest

pytest.importorskip("httpx")
pytest.importorskip("dotenv")
pytest.importorskip("mcp")

from tools import search_server, weather_server


def test_weather_server_fetch_weather() -> None:
    result = asyncio.run(weather_server.fetch_weather("Boston"))
    print(f"Weather results: {result}")
    assert isinstance(result, dict)

    if "error" in result:
        assert isinstance(result["error"], str)
    else:
        assert "location" in result
        assert "current" in result


def test_search_server_search() -> None:
    result = asyncio.run(search_server.search("RePORT agent", max_results=1))
    print(f"Search results: {result}")
    assert isinstance(result, dict)

    if os.getenv("TAVILY_API_KEY"):
        assert "error" not in result or result.get("error") == "HTTP error: 401"
    else:
        assert result.get("error") == "Missing TAVILY_API_KEY"