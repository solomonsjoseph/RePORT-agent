from __future__ import annotations

import importlib
import os
import sys
from types import ModuleType


def test_build_llm_passes_timeout_and_retries_to_anthropic(monkeypatch) -> None:
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SEC", "17")

    captured: dict[str, object] = {}

    class _ChatAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    anthropic_mod = ModuleType("langchain_anthropic")
    anthropic_mod.ChatAnthropic = _ChatAnthropic
    sys.modules["langchain_anthropic"] = anthropic_mod
    sys.modules["requests"] = ModuleType("requests")

    sys.modules.pop("llm_vllm", None)
    llm_vllm = importlib.import_module("llm_vllm")

    llm_vllm.build_llm(
        model_name="claude-test",
        temperature=0.2,
        top_p=0.8,
        base_url="",
        api_key="secret",
        provider="anthropic",
    )

    assert captured["model"] == "claude-test"
    assert captured["temperature"] == 0.2
    assert captured["max_tokens"] == 4096
    assert captured["timeout"] == 17.0
    assert captured["max_retries"] == 0
    assert "top_p" not in captured
    assert os.environ["ANTHROPIC_API_KEY"] == "secret"
