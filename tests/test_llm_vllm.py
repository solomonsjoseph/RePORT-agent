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


def test_build_llm_passes_sampling_params_to_openai_gpt41_models(monkeypatch) -> None:
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SEC", "21")

    captured: dict[str, object] = {}

    class _ChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    openai_mod = ModuleType("langchain_openai")
    openai_mod.ChatOpenAI = _ChatOpenAI
    sys.modules["langchain_openai"] = openai_mod
    sys.modules["requests"] = ModuleType("requests")

    sys.modules.pop("llm_vllm", None)
    llm_vllm = importlib.import_module("llm_vllm")

    llm_vllm.build_llm(
        model_name="gpt-4.1-mini-2025-04-14",
        temperature=0.3,
        top_p=0.7,
        base_url="",
        api_key="secret-openai",
        provider="openai",
    )

    assert captured["model"] == "gpt-4.1-mini-2025-04-14"
    assert captured["temperature"] == 0.3
    assert captured["top_p"] == 0.7
    assert captured["max_tokens"] == 4096
    assert captured["timeout"] == 21.0
    assert captured["max_retries"] == 0
    assert os.environ["OPENAI_API_KEY"] == "secret-openai"


def test_build_llm_omits_sampling_params_for_openai_gpt5_models(monkeypatch) -> None:
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SEC", "19")

    captured: dict[str, object] = {}

    class _ChatOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    openai_mod = ModuleType("langchain_openai")
    openai_mod.ChatOpenAI = _ChatOpenAI
    sys.modules["langchain_openai"] = openai_mod
    sys.modules["requests"] = ModuleType("requests")

    sys.modules.pop("llm_vllm", None)
    llm_vllm = importlib.import_module("llm_vllm")

    llm_vllm.build_llm(
        model_name="gpt-5",
        temperature=0.3,
        top_p=0.7,
        base_url="",
        api_key="secret-openai",
        provider="openai",
    )

    assert captured["model"] == "gpt-5"
    assert captured["timeout"] == 19.0
    assert captured["max_retries"] == 0
    assert captured["max_completion_tokens"] == 4096
    assert "temperature" not in captured
    assert "top_p" not in captured
    assert "max_tokens" not in captured
