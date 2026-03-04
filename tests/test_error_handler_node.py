from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace


class _Prompt:
    def invoke(self, payload):
        return payload


class _LLM:
    def __init__(self, content: str):
        self.content = content

    def invoke(self, _prompt):
        return SimpleNamespace(content=self.content)


def _install_stubs() -> None:
    messages_mod = ModuleType("langchain_core.messages")
    messages_mod.BaseMessage = object

    fix_mod = ModuleType("prompts.fix_prompt")
    fix_mod.make_fix_code_prompt = lambda: _Prompt()

    sys.modules["langchain_core.messages"] = messages_mod
    sys.modules["prompts.fix_prompt"] = fix_mod


def test_error_handler_non_code_response_sets_clarification_wait() -> None:
    _install_stubs()
    sys.modules.pop("graph.nodes.error_handler", None)
    mod = importlib.import_module("graph.nodes.error_handler")

    state = {
        "messages": [],
        "output": {"generated_code": "print('x')", "error": {"type": "SyntaxError", "message": "bad"}},
        "meta": {"error_iterations": 0},
        "agents": {"human_review": {}, "executor": {}},
    }

    updated = mod.error_handler_node(state, _LLM("Please provide schema columns first."), context="ctx")

    assert updated["output"]["generated_code"] == ""
    assert updated["meta"]["awaiting_user_clarification"] is True
    assert updated["agents"]["executor"]["run_status"] == "idle"
