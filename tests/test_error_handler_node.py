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


class _AIMessage:
    type = "ai"
    def __init__(self, content: str = ""):
        self.content = content


def _install_stubs() -> None:
    messages_mod = ModuleType("langchain_core.messages")
    messages_mod.BaseMessage = object
    messages_mod.AIMessage = _AIMessage

    fix_mod = ModuleType("prompts.fix_prompt")
    fix_mod.make_fix_code_prompt = lambda: _Prompt()

    sys.modules["langchain_core.messages"] = messages_mod
    sys.modules["prompts.fix_prompt"] = fix_mod


def test_error_handler_non_code_response_sets_clarification_wait() -> None:
    _install_stubs()
    for _mod in ("utils.message_window", "graph.nodes.error_handler"):
        sys.modules.pop(_mod, None)
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


def test_error_handler_preserves_approval_during_retry_loop() -> None:
    _install_stubs()
    for _mod in ("utils.message_window", "graph.nodes.error_handler"):
        sys.modules.pop(_mod, None)
    mod = importlib.import_module("graph.nodes.error_handler")

    state = {
        "messages": [],
        "output": {"generated_code": "print('x')", "error": {"type": "NameError", "message": "bad"}},
        "meta": {"error_iterations": 1, "current_code_hash": "old"},
        "agents": {
            "human_review": {"before_run_decision": "approve", "approved_code_hash": "old"},
            "executor": {"run_status": "error"},
        },
    }

    updated = mod.error_handler_node(state, _LLM("```python\nprint('fixed')\n```"), context="ctx")

    review = updated["agents"]["human_review"]
    assert review["before_run_decision"] == "approve"
    assert review["approved_code_hash"] == updated["meta"]["current_code_hash"]
    assert updated["agents"]["executor"]["run_status"] == "pending"
