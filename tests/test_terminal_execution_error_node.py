from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace


def _install_stubs() -> None:
    messages_mod = ModuleType("langchain_core.messages")

    class _AIMessage:
        def __init__(self, content: str, additional_kwargs: dict | None = None):
            self.type = "ai"
            self.content = content
            self.additional_kwargs = additional_kwargs or {}

    messages_mod.AIMessage = _AIMessage
    messages_mod.BaseMessage = SimpleNamespace

    sys.modules["langchain_core.messages"] = messages_mod


def test_terminal_execution_error_node_emits_detailed_policy_message() -> None:
    _install_stubs()
    sys.modules.pop("graph.nodes.terminal_execution_error", None)
    mod = importlib.import_module("graph.nodes.terminal_execution_error")

    state = {
        "messages": [],
        "output": {
            "error": {
                "category": "policy_blocked",
                "type": "PolicyBlockedError",
                "message": "Disallowed import: subprocess",
            }
        },
        "agents": {},
    }

    updated = mod.terminal_execution_error_node(state)

    assert updated["messages"][-1].type == "ai"
    assert "sandbox stopped this request" in updated["messages"][-1].content.lower()
    assert "subprocess" in updated["messages"][-1].content
    assert updated["output"]["qa_response"] == updated["messages"][-1].content
    assert updated["agents"]["terminal_execution_error"]["status"] == "done"


def test_terminal_execution_error_node_explains_missing_package() -> None:
    _install_stubs()
    sys.modules.pop("graph.nodes.terminal_execution_error", None)
    mod = importlib.import_module("graph.nodes.terminal_execution_error")

    state = {
        "messages": [],
        "output": {
            "error": {
                "category": "unsupported_runtime",
                "type": "DependencyNotAvailableError",
                "message": "Sandbox image does not include package: statsmodels",
            }
        },
        "agents": {},
    }

    updated = mod.terminal_execution_error_node(state)

    assert "statsmodels" in updated["messages"][-1].content
    assert "runtime image" in updated["messages"][-1].content.lower()
