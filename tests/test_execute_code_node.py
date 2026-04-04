from __future__ import annotations

import importlib
import sys
from types import ModuleType


def _load_execute_code_module():
    messages_mod = ModuleType("langchain_core.messages")
    messages_mod.BaseMessage = object
    sys.modules["langchain_core.messages"] = messages_mod
    graph_message_mod = ModuleType("langgraph.graph.message")
    graph_message_mod.add_messages = lambda current, new: (current or []) + (new or [])
    sys.modules["langgraph.graph.message"] = graph_message_mod

    lifelines_mod = ModuleType("lifelines")
    lifelines_mod.KaplanMeierFitter = object
    lifelines_mod.CoxPHFitter = object
    sys.modules["lifelines"] = lifelines_mod
    sys.modules.pop("tools.execution", None)
    sys.modules.pop("graph.nodes.execute_code", None)
    return importlib.import_module("graph.nodes.execute_code")


def test_execute_code_node_preserves_structured_sandbox_errors(monkeypatch) -> None:
    execute_code = _load_execute_code_module()

    def fake_run_python_user(_code, _df):
        return None, "", b"", {
            "category": "policy_blocked",
            "type": "PolicyBlockedError",
            "message": "Disallowed import: subprocess",
        }

    monkeypatch.setattr(execute_code, "run_python_user", fake_run_python_user)

    state = {
        "output": {"generated_code": "import subprocess"},
        "agents": {"executor": {"run_status": "pending"}},
    }

    updated = execute_code.execute_code_node(state, df=None)

    assert updated["output"]["error"] == {
        "category": "policy_blocked",
        "type": "PolicyBlockedError",
        "message": "Disallowed import: subprocess",
    }
    assert updated["agents"]["executor"]["run_status"] == "error"
