from __future__ import annotations

import sys
from types import ModuleType


def _install_langchain_and_langgraph_stubs() -> None:
    messages_mod = ModuleType("langchain_core.messages")
    messages_mod.BaseMessage = object

    langchain_core_mod = ModuleType("langchain_core")
    langchain_core_mod.messages = messages_mod

    graph_message_mod = ModuleType("langgraph.graph.message")
    graph_message_mod.add_messages = lambda current, new: (current or []) + (new or [])

    sys.modules["langchain_core"] = langchain_core_mod
    sys.modules["langchain_core.messages"] = messages_mod
    sys.modules["langgraph.graph.message"] = graph_message_mod


_install_langchain_and_langgraph_stubs()

from graph.nodes.orchestrator.action_mask import mask_actions


def test_mask_actions_blocks_execute_without_generated_code() -> None:
    state = {"artifacts": {"generated_code": ""}, "node_data": {"executor": {"run_status": "idle"}}}
    allowed, blocked = mask_actions(state, ["qa", "generate_code", "execute_code"])

    assert "execute_code" not in allowed
    assert blocked["execute_code"] == "requires generated code"


def test_mask_actions_forces_clarification_resume_when_tool_clarification_is_pending() -> None:
    state = {
        "meta": {
            "awaiting_user_clarification": True,
            "clarification_kind": "qa_tool",
        }
    }
    allowed, blocked = mask_actions(state, ["qa", "clarification", "tool_handler"])

    assert allowed == ["clarification"]
    assert blocked["qa"] == "clarification loop active"
