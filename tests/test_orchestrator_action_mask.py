from __future__ import annotations

import sys
from types import ModuleType


def _install_langchain_and_langgraph_stubs() -> None:
    class _MessagesPlaceholder:
        def __init__(self, variable_name: str, optional: bool = False) -> None:
            self.variable_name = variable_name
            self.optional = optional

    class _ChatPromptTemplate:
        @staticmethod
        def from_messages(messages):
            return messages

    messages_mod = ModuleType("langchain_core.messages")
    messages_mod.BaseMessage = object
    prompts_mod = ModuleType("langchain_core.prompts")
    prompts_mod.ChatPromptTemplate = _ChatPromptTemplate
    prompts_mod.MessagesPlaceholder = _MessagesPlaceholder

    langchain_core_mod = ModuleType("langchain_core")
    langchain_core_mod.messages = messages_mod
    langchain_core_mod.prompts = prompts_mod

    graph_message_mod = ModuleType("langgraph.graph.message")
    graph_message_mod.add_messages = lambda current, new: (current or []) + (new or [])

    sys.modules["langchain_core"] = langchain_core_mod
    sys.modules["langchain_core.messages"] = messages_mod
    sys.modules["langchain_core.prompts"] = prompts_mod
    sys.modules["langgraph.graph.message"] = graph_message_mod


_install_langchain_and_langgraph_stubs()

from graph.nodes.orchestrator.action_mask import mask_actions


def test_mask_actions_blocks_execute_without_generated_code() -> None:
    state = {"artifacts": {"generated_code": ""}, "node_data": {"executor": {"run_status": "idle"}}}
    allowed, blocked = mask_actions(state, ["qa", "generate_code", "execute_code"])

    assert "execute_code" not in allowed
    assert blocked["execute_code"] == "requires approved generated code ready to run"


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


def test_mask_actions_blocks_inactive_control_actions_without_preconditions() -> None:
    state = {
        "artifacts": {},
        "agents": {
            "qa": {},
            "executor": {"run_status": "idle"},
        },
        "meta": {},
    }

    allowed, blocked = mask_actions(state, ["qa", "tool_handler", "clarification"])

    assert allowed == ["qa"]
    assert blocked["tool_handler"] == "requires active tool request"
    assert blocked["clarification"] == "requires active clarification loop"


def test_mask_actions_blocks_inactive_review_control_actions_without_preconditions() -> None:
    state = {
        "artifacts": {},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"final_decision": None},
        },
        "meta": {},
    }

    allowed, blocked = mask_actions(state, ["qa", "human_review_before_output"])

    assert allowed == ["qa"]
    assert blocked["human_review_before_output"] == "requires successful execution awaiting final review"


def test_mask_actions_explain_execute_code_block_when_final_review_is_pending() -> None:
    state = {
        "artifacts": {"generated_code": "print(1)"},
        "agents": {
            "executor": {"run_status": "ok"},
            "human_review": {"final_decision": None},
        },
        "meta": {"current_code_hash": "h1"},
    }

    allowed, blocked = mask_actions(state, ["qa", "execute_code", "human_review_before_output"])

    assert allowed == ["qa", "human_review_before_output"]
    assert blocked["execute_code"] == "already succeeded; move to human_review_before_output"


def test_action_mask_blocks_inactive_human_review_rag_db_column_selection() -> None:
    state = {
        "agents": {
            "rag_db_qa": {},
        },
        "meta": {},
    }

    allowed, blocked = mask_actions(state, ["qa", "human_review_rag_db_column_selection"])

    assert allowed == ["qa"]
    assert blocked["human_review_rag_db_column_selection"] == (
        "requires DB-RAG column selection awaiting review"
    )
