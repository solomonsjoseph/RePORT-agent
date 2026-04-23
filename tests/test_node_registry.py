from __future__ import annotations

import importlib
import sys
from types import ModuleType


def _load_registry_module():
    messages_mod = ModuleType("langchain_core.messages")
    messages_mod.BaseMessage = object
    sys.modules["langchain_core.messages"] = messages_mod
    prompts_mod = ModuleType("langchain_core.prompts")
    prompts_mod.ChatPromptTemplate = object
    prompts_mod.MessagesPlaceholder = object
    sys.modules["langchain_core.prompts"] = prompts_mod

    graph_message_mod = ModuleType("langgraph.graph.message")
    graph_message_mod.add_messages = lambda current, new: (current or []) + (new or [])
    sys.modules["langgraph.graph.message"] = graph_message_mod

    sys.modules.pop("graph.state", None)
    sys.modules.pop("graph.nodes.node_registry", None)
    return importlib.import_module("graph.nodes.node_registry")


def _state_with_error(category: str, *, error_iterations: int = 0):
    return {
        "output": {"generated_code": "print(1)", "error": {"category": category, "type": "X", "message": "msg"}},
        "agents": {
            "executor": {"run_status": "error"},
            "human_review": {"after_error_decision": None},
        },
        "meta": {"error_iterations": error_iterations},
    }


def test_error_handler_ready_only_for_retryable_errors() -> None:
    mod = _load_registry_module()
    error_handler = mod.NODE_REGISTRY_MAP["error_handler"]

    assert error_handler.is_ready(_state_with_error("retryable_code")) is True
    assert error_handler.is_ready(_state_with_error("policy_blocked")) is False


def test_terminal_execution_error_ready_for_terminal_error_immediately() -> None:
    mod = _load_registry_module()
    review = mod.NODE_REGISTRY_MAP["terminal_execution_error"]

    assert review.is_ready(_state_with_error("timeout")) is True
    assert review.is_ready(_state_with_error("infrastructure")) is True
    assert review.is_ready(_state_with_error("unsupported_runtime")) is True


def test_human_review_after_error_ready_after_retry_budget_exhausted() -> None:
    mod = _load_registry_module()
    review = mod.NODE_REGISTRY_MAP["human_review_after_error"]

    assert review.is_ready(_state_with_error("retryable_code", error_iterations=mod.MAX_ERROR_ITERATIONS)) is True
    assert review.is_ready(_state_with_error("retryable_code", error_iterations=1)) is False


def test_human_review_after_error_not_ready_for_terminal_error() -> None:
    mod = _load_registry_module()
    review = mod.NODE_REGISTRY_MAP["human_review_after_error"]

    assert review.is_ready(_state_with_error("policy_blocked")) is False


def test_human_review_rag_db_column_selection_ready_when_selection_waits_for_review() -> None:
    mod = _load_registry_module()
    review = mod.NODE_REGISTRY_MAP["human_review_rag_db_column_selection"]
    state = {
        "agents": {
            "rag_db_qa": {
                "pending_column_review": {
                    "status": "awaiting_review",
                }
            }
        }
    }

    assert review.is_ready(state) is True


def test_action_capabilities_are_sourced_from_node_modules() -> None:
    from graph.nodes.action_metadata import ACTION_CAPABILITIES

    assert ACTION_CAPABILITIES["qa"].startswith("Handle direct user-facing Q&A")
    assert ACTION_CAPABILITIES["clarification"].startswith("Resume an active clarification loop")
