from __future__ import annotations

import sys
from types import ModuleType

langchain_core_mod = ModuleType("langchain_core")
messages_mod = ModuleType("langchain_core.messages")
messages_mod.BaseMessage = object
langchain_core_mod.messages = messages_mod
sys.modules["langchain_core"] = langchain_core_mod

sys.modules["langchain_core.messages"] = messages_mod

langgraph_mod = ModuleType("langgraph")
langgraph_graph_mod = ModuleType("langgraph.graph")
graph_message_mod = ModuleType("langgraph.graph.message")
graph_message_mod.add_messages = lambda current, new: (current or []) + (new or [])
langgraph_graph_mod.message = graph_message_mod
langgraph_mod.graph = langgraph_graph_mod
sys.modules["langgraph"] = langgraph_mod
sys.modules["langgraph.graph"] = langgraph_graph_mod
sys.modules["langgraph.graph.message"] = graph_message_mod

from graph.nodes.orchestrator.workflow_status import derive_workflow_status


def test_direct_qa_answer_is_classified_as_answered_complete() -> None:
    state = {
        "messages": [],
        "output": {"qa_response": "I am a data-analysis assistant."},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "after_error_decision": None, "final_decision": None},
        },
        "meta": {"workflow_trace": ["orchestrator", "qa"]},
        "last_action": "qa",
    }

    status = derive_workflow_status(state)

    assert status["milestone"] == "answered"
    assert status["completion_status"] == "complete"
    assert status["blocker_signature"] is None


def test_generated_code_without_ticket_is_awaiting_run_review() -> None:
    state = {
        "output": {"generated_code": "print(1)"},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "final_decision": None},
        },
        "meta": {"current_code_hash": "h1"},
        "last_action": "generate_code",
    }

    status = derive_workflow_status(state)

    assert status["milestone"] == "awaiting_run_review"
    assert status["completion_status"] == "blocked_waiting"
    assert status["blocker_signature"] == "waiting_for_before_run_review"


def test_retryable_error_with_budget_remaining_is_retrying_after_error() -> None:
    state = {
        "output": {"error": {"category": "retryable_code", "type": "NameError", "message": "name x is not defined"}},
        "agents": {"executor": {"run_status": "error"}},
        "meta": {"error_iterations": 1},
        "last_action": "execute_code",
    }

    status = derive_workflow_status(state)

    assert status["milestone"] == "retrying_after_error"
    assert status["completion_status"] == "incomplete"
    assert status["blocker_signature"].startswith("retryable_error:NameError:")


def test_terminal_error_is_classified_as_complete_terminal_error() -> None:
    state = {
        "output": {"generated_code": "print(1)", "error": {"category": "timeout", "type": "TimeoutError", "message": "timed out"}},
        "agents": {"executor": {"run_status": "error"}},
        "meta": {"current_code_hash": "h1"},
        "last_action": "execute_code",
    }

    status = derive_workflow_status(state)

    assert status["milestone"] == "terminal_error"
    assert status["completion_status"] == "complete"
    assert status["blocker_signature"] == "terminal_error:timeout"
