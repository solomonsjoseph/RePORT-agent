from __future__ import annotations

import sys
from types import ModuleType
from types import SimpleNamespace

langchain_core_mod = ModuleType("langchain_core")
messages_mod = ModuleType("langchain_core.messages")
prompts_mod = ModuleType("langchain_core.prompts")
messages_mod.BaseMessage = object
prompts_mod.ChatPromptTemplate = object
prompts_mod.MessagesPlaceholder = object
langchain_core_mod.messages = messages_mod
langchain_core_mod.prompts = prompts_mod
sys.modules["langchain_core"] = langchain_core_mod

sys.modules["langchain_core.messages"] = messages_mod
sys.modules["langchain_core.prompts"] = prompts_mod

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


def test_stale_qa_response_with_new_unanswered_human_message_is_not_complete() -> None:
    state = {
        "messages": [
            SimpleNamespace(type="human", content="who are you"),
            SimpleNamespace(type="ai", content="I am a data-analysis assistant."),
            SimpleNamespace(type="human", content="what's the weather today?"),
        ],
        "output": {"qa_response": "I am a data-analysis assistant."},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "final_decision": None},
        },
        "meta": {"workflow_trace": ["orchestrator", "qa", "orchestrator"]},
        "last_action": "qa",
    }

    status = derive_workflow_status(state)

    assert status["milestone"] == "needs_code"
    assert status["completion_status"] == "incomplete"
    assert status["blocker_signature"] == "missing_next_step"


def test_active_retry_recovery_with_ticket_does_not_fall_through_to_ready_to_execute() -> None:
    state = {
        "output": {
            "generated_code": "print(2)",
            "error": {"category": "retryable_code", "type": "NameError", "message": "name x is not defined"},
        },
        "agents": {"executor": {"run_status": "idle"}},
        "meta": {
            "current_code_hash": "h2",
            "execution_ticket_hash": "h2",
            "error_recovery_active": True,
            "error_iterations": 1,
        },
        "last_action": "error_handler",
    }

    status = derive_workflow_status(state)

    assert status["milestone"] == "retrying_after_error"
    assert status["completion_status"] == "incomplete"
    assert status["blocker_signature"].startswith("retryable_error:NameError:")


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


def test_exhausted_retry_review_with_decision_is_not_blocked_waiting() -> None:
    state = {
        "output": {"error": {"category": "retryable_code", "type": "NameError", "message": "name x is not defined"}},
        "agents": {
            "executor": {"run_status": "error"},
            "human_review": {"after_error_decision": "regenerate"},
        },
        "meta": {"error_iterations": 5},
        "last_action": "execute_code",
    }

    status = derive_workflow_status(state)

    assert status["milestone"] == "retrying_after_error"
    assert status["completion_status"] == "incomplete"
    assert status["blocker_signature"].startswith("retryable_error:NameError:")


def test_terminal_error_is_complete_after_terminal_response_node_runs() -> None:
    state = {
        "output": {
            "generated_code": "print(1)",
            "error": {"category": "timeout", "type": "TimeoutError", "message": "timed out"},
            "qa_response": "This analysis ran longer than the allowed execution time.",
        },
        "agents": {"executor": {"run_status": "error"}},
        "meta": {"current_code_hash": "h1"},
        "last_action": "terminal_execution_error",
    }

    status = derive_workflow_status(state)

    assert status["milestone"] == "terminal_error"
    assert status["completion_status"] == "complete"
    assert status["blocker_signature"] == "terminal_error:timeout"


def test_successful_execution_with_approved_final_review_is_not_awaiting_final_review() -> None:
    state = {
        "output": {"generated_code": "print(1)", "text": "ok"},
        "agents": {
            "executor": {"run_status": "ok"},
            "human_review": {"final_decision": None},
        },
        "meta": {"current_code_hash": "h1", "final_approved_code_hash": "h1"},
        "last_action": "human_review_before_output",
    }

    status = derive_workflow_status(state)

    assert status["milestone"] == "analysis_complete"
    assert status["completion_status"] == "complete"
    assert status["blocker_signature"] is None
