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


def test_direct_rag_db_answer_is_classified_as_answered_complete() -> None:
    state = {
        "messages": [],
        "output": {"qa_response": "DB-RAG assets are not initialized."},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "after_error_decision": None, "final_decision": None},
        },
        "meta": {"workflow_trace": ["orchestrator", "rag_db_qa"]},
        "last_action": "rag_db_qa",
    }

    status = derive_workflow_status(state)

    assert status["milestone"] == "answered"
    assert status["completion_status"] == "complete"
    assert status["blocker_signature"] is None


def test_clarification_resumed_qa_answer_is_classified_as_answered_complete() -> None:
    state = {
        "messages": [],
        "output": {"qa_response": "Boston weather today is cool and rainy."},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "after_error_decision": None, "final_decision": None},
        },
        "meta": {
            "workflow_trace": ["orchestrator", "qa", "orchestrator", "clarification"],
            "semantic_last_action": "qa",
        },
        "last_action": "clarification",
    }

    status = derive_workflow_status(state)

    assert status["milestone"] == "answered"
    assert status["completion_status"] == "complete"
    assert status["blocker_signature"] is None


def test_pending_rag_db_column_review_is_blocked_waiting() -> None:
    state = {
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "after_error_decision": None, "final_decision": None},
            "rag_db_qa": {
                "pending_column_review": {"status": "awaiting_review", "selection_id": "sel-1"},
            },
        },
        "last_action": "rag_db_qa",
    }

    status = derive_workflow_status(state)

    assert status["milestone"] == "awaiting_rag_db_column_review"
    assert status["completion_status"] == "blocked_waiting"
    assert status["blocker_signature"] == "waiting_for_rag_db_column_review:sel-1"


def test_workflow_status_blocks_on_rag_db_extraction_opt_in() -> None:
    state = {
        "meta": {},
        "artifacts": {"datasets": {}},
        "output": {"qa_response": "Would you like me to identify the tables and columns suitable for this extraction?"},
        "agents": {
            "rag_db_qa": {
                "pending_extraction_opt_in": {
                    "intent_id": "intent-1",
                    "goal_text": "subset age among index cases",
                    "status": "awaiting_reply",
                }
            }
        },
        "last_action": "rag_db_qa",
        "messages": [],
        "node_data": {},
    }

    status = derive_workflow_status(state)

    assert status["milestone"] == "awaiting_rag_db_extraction_opt_in"
    assert status["completion_status"] == "blocked_waiting"
    assert status["blocker_signature"] == "waiting_for_rag_db_extraction_opt_in:intent-1"


def test_pending_rag_db_sql_review_is_blocked_waiting() -> None:
    state = {
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "after_error_decision": None, "final_decision": None},
            "rag_db_qa": {
                "pending_sql_candidate": {"selection_id": "sel-1", "status": "prepared"},
            },
        },
        "last_action": "rag_db_qa",
    }

    status = derive_workflow_status(state)

    assert status["milestone"] == "awaiting_rag_db_sql_review"
    assert status["completion_status"] == "blocked_waiting"
    assert status["blocker_signature"] == "waiting_for_rag_db_sql_review:sel-1"


def test_db_rag_sql_execution_error_is_blocked_waiting_after_sql_review_node_runs() -> None:
    state = {
        "output": {
            "qa_response": "DB-RAG SQL execution failed and the workflow stopped.",
            "error": {
                "category": "db_rag_sql",
                "type": "BinderException",
                "message": 'Referenced column "SEX" not found in FROM clause',
            },
            "generated_sql": 'SELECT "AGE", "SEX" FROM "Form 1A"',
        },
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "after_error_decision": None, "final_decision": None},
            "rag_db_qa": {
                "status": "error",
                "active_thread": False,
            },
        },
        "meta": {},
        "last_action": "human_review_rag_db_sql_execution",
    }

    status = derive_workflow_status(state)

    assert status["milestone"] == "db_rag_sql_error"
    assert status["completion_status"] == "blocked_waiting"
    assert status["blocker_signature"].startswith("db_rag_sql_error:BinderException:")


def test_completed_db_rag_sql_execution_is_complete() -> None:
    state = {
        "output": {
            "qa_response": "Read-only SQL execution completed with 1 result row(s).",
            "generated_sql": 'SELECT "AGE" FROM "Form 1A"',
        },
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "after_error_decision": None, "final_decision": None},
            "rag_db_qa": {
                "status": "done",
                "active_thread": False,
                "thread_status": "completed",
            },
        },
        "meta": {},
        "last_action": "human_review_rag_db_sql_execution",
    }

    status = derive_workflow_status(state)

    assert status["milestone"] == "db_rag_sql_completed"
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
