from __future__ import annotations

import sys
from types import ModuleType

messages_mod = ModuleType("langchain_core.messages")
messages_mod.BaseMessage = object
sys.modules.setdefault("langchain_core.messages", messages_mod)

graph_message_mod = ModuleType("langgraph.graph.message")
graph_message_mod.add_messages = lambda current, new: (current or []) + (new or [])
sys.modules.setdefault("langgraph.graph.message", graph_message_mod)

from graph.nodes.orchestrator.progress_controller import (
    apply_recurrence_guard,
    classify_progress,
    update_recurrence_state,
)


def test_classify_progress_marks_code_change_inside_same_review_milestone_as_weak() -> None:
    previous = {
        "milestone": "awaiting_run_review",
        "completion_status": "blocked_waiting",
        "blocker_signature": "waiting_for_before_run_review",
        "generated_code_hash": "old",
        "error_signature": None,
        "selected_action": "generate_code",
        "user_turn_hash": "u1",
    }
    current = {
        **previous,
        "generated_code_hash": "new",
    }

    assert classify_progress(previous, current) == "weak"


def test_classify_progress_marks_milestone_advance_as_hard() -> None:
    previous = {
        "milestone": "needs_code",
        "completion_status": "incomplete",
        "blocker_signature": "missing_next_step",
        "generated_code_hash": None,
        "error_signature": None,
        "selected_action": "qa",
        "user_turn_hash": "u1",
    }
    current = {
        "milestone": "awaiting_run_review",
        "completion_status": "blocked_waiting",
        "blocker_signature": "waiting_for_before_run_review",
        "generated_code_hash": "new",
        "error_signature": None,
        "selected_action": "generate_code",
        "user_turn_hash": "u1",
    }

    assert classify_progress(previous, current) == "hard"


def test_update_recurrence_state_increments_stagnation_for_same_blocker_without_delta() -> None:
    state = {
        "last_action": "qa",
        "output": {"qa_response": "I am a data-analysis assistant."},
        "meta": {
            "workflow_milestone": "answered",
            "completion_status": "complete",
            "blocker_signature": None,
            "progress_snapshot": {
                "milestone": "answered",
                "completion_status": "complete",
                "blocker_signature": None,
                "generated_code_hash": None,
                "error_signature": None,
                "selected_action": "qa",
                "user_turn_hash": "u1",
            },
            "stagnation_count": 1,
            "weak_progress_count": 0,
            "last_user_message_hash": "u1",
        },
    }

    updated = update_recurrence_state(state)

    assert updated["meta"]["progress_class"] == "none"
    assert updated["meta"]["stagnation_count"] == 2
    assert updated["meta"]["weak_progress_count"] == 0


def test_apply_recurrence_guard_ends_after_repeated_weak_churn_in_same_milestone() -> None:
    state = {
        "meta": {
            "workflow_milestone": "awaiting_run_review",
            "blocker_signature": "waiting_for_before_run_review",
            "weak_progress_count": 4,
            "stagnation_count": 0,
        }
    }

    action, observations, fired = apply_recurrence_guard("generate_code", state, [])

    assert fired is True
    assert action == "end"
    assert observations and "weak-progress churn" in observations[-1]


def test_apply_recurrence_guard_does_not_block_retry_loop_before_retry_budget() -> None:
    state = {
        "meta": {
            "workflow_milestone": "retrying_after_error",
            "blocker_signature": "retryable_error:PythonRuntimeError:deadbeef",
            "stagnation_count": 4,
            "weak_progress_count": 0,
            "error_iterations": 2,
        }
    }

    action, observations, fired = apply_recurrence_guard("error_handler", state, [])

    assert fired is False
    assert action == "error_handler"
    assert observations and "retry budget controls" in observations[-1]


def test_apply_recurrence_guard_preserves_after_error_review_handoff() -> None:
    state = {
        "meta": {
            "workflow_milestone": "awaiting_after_error_review",
            "blocker_signature": "waiting_for_after_error_review:retryable_code:PythonRuntimeError",
            "stagnation_count": 4,
            "weak_progress_count": 0,
            "error_iterations": 5,
        }
    }

    action, observations, fired = apply_recurrence_guard("human_review_after_error", state, [])

    assert fired is False
    assert action == "human_review_after_error"
    assert observations and "preserving after-error human review" in observations[-1]
