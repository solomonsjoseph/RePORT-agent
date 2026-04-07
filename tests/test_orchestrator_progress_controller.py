from __future__ import annotations

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
