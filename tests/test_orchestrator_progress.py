from graph.nodes.orchestrator.progress import (
    build_progress_snapshot,
    classify_progress,
    update_progress_tracking,
)


def test_classify_progress_true_when_generated_code_changes() -> None:
    before = build_progress_snapshot({"last_action": "qa", "artifacts": {"generated_code": ""}, "meta": {}})
    after = build_progress_snapshot({"last_action": "generate_code", "artifacts": {"generated_code": "print(1)"}, "meta": {}})

    assert classify_progress(before, after) is True


def test_build_progress_snapshot_counts_structured_tool_results_and_ignores_legacy_string_output() -> None:
    state = {
        "agents": {
            "qa": {"tool_results": [{"tool_name": "search"}]},
            "generate_code": {"tool_results": [{"tool_name": "calc"}, {"tool_name": "plot"}]},
        },
        "node_data": {"executor": {"tool_results": [{"tool_name": "run"}]}},
        "output": {"tool_results": "formatted legacy string"},
        "artifacts": {"tool_results": "another legacy string"},
        "meta": {},
    }

    snapshot = build_progress_snapshot(state)

    assert snapshot["tool_result_count"] == 4


def test_build_progress_snapshot_preserves_distinct_review_decisions() -> None:
    before = build_progress_snapshot(
        {
            "agents": {
                "human_review": {
                    "final_decision": None,
                    "before_run_decision": "approve",
                    "after_error_decision": None,
                }
            },
            "meta": {},
        }
    )
    after = build_progress_snapshot(
        {
            "agents": {
                "human_review": {
                    "final_decision": None,
                    "before_run_decision": "approve",
                    "after_error_decision": "regenerate",
                }
            },
            "meta": {},
        }
    )

    assert before["review_status"] == {
        "final_decision": None,
        "before_run_decision": "approve",
        "after_error_decision": None,
    }
    assert after["review_status"] == {
        "final_decision": None,
        "before_run_decision": "approve",
        "after_error_decision": "regenerate",
    }
    assert classify_progress(before, after) is True


def test_classify_progress_true_when_code_changes_even_if_error_signature_stays_the_same() -> None:
    before = {
        "last_action": "execute_code",
        "generated_code_hash": "abc",
        "execution_status": "error",
        "error_signature": "retryable_code:NameError",
        "tool_result_count": 0,
        "clarification_pending": False,
        "review_status": {
            "final_decision": None,
            "before_run_decision": None,
            "after_error_decision": None,
        },
        "user_visible_output_hash": None,
    }
    after = {
        **before,
        "generated_code_hash": "def",
    }

    assert classify_progress(before, after) is True


def test_update_progress_tracking_increments_stagnation_on_identical_error_state() -> None:
    state = {
        "last_action": "execute_code",
        "artifacts": {"generated_code": "print(1)", "error": {"category": "retryable_code", "message": "NameError"}},
        "agents": {"human_review": {"final_decision": None, "before_run_decision": None, "after_error_decision": None}},
        "meta": {
            "progress_snapshot": build_progress_snapshot(
                {
                    "last_action": "execute_code",
                    "artifacts": {
                        "generated_code": "print(1)",
                        "error": {"category": "retryable_code", "message": "NameError"},
                    },
                    "agents": {
                        "human_review": {
                            "final_decision": None,
                            "before_run_decision": None,
                            "after_error_decision": None,
                        }
                    },
                    "node_data": {"executor": {"run_status": "error"}},
                    "meta": {},
                }
            ),
            "stagnation_count": 1,
        },
        "node_data": {"executor": {"run_status": "error"}},
    }

    updated_meta = update_progress_tracking(state)["meta"]

    assert updated_meta["progress_made_last_step"] is False
    assert updated_meta["stagnation_count"] == 2
