from graph.nodes.orchestrator.progress import (
    build_progress_snapshot,
    classify_progress,
    update_progress_tracking,
)


def test_classify_progress_true_when_generated_code_changes() -> None:
    before = build_progress_snapshot({"last_action": "qa", "artifacts": {"generated_code": ""}, "meta": {}})
    after = build_progress_snapshot({"last_action": "generate_code", "artifacts": {"generated_code": "print(1)"}, "meta": {}})

    assert classify_progress(before, after) is True


def test_update_progress_tracking_increments_stagnation_on_same_error() -> None:
    state = {
        "last_action": "execute_code",
        "artifacts": {"generated_code": "print(1)", "error": {"category": "retryable_code", "message": "NameError"}},
        "meta": {
            "progress_snapshot": {
                "last_action": "execute_code",
                "generated_code_hash": "abc",
                "execution_status": "error",
                "error_signature": "retryable_code:NameError",
                "tool_result_count": 0,
                "clarification_pending": False,
                "review_status": None,
                "user_visible_output_hash": None,
            },
            "stagnation_count": 1,
        },
        "node_data": {"executor": {"run_status": "error"}},
    }

    updated_meta = update_progress_tracking(state)["meta"]

    assert updated_meta["progress_made_last_step"] is False
    assert updated_meta["stagnation_count"] == 2
