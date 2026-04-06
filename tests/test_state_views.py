from __future__ import annotations

from graph.state_views import (
    get_artifacts,
    get_node_data,
    get_planner_state,
    merge_state_patch,
)


def test_get_artifacts_reads_legacy_output_until_artifacts_exist() -> None:
    state = {"output": {"generated_code": "print(1)", "error": {"category": "retryable_code"}}}
    artifacts = get_artifacts(state)

    assert artifacts["generated_code"] == "print(1)"
    assert artifacts["error"] == {"category": "retryable_code"}


def test_merge_state_patch_updates_new_sections_without_dropping_legacy_keys() -> None:
    state = {
        "output": {"generated_code": ""},
        "agents": {"qa": {"status": "idle"}},
        "artifacts": {"generated_code": ""},
        "node_data": {"qa": {"status": "idle"}},
    }
    patch = {
        "artifacts": {"generated_code": "print(2)"},
        "node_data": {"qa": {"status": "done", "response": "ok"}},
        "planner": {"last_decision": {"action": "qa"}},
    }

    updated = merge_state_patch(state, patch)

    assert updated["artifacts"]["generated_code"] == "print(2)"
    assert updated["output"]["generated_code"] == "print(2)"
    assert updated["node_data"]["qa"]["response"] == "ok"
    assert updated["agents"]["qa"]["response"] == "ok"
    assert updated["planner"]["last_decision"]["action"] == "qa"
