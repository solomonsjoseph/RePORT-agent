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


def test_get_node_data_merges_legacy_and_new_sections_with_new_values_taking_precedence() -> None:
    state = {
        "agents": {"qa": {"status": "idle", "attempts": 1}},
        "node_data": {"qa": {"status": "done", "response": "ok"}},
    }

    node_data = get_node_data(state, "qa")

    assert node_data["status"] == "done"
    assert node_data["attempts"] == 1
    assert node_data["response"] == "ok"


def test_get_planner_state_reads_legacy_orchestrator_and_preserves_nested_planner_fields() -> None:
    state = {
        "orchestrator": {
            "last_decision": {"action": "qa", "reason": "legacy"},
            "thoughts": ["legacy"],
        },
        "planner": {"last_decision": {"action": "qa"}},
    }

    planner = get_planner_state(state)

    assert planner["last_decision"]["action"] == "qa"
    assert planner["last_decision"]["reason"] == "legacy"
    assert planner["decision_trace"] == ["legacy"]


def test_merge_state_patch_updates_new_sections_without_dropping_legacy_keys() -> None:
    state = {
        "output": {"generated_code": ""},
        "agents": {"qa": {"status": "idle"}},
        "artifacts": {"generated_code": ""},
        "node_data": {"qa": {"status": "idle", "details": {"attempts": 1, "last_error": "old"}}},
        "orchestrator": {
            "last_decision": {"action": "qa", "reason": "legacy"},
            "thoughts": ["legacy"],
        },
        "planner": {"last_decision": {"action": "qa", "reason": "legacy"}},
    }
    patch = {
        "artifacts": {"generated_code": "print(2)"},
        "node_data": {"qa": {"status": "done", "details": {"attempts": 2}, "response": "ok"}},
        "planner": {"last_decision": {"action": "qa"}, "decision_trace": ["qa selected"]},
    }

    updated = merge_state_patch(state, patch)

    assert updated["artifacts"]["generated_code"] == "print(2)"
    assert updated["output"]["generated_code"] == "print(2)"
    assert updated["node_data"]["qa"]["status"] == "done"
    assert updated["node_data"]["qa"]["details"]["attempts"] == 2
    assert updated["node_data"]["qa"]["details"]["last_error"] == "old"
    assert updated["node_data"]["qa"]["response"] == "ok"
    assert updated["agents"]["qa"]["response"] == "ok"
    assert updated["agents"]["qa"]["details"]["last_error"] == "old"
    assert updated["planner"]["last_decision"]["action"] == "qa"
    assert updated["planner"]["last_decision"]["reason"] == "legacy"
    assert updated["orchestrator"]["last_decision"]["action"] == "qa"
    assert updated["orchestrator"]["last_decision"]["reason"] == "legacy"
    assert updated["orchestrator"]["thoughts"] == ["qa selected"]
    assert get_planner_state(updated)["last_decision"]["reason"] == "legacy"
    assert get_planner_state(updated)["decision_trace"] == ["qa selected"]
