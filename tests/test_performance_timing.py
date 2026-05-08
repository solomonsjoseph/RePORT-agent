from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.performance import append_workflow_timings, collect_timings, timing_stage


def test_timing_stage_records_elapsed_time() -> None:
    with collect_timings() as records:
        with timing_stage("db_rag.example", model="test-model", count=3):
            pass

    assert len(records) == 1
    assert records[0]["stage"] == "db_rag.example"
    assert records[0]["model"] == "test-model"
    assert records[0]["count"] == 3
    assert records[0]["elapsed_ms"] >= 0


def test_timing_stage_is_noop_without_collector() -> None:
    with timing_stage("db_rag.noop"):
        pass


def test_append_workflow_timings_stores_bounded_records() -> None:
    state = {
        "meta": {
            "workflow_timing": {
                "stages": [
                    {"stage": f"old.{index}", "elapsed_ms": index}
                    for index in range(99)
                ]
            }
        }
    }
    records = [
        {"stage": "node.qa", "elapsed_ms": 12.5},
        {"stage": "planner.llm_invoke", "elapsed_ms": 42.0, "actions": 4},
    ]

    updated = append_workflow_timings(state, records)

    stages = updated["meta"]["workflow_timing"]["stages"]
    assert len(stages) == 100
    assert stages[0]["stage"] == "old.1"
    assert stages[-2] == {"stage": "node.qa", "elapsed_ms": 12.5}
    assert stages[-1] == {
        "stage": "planner.llm_invoke",
        "elapsed_ms": 42.0,
        "actions": 4,
    }


def test_append_workflow_timings_ignores_empty_records() -> None:
    state = {"meta": {"existing": True}}

    updated = append_workflow_timings(state, [])

    assert updated is state
