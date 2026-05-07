from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.performance import collect_timings, timing_stage


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
