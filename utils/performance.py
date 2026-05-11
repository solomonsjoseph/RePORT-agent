from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from typing import Any, Iterator

TimingRecord = dict[str, Any]

WORKFLOW_TIMING_META_KEY = "workflow_timing"
TIMING_STAGE_LIMIT = 100

_ACTIVE_TIMING_RECORDS: ContextVar[list[TimingRecord] | None] = ContextVar(
    "active_timing_records",
    default=None,
)


@contextmanager
def collect_timings() -> Iterator[list[TimingRecord]]:
    records: list[TimingRecord] = []
    token = _ACTIVE_TIMING_RECORDS.set(records)
    try:
        yield records
    finally:
        _ACTIVE_TIMING_RECORDS.reset(token)


@contextmanager
def timing_stage(stage: str, **metadata: Any) -> Iterator[None]:
    records = _ACTIVE_TIMING_RECORDS.get()
    if records is None:
        yield
        return

    start = perf_counter()
    try:
        yield
    finally:
        elapsed_ms = round((perf_counter() - start) * 1000, 2)
        record: TimingRecord = {"stage": stage, "elapsed_ms": elapsed_ms}
        for key, value in metadata.items():
            if value is not None:
                record[key] = value
        records.append(record)


def append_workflow_timings(
    state: dict[str, Any],
    records: list[TimingRecord],
) -> dict[str, Any]:
    if not records:
        return state

    meta = dict(state.get("meta") or {})
    timing = dict(meta.get(WORKFLOW_TIMING_META_KEY) or {})
    stages = list(timing.get("stages") or [])
    stages.extend(dict(record) for record in records)
    timing["stages"] = stages[-TIMING_STAGE_LIMIT:]
    meta[WORKFLOW_TIMING_META_KEY] = timing
    return {
        **state,
        "meta": meta,
    }


def combined_timing_stages(meta: dict[str, Any]) -> list[TimingRecord]:
    combined: list[TimingRecord] = []
    for source, key in (
        ("workflow", WORKFLOW_TIMING_META_KEY),
        ("db_rag", "db_rag_timing"),
    ):
        timing = dict(meta.get(key) or {})
        for record in list(timing.get("stages") or []):
            if not isinstance(record, dict):
                continue
            combined.append({"source": source, **dict(record)})
    return combined
