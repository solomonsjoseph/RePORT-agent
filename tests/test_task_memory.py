from __future__ import annotations

import re

import pytest

from graph.memory import (
    ALLOWED_REFERENCE_LABELS,
    ALLOWED_RELATIONSHIPS,
    ALLOWED_TASK_KINDS,
    EDITABLE_TASK_FIELDS,
    cache_reference_resolution,
    complete_task,
    ensure_memory_state,
    get_cached_reference_resolution,
    is_json_safe,
    latest_task_cards,
    update_task_enrichment,
)
from graph.state import AgentState, MetaKeys


EXPECTED_EMPTY_MEMORY = {
    "completed_tasks": {},
    "failed_tasks": {},
    "task_order": [],
    "last_task_id": None,
    "last_task_id_by_kind": {},
    "last_failed_task_id_by_kind": {},
    "last_reference_resolution": None,
    "pending_reference_clarification": None,
}


def _old_state() -> dict:
    return {
        "messages": [],
        "output": {},
        "artifacts": {"files": {}, "datasets": {}},
        "next_action": None,
        "last_action": None,
        "observations": [],
        "orchestrator": {},
        "planner": {},
        "agents": {},
        "node_data": {},
        "meta": {},
    }


def _complete(
    state: dict,
    *,
    kind: str = "db_rag_sql_extraction",
    source_question: object = "Subset index cases with diabetes",
    goal_text: object = "Create a diabetes subset",
    label: str = "Diabetes subset",
    summary: object = "Reviewed SQL was executed and saved as a subset dataset.",
    artifact_refs: dict | None = None,
    event_refs: dict | None = None,
    provenance: dict | None = None,
    tags: list | None = None,
    parent_task_id: str | None = None,
    relationship_to_parent: str | None = None,
) -> dict:
    return complete_task(
        state,
        kind=kind,
        source_question=source_question,
        goal_text=goal_text,
        label=label,
        summary=summary,
        artifact_refs=artifact_refs
        or {
            "selection_artifact_id": "sel-art-1",
            "sql_candidate_artifact_id": "sql-art-1",
            "dataset_artifact_id": "dataset-1",
        },
        event_refs=event_refs
        if event_refs is not None
        else {"user_event_id": "evt-user-1", "completion_event_id": "evt-done-1"},
        parent_task_id=parent_task_id,
        relationship_to_parent=relationship_to_parent,
        provenance=provenance
        if provenance is not None
        else {"producer_node": "rag_db_qa", "model": "test-model"},
        tags=tags,
    )


def test_ensure_memory_state_initializes_old_state() -> None:
    state, memory = ensure_memory_state(_old_state())

    assert state["memory"] == EXPECTED_EMPTY_MEMORY
    assert memory is state["memory"]


def test_agent_state_declares_memory_and_meta_keys_are_exact() -> None:
    assert "memory" in AgentState.__annotations__
    assert MetaKeys.RESOLVED_TASK_ID == "resolved_task_id"
    assert MetaKeys.RESOLVED_TASK_KIND == "resolved_task_kind"
    assert MetaKeys.RESOLVED_TASK_RELATIONSHIP == "resolved_task_relationship"
    assert MetaKeys.RESOLVED_TASK_INTENDED_ACTION == "resolved_task_intended_action"
    assert MetaKeys.RESOLVED_TASK_USER_MESSAGE_HASH == "resolved_task_user_message_hash"


def test_schema_constants_are_exact() -> None:
    assert ALLOWED_TASK_KINDS == {
        "qa_answer",
        "db_rag_metadata_answer",
        "db_rag_sql_extraction",
        "code_analysis",
    }
    assert ALLOWED_REFERENCE_LABELS == {"resolved", "new_task", "ambiguous", "unknown"}
    assert ALLOWED_RELATIONSHIPS == {
        "revision",
        "rerun",
        "explain",
        "inspect_artifact",
        "use_as_input",
        "compare",
    }
    assert EDITABLE_TASK_FIELDS == {"label", "summary", "tags", "last_enriched_at"}


def test_complete_task_updates_order_and_kind_pointers() -> None:
    state = _complete(_old_state())
    memory = state["memory"]
    task_id = memory["task_order"][0]

    assert memory["last_task_id"] == task_id
    assert memory["last_task_id_by_kind"] == {"db_rag_sql_extraction": task_id}
    assert memory["completed_tasks"][task_id]["task_id"] == task_id
    assert memory["completed_tasks"][task_id]["display_ordinal"] == 1


def test_complete_task_rejects_non_json_artifact_refs() -> None:
    with pytest.raises(ValueError, match="artifact_refs"):
        _complete(_old_state(), artifact_refs={"bad": object()})


@pytest.mark.parametrize(
    ("field_name", "kwargs"),
    [
        ("source_question", {"source_question": 123}),
        ("source_question", {"source_question": {"question": "bad"}}),
        ("source_question", {"source_question": ["bad"]}),
        ("source_question", {"source_question": True}),
        ("source_question", {"source_question": object()}),
        ("goal_text", {"goal_text": 123}),
        ("goal_text", {"goal_text": {"goal": "bad"}}),
        ("goal_text", {"goal_text": ["bad"]}),
        ("goal_text", {"goal_text": False}),
        ("goal_text", {"goal_text": object()}),
    ],
)
def test_complete_task_rejects_non_string_goal_fields(field_name: str, kwargs: dict) -> None:
    with pytest.raises(ValueError, match=field_name):
        _complete(_old_state(), **kwargs)


@pytest.mark.parametrize(
    "artifact_refs",
    [
        {"sql_candidate_artifact_id": {"sql": "SELECT 1"}},
        {"sql_candidate_artifact_id": 123},
        {"sql_candidate_artifact_id": True},
        {"sql_candidate_artifact_id": ["sql-art-1"]},
    ],
)
def test_complete_task_rejects_non_string_artifact_ref_values(artifact_refs: dict) -> None:
    with pytest.raises(ValueError, match="artifact_refs"):
        _complete(_old_state(), artifact_refs=artifact_refs)


@pytest.mark.parametrize(
    "event_refs",
    [
        {"completion_event_id": {"event": "evt-1"}},
        {"completion_event_id": 123},
        {"completion_event_id": False},
        {"completion_event_id": ["evt-1"]},
    ],
)
def test_complete_task_rejects_non_string_event_ref_values(event_refs: dict) -> None:
    with pytest.raises(ValueError, match="event_refs"):
        _complete(_old_state(), event_refs=event_refs)


@pytest.mark.parametrize(
    ("field_name", "kwargs"),
    [
        ("artifact_refs", {"artifact_refs": {1: "non-string-key"}}),
        ("event_refs", {"event_refs": {"bad": object()}}),
        ("event_refs", {"event_refs": {1: "non-string-key"}}),
        ("provenance", {"provenance": {"bad": object()}}),
        ("provenance", {"provenance": {1: "non-string-key"}}),
        ("label", {"label": object()}),
        ("summary", {"summary": object()}),
        ("tags", {"tags": ["valid", object()]}),
        ("tags", {"tags": [{"bad": object()}]}),
        ("tags", {"tags": [{1: "non-string-key"}]}),
    ],
)
def test_complete_task_rejects_non_json_values(field_name: str, kwargs: dict) -> None:
    with pytest.raises(ValueError, match=field_name):
        _complete(_old_state(), **kwargs)


def test_json_safety_rejects_direct_objects_and_non_string_dict_keys() -> None:
    assert is_json_safe({"nested": [None, "text", 1, 1.5, True]})
    assert not is_json_safe(object())
    assert not is_json_safe({1: "not-json-object-key"})


def test_complete_task_rejects_unknown_kind_parent_and_relationship() -> None:
    with pytest.raises(ValueError, match="Unknown task kind"):
        _complete(_old_state(), kind="unknown_kind")

    with pytest.raises(ValueError, match="Unknown parent_task_id"):
        _complete(_old_state(), parent_task_id="task_missing", relationship_to_parent="revision")

    parent_state = _complete(_old_state())
    parent_id = parent_state["memory"]["last_task_id"]
    with pytest.raises(ValueError, match="Unknown relationship_to_parent"):
        _complete(
            parent_state,
            parent_task_id=parent_id,
            relationship_to_parent="invalid_relationship",
        )


def test_complete_task_stores_refs_without_artifact_payloads() -> None:
    state = _complete(
        _old_state(),
        artifact_refs={"sql_candidate_artifact_id": "sql-art-1", "payload": "literal-id-only"},
    )
    card = latest_task_cards(state)[0]

    assert card["artifact_refs"] == {
        "sql_candidate_artifact_id": "sql-art-1",
        "payload": "literal-id-only",
    }
    assert "provenance" not in card
    assert "event_refs" not in card


def test_task_ids_are_opaque_and_display_ordinals_are_monotonic() -> None:
    state = _complete(_old_state(), label="First")
    state = _complete(state, kind="qa_answer", label="Second")
    memory = state["memory"]
    first_id, second_id = memory["task_order"]

    assert re.fullmatch(r"task_[0-9a-f]{8}", first_id)
    assert re.fullmatch(r"task_[0-9a-f]{8}", second_id)
    assert first_id != second_id
    assert memory["completed_tasks"][first_id]["display_ordinal"] == 1
    assert memory["completed_tasks"][second_id]["display_ordinal"] == 2


def test_failed_task_containers_initialize_on_old_state() -> None:
    _state, memory = ensure_memory_state(_old_state())

    assert memory["failed_tasks"] == {}
    assert memory["last_failed_task_id_by_kind"] == {}


def test_ensure_memory_state_normalizes_malformed_memory_containers() -> None:
    state = _old_state()
    state["memory"] = {
        "completed_tasks": [],
        "failed_tasks": "bad",
        "task_order": None,
        "last_task_id": 123,
        "last_task_id_by_kind": None,
        "last_failed_task_id_by_kind": [],
        "last_reference_resolution": "bad",
        "pending_reference_clarification": "bad",
    }

    _state, memory = ensure_memory_state(state)

    assert memory == EXPECTED_EMPTY_MEMORY


def test_ensure_memory_state_removes_stale_task_order_ids() -> None:
    state = _old_state()
    state["memory"] = {
        **EXPECTED_EMPTY_MEMORY,
        "completed_tasks": {},
        "task_order": ["task_missing"],
        "last_task_id": "task_missing",
        "last_task_id_by_kind": {"qa_answer": "task_missing"},
    }

    _state, memory = ensure_memory_state(state)
    assert memory["task_order"] == []
    assert memory["last_task_id"] is None
    assert memory["last_task_id_by_kind"] == {}

    state = _complete(state, kind="qa_answer")
    task_id = state["memory"]["last_task_id"]
    assert state["memory"]["completed_tasks"][task_id]["display_ordinal"] == 1


def test_ensure_memory_state_resets_indexes_when_completed_tasks_malformed() -> None:
    state = _old_state()
    state["memory"] = {
        **EXPECTED_EMPTY_MEMORY,
        "completed_tasks": [],
        "task_order": ["task_stale"],
        "last_task_id": "task_stale",
        "last_task_id_by_kind": {"qa_answer": "task_stale"},
    }

    _state, memory = ensure_memory_state(state)
    assert memory["completed_tasks"] == {}
    assert memory["task_order"] == []
    assert memory["last_task_id"] is None
    assert memory["last_task_id_by_kind"] == {}

    state = _complete(state, kind="qa_answer")
    task_id = state["memory"]["last_task_id"]
    assert state["memory"]["completed_tasks"][task_id]["display_ordinal"] == 1


def test_update_task_enrichment_preserves_immutable_fields() -> None:
    state = _complete(_old_state())
    task_id = state["memory"]["last_task_id"]
    original = dict(state["memory"]["completed_tasks"][task_id])

    updated = update_task_enrichment(
        state,
        task_id,
        label="Updated label",
        summary="Updated summary",
        tags=["reviewed", "subset"],
        last_enriched_at="2026-05-05T12:00:00+00:00",
    )
    card = updated["memory"]["completed_tasks"][task_id]

    assert card["label"] == "Updated label"
    assert card["summary"] == "Updated summary"
    assert card["tags"] == ["reviewed", "subset"]
    assert card["last_enriched_at"] == "2026-05-05T12:00:00+00:00"
    for field in (
        "task_id",
        "display_ordinal",
        "kind",
        "source_question",
        "goal_text",
        "artifact_refs",
        "event_refs",
        "parent_task_id",
        "relationship_to_parent",
        "created_at",
        "completed_at",
        "status",
        "provenance",
    ):
        assert card[field] == original[field]

    with pytest.raises(ValueError, match="immutable"):
        update_task_enrichment(updated, task_id, kind="qa_answer")


def test_complete_task_deepcopies_mutable_inputs() -> None:
    artifact_refs = {"sql_candidate_artifact_id": "sql-art-1"}
    event_refs = {"completion_event_id": "evt-1"}
    provenance = {"nested": {"model": "test-model"}}
    tags = ["initial"]

    state = complete_task(
        _old_state(),
        kind="db_rag_sql_extraction",
        source_question="Subset records",
        goal_text="Create subset",
        label="Subset",
        summary="Subset summary",
        artifact_refs=artifact_refs,
        event_refs=event_refs,
        provenance=provenance,
        tags=tags,
    )
    task_id = state["memory"]["last_task_id"]

    artifact_refs["sql_candidate_artifact_id"] = "mutated"
    event_refs["completion_event_id"] = "mutated"
    provenance["nested"]["model"] = "mutated"
    tags.append("mutated")

    card = state["memory"]["completed_tasks"][task_id]
    assert card["artifact_refs"] == {"sql_candidate_artifact_id": "sql-art-1"}
    assert card["event_refs"] == {"completion_event_id": "evt-1"}
    assert card["provenance"] == {"nested": {"model": "test-model"}}
    assert card["tags"] == ["initial"]


def test_latest_task_cards_returns_deepcopies() -> None:
    state = _complete(_old_state())
    task_id = state["memory"]["last_task_id"]
    card = latest_task_cards(state)[0]

    card["artifact_refs"]["sql_candidate_artifact_id"] = "mutated"

    stored = state["memory"]["completed_tasks"][task_id]
    assert stored["artifact_refs"]["sql_candidate_artifact_id"] == "sql-art-1"


def test_update_task_enrichment_allowed_fields_are_exact() -> None:
    state = _complete(_old_state())
    task_id = state["memory"]["last_task_id"]
    for field_name in EDITABLE_TASK_FIELDS:
        value = ["tag"] if field_name == "tags" else f"{field_name}-value"
        state = update_task_enrichment(state, task_id, **{field_name: value})
        assert state["memory"]["completed_tasks"][task_id][field_name] == value

    for field_name in {"kind", "source_question", "artifact_refs", "status"}:
        with pytest.raises(ValueError, match="immutable"):
            update_task_enrichment(state, task_id, **{field_name: "mutated"})


def test_update_task_enrichment_accepts_last_enriched_at_string_or_none() -> None:
    state = _complete(_old_state())
    task_id = state["memory"]["last_task_id"]

    state = update_task_enrichment(state, task_id, last_enriched_at="2026-05-05T12:00:00+00:00")
    assert state["memory"]["completed_tasks"][task_id]["last_enriched_at"] == "2026-05-05T12:00:00+00:00"

    state = update_task_enrichment(state, task_id, last_enriched_at=None)
    assert state["memory"]["completed_tasks"][task_id]["last_enriched_at"] is None


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("label", object()),
        ("label", 123),
        ("summary", {"bad": object()}),
        ("summary", 123),
        ("tags", "not-a-list"),
        ("tags", {"not": "a-list"}),
        ("tags", ["valid", object()]),
        ("tags", ["valid", 123]),
        ("tags", [{"nested": "dict"}]),
        ("tags", [{1: "non-string-key"}]),
        ("last_enriched_at", object()),
        ("last_enriched_at", {"ts": "2026-05-05T12:00:00+00:00"}),
        ("last_enriched_at", ["2026-05-05T12:00:00+00:00"]),
        ("last_enriched_at", 123),
        ("last_enriched_at", True),
    ],
)
def test_update_task_enrichment_rejects_non_json_values(field_name: str, value: object) -> None:
    state = _complete(_old_state())
    task_id = state["memory"]["last_task_id"]

    with pytest.raises(ValueError, match=field_name):
        update_task_enrichment(state, task_id, **{field_name: value})


def test_reference_resolution_cache_round_trips_by_user_message_hash() -> None:
    result = {
        "label": "resolved",
        "task_id": "task_12345678",
        "relationship": "inspect_artifact",
        "intended_action": "show_sql",
    }
    state = cache_reference_resolution(_old_state(), "hash-1", result)

    assert get_cached_reference_resolution(state, "hash-1") == result
    assert get_cached_reference_resolution(state, "hash-2") is None
    assert state["memory"]["last_reference_resolution"] == {
        "user_message_hash": "hash-1",
        "result": result,
    }


def test_reference_resolution_cache_deepcopies_result() -> None:
    result = {
        "label": "resolved",
        "task_id": "task_12345678",
        "metadata": {"relationship": "inspect_artifact"},
    }
    state = cache_reference_resolution(_old_state(), "hash-1", result)

    result["metadata"]["relationship"] = "mutated"
    cached = get_cached_reference_resolution(state, "hash-1")
    assert cached == {
        "label": "resolved",
        "task_id": "task_12345678",
        "metadata": {"relationship": "inspect_artifact"},
    }

    cached["metadata"]["relationship"] = "mutated-again"
    assert state["memory"]["last_reference_resolution"]["result"]["metadata"] == {
        "relationship": "inspect_artifact"
    }


def test_reference_resolution_cache_rejects_non_json_result() -> None:
    with pytest.raises(ValueError, match="result"):
        cache_reference_resolution(_old_state(), "hash-1", {"bad": object()})
