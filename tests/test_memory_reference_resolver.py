from __future__ import annotations

import json

import pytest

from graph.memory import complete_task
from graph.memory.reference_resolver import build_resolver_payload, resolve_reference_for_turn
from graph.memory.validation import validate_reference_resolution


def _state() -> dict:
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


def _add_sql_task(
    state: dict,
    *,
    label: str = "Diabetes subset",
    sql_artifact_id: str = "sql-art-1",
    selection_artifact_id: str = "sel-art-1",
    dataset_artifact_id: str = "dataset-1",
) -> dict:
    state.setdefault("artifacts", {}).setdefault("files", {})[sql_artifact_id] = {
        "kind": "db_rag_sql_candidate",
        "content": "SELECT secret_payload FROM patients",
    }
    state["artifacts"].setdefault("files", {})[selection_artifact_id] = {
        "kind": "db_rag_column_selection",
        "content": {"columns": ["secret_column"]},
    }
    state["artifacts"].setdefault("datasets", {})[dataset_artifact_id] = {
        "kind": "subset",
        "name": "secret_dataset_name",
        "rows": 10,
    }
    return complete_task(
        state,
        kind="db_rag_sql_extraction",
        source_question=f"Create {label}",
        goal_text=f"Create {label}",
        label=label,
        summary=f"Completed {label}",
        artifact_refs={
            "selection_artifact_id": selection_artifact_id,
            "sql_candidate_artifact_id": sql_artifact_id,
            "dataset_artifact_id": dataset_artifact_id,
        },
        event_refs={"completion_event_id": f"evt-{sql_artifact_id}"},
        provenance={"producer_node": "rag_db_qa"},
    )


class FailingLLM:
    def invoke(self, _prompt: str):  # pragma: no cover - should not be reached
        raise AssertionError("LLM should not be invoked")


class StaticLLM:
    def __init__(self, result: dict):
        self.result = result
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return json.dumps(self.result)


class ContentBlocksResponse:
    def __init__(self, content: list[dict[str, str]]):
        self.content = content


class ContentBlocksLLM:
    def __init__(self, result: dict):
        self.result = result
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return ContentBlocksResponse(
            [{"type": "text", "text": json.dumps(self.result)}]
        )


def test_exact_task_id_resolves_without_llm() -> None:
    state = _add_sql_task(_state())
    task_id = state["memory"]["last_task_id"]

    result = resolve_reference_for_turn(state, FailingLLM(), f"show SQL for {task_id}", "hash-1")

    assert result["label"] == "resolved"
    assert result["task_id"] == task_id
    assert result["relationship"] == "inspect_artifact"


def test_multiple_exact_task_references_are_ambiguous_without_llm() -> None:
    state = _add_sql_task(_state(), label="First")
    first_task_id = state["memory"]["last_task_id"]
    state = _add_sql_task(
        state,
        label="Second",
        sql_artifact_id="sql-art-2",
        selection_artifact_id="sel-art-2",
        dataset_artifact_id="dataset-2",
    )
    second_task_id = state["memory"]["last_task_id"]

    result = resolve_reference_for_turn(
        state,
        FailingLLM(),
        f"compare {first_task_id} with {second_task_id}",
        "hash-1",
    )

    assert result["label"] == "ambiguous"
    assert result["needs_reference"] is True
    assert result["task_id"] is None
    candidate_ids = {card["task_id"] for card in result["candidate_cards"]}
    assert {first_task_id, second_task_id} <= candidate_ids


def test_exact_display_ordinal_resolves_without_llm() -> None:
    state = _add_sql_task(_state(), label="First")
    state = _add_sql_task(state, label="Second", sql_artifact_id="sql-art-2", selection_artifact_id="sel-art-2", dataset_artifact_id="dataset-2")

    result = resolve_reference_for_turn(state, FailingLLM(), "show SQL for Task 1", "hash-1")

    assert result["label"] == "resolved"
    assert result["task_id"] == state["memory"]["task_order"][0]
    assert result["relationship"] == "inspect_artifact"


def test_exact_task_id_for_non_db_task_resolves_without_artifacts() -> None:
    state = complete_task(
        _state(),
        kind="qa_answer",
        source_question="What is RePORTER?",
        goal_text="Answer a metadata question",
        label="RePORTER answer",
        summary="Answered a user question.",
    )
    task_id = state["memory"]["last_task_id"]

    result = resolve_reference_for_turn(state, FailingLLM(), f"explain {task_id}", "hash-1")

    assert result["label"] == "resolved"
    assert result["task_id"] == task_id
    assert result["relationship"] == "explain"


def test_exact_artifact_id_adds_direct_candidate() -> None:
    state = _add_sql_task(_state(), label="Old", sql_artifact_id="sql-old", selection_artifact_id="sel-old", dataset_artifact_id="dataset-old")
    old_task_id = state["memory"]["last_task_id"]
    for idx in range(13):
        state = _add_sql_task(
            state,
            label=f"Recent {idx}",
            sql_artifact_id=f"sql-recent-{idx}",
            selection_artifact_id=f"sel-recent-{idx}",
            dataset_artifact_id=f"dataset-recent-{idx}",
        )

    payload = build_resolver_payload(state, "inspect sql-old", "hash-1", limit=12)

    assert old_task_id in {card["task_id"] for card in payload["completed_tasks"]}


def test_exact_artifact_id_resolves_without_llm() -> None:
    state = _add_sql_task(_state(), sql_artifact_id="sql-direct")
    task_id = state["memory"]["last_task_id"]

    result = resolve_reference_for_turn(
        state,
        FailingLLM(),
        "show me the SQL in sql-direct",
        "hash-1",
    )

    assert result["label"] == "resolved"
    assert result["task_id"] == task_id
    assert result["relationship"] == "inspect_artifact"


def test_valid_llm_resolved_output_is_accepted() -> None:
    state = _add_sql_task(_state())
    task_id = state["memory"]["last_task_id"]
    llm = StaticLLM(
        {
            "label": "resolved",
            "task_id": task_id,
            "relationship": "use_as_input",
            "intended_action": "analyze_dataset",
            "confidence": "high",
            "needs_reference": False,
            "reason": "User asks to analyze the subset.",
        }
    )

    result = resolve_reference_for_turn(state, llm, "analyze that subset", "hash-1")

    assert result["label"] == "resolved"
    assert result["task_id"] == task_id
    assert result["relationship"] == "use_as_input"


def test_llm_content_list_text_blocks_are_parsed() -> None:
    state = _add_sql_task(_state())
    task_id = state["memory"]["last_task_id"]
    llm = ContentBlocksLLM(
        {
            "label": "resolved",
            "task_id": task_id,
            "relationship": "use_as_input",
            "intended_action": "analyze_dataset",
            "confidence": "high",
            "needs_reference": False,
            "reason": "User asks to analyze the subset.",
        }
    )

    result = resolve_reference_for_turn(state, llm, "analyze that subset", "hash-1")

    assert result["label"] == "resolved"
    assert result["task_id"] == task_id
    assert result["relationship"] == "use_as_input"


def test_unsupported_label_is_rejected() -> None:
    state = _add_sql_task(_state())

    with pytest.raises(ValueError, match="label"):
        validate_reference_resolution(
            state,
            {
                "label": "maybe",
                "task_id": state["memory"]["last_task_id"],
                "relationship": "inspect_artifact",
                "needs_reference": False,
            },
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("reason", {"text": "not a string"}),
        ("reason", ["not a string"]),
        ("reason", True),
        ("intended_action", {"action": "show_sql"}),
        ("intended_action", ["show_sql"]),
        ("intended_action", True),
        ("confidence", {"score": "high"}),
        ("confidence", ["high"]),
        ("confidence", True),
    ],
)
def test_validation_rejects_invalid_scalar_field_types(field_name: str, value: object) -> None:
    state = _add_sql_task(_state())
    raw_result = {
        "label": "resolved",
        "task_id": state["memory"]["last_task_id"],
        "relationship": "inspect_artifact",
        "intended_action": "show_sql",
        "confidence": "high",
        "needs_reference": False,
        "reason": "User asks for SQL.",
    }
    raw_result[field_name] = value

    with pytest.raises(ValueError, match=field_name):
        validate_reference_resolution(state, raw_result)


def test_disallowed_relationship_for_task_kind_is_rejected() -> None:
    state = complete_task(
        _state(),
        kind="qa_answer",
        source_question="What is the study?",
        goal_text="Answer a question",
        label="Study answer",
        summary="Answered a question.",
    )

    with pytest.raises(ValueError, match="not allowed"):
        validate_reference_resolution(
            state,
            {
                "label": "resolved",
                "task_id": state["memory"]["last_task_id"],
                "relationship": "use_as_input",
                "needs_reference": False,
            },
        )


def test_missing_task_id_is_rejected() -> None:
    state = _add_sql_task(_state())

    with pytest.raises(ValueError, match="task_id"):
        validate_reference_resolution(
            state,
            {
                "label": "resolved",
                "task_id": "task_missing",
                "relationship": "inspect_artifact",
                "needs_reference": False,
            },
        )


def test_missing_required_artifact_is_rejected() -> None:
    state = _add_sql_task(_state())
    task_id = state["memory"]["last_task_id"]
    del state["artifacts"]["files"]["sql-art-1"]

    with pytest.raises(ValueError, match="sql_candidate_artifact_id"):
        validate_reference_resolution(
            state,
            {
                "label": "resolved",
                "task_id": task_id,
                "relationship": "inspect_artifact",
                "needs_reference": False,
            },
        )


@pytest.mark.parametrize(
    "missing_ref",
    ["selection_artifact_id", "sql_candidate_artifact_id"],
)
def test_revision_requires_selection_and_sql_artifact_refs(missing_ref: str) -> None:
    state = _add_sql_task(_state())
    task_id = state["memory"]["last_task_id"]
    del state["memory"]["completed_tasks"][task_id]["artifact_refs"][missing_ref]

    with pytest.raises(ValueError, match=missing_ref):
        validate_reference_resolution(
            state,
            {
                "label": "resolved",
                "task_id": task_id,
                "relationship": "revision",
                "needs_reference": False,
            },
        )


@pytest.mark.parametrize(
    ("ref_name", "wrong_kind"),
    [
        ("selection_artifact_id", "text"),
        ("selection_artifact_id", "db_rag_sql_candidate"),
        ("sql_candidate_artifact_id", "generic"),
        ("sql_candidate_artifact_id", "db_rag_column_selection"),
    ],
)
def test_required_file_artifact_refs_reject_wrong_kinds(ref_name: str, wrong_kind: str) -> None:
    state = _add_sql_task(_state())
    task_id = state["memory"]["last_task_id"]
    artifact_id = state["memory"]["completed_tasks"][task_id]["artifact_refs"][ref_name]
    state["artifacts"]["files"][artifact_id]["kind"] = wrong_kind

    with pytest.raises(ValueError, match=ref_name):
        validate_reference_resolution(
            state,
            {
                "label": "resolved",
                "task_id": task_id,
                "relationship": "revision",
                "needs_reference": False,
            },
        )


def test_use_as_input_requires_dataset_ref_in_dataset_store_not_files() -> None:
    state = _add_sql_task(_state(), dataset_artifact_id="dataset-only-in-files")
    task_id = state["memory"]["last_task_id"]
    dataset_entry = state["artifacts"]["datasets"].pop("dataset-only-in-files")
    state["artifacts"]["files"]["dataset-only-in-files"] = dataset_entry

    with pytest.raises(ValueError, match="dataset_artifact_id"):
        validate_reference_resolution(
            state,
            {
                "label": "resolved",
                "task_id": task_id,
                "relationship": "use_as_input",
                "needs_reference": False,
            },
        )


def test_use_as_input_rejects_uploaded_dataset_kind() -> None:
    state = _add_sql_task(_state())
    task_id = state["memory"]["last_task_id"]
    dataset_id = state["memory"]["completed_tasks"][task_id]["artifact_refs"]["dataset_artifact_id"]
    state["artifacts"]["datasets"][dataset_id]["kind"] = "uploaded"

    with pytest.raises(ValueError, match="dataset_artifact_id"):
        validate_reference_resolution(
            state,
            {
                "label": "resolved",
                "task_id": task_id,
                "relationship": "use_as_input",
                "needs_reference": False,
            },
        )


def test_use_as_input_accepts_subset_dataset_kind() -> None:
    state = _add_sql_task(_state())
    task_id = state["memory"]["last_task_id"]
    dataset_id = state["memory"]["completed_tasks"][task_id]["artifact_refs"]["dataset_artifact_id"]

    result = validate_reference_resolution(
        state,
        {
            "label": "resolved",
            "task_id": task_id,
            "relationship": "use_as_input",
            "needs_reference": False,
        },
    )

    assert state["artifacts"]["datasets"][dataset_id]["kind"] == "subset"
    assert result["label"] == "resolved"
    assert result["task_id"] == task_id
    assert result["relationship"] == "use_as_input"


def test_ambiguous_output_preserves_candidate_cards() -> None:
    state = _add_sql_task(_state())
    cards = build_resolver_payload(state, "show that", "hash-1")["completed_tasks"]

    result = validate_reference_resolution(
        state,
        {"label": "ambiguous", "needs_reference": True, "reason": "Multiple possible tasks."},
        candidate_cards=cards,
    )

    assert result["label"] == "ambiguous"
    assert result["needs_reference"] is True
    assert result["candidate_cards"] == cards


def test_unknown_needs_reference_routes_to_clarification() -> None:
    state = _add_sql_task(_state())
    cards = build_resolver_payload(state, "show that", "hash-1")["completed_tasks"]

    result = validate_reference_resolution(
        state,
        {"label": "unknown", "needs_reference": True, "reason": "Reference is unclear."},
        candidate_cards=cards,
    )

    assert result["label"] == "unknown"
    assert result["needs_reference"] is True
    assert result["candidate_cards"] == cards
    assert result["task_id"] is None


@pytest.mark.parametrize(
    "needs_reference",
    ["false", "true", 0, 1, {"value": False}, [False], None],
)
def test_unknown_rejects_non_bool_needs_reference(needs_reference: object) -> None:
    state = _add_sql_task(_state())

    with pytest.raises(ValueError, match="needs_reference"):
        validate_reference_resolution(
            state,
            {
                "label": "unknown",
                "needs_reference": needs_reference,
                "reason": "Reference status is unclear.",
            },
        )


def test_unknown_without_reference_routes_as_new_task() -> None:
    state = _add_sql_task(_state())

    result = validate_reference_resolution(
        state,
        {"label": "unknown", "needs_reference": False, "task_id": state["memory"]["last_task_id"]},
    )

    assert result["label"] == "new_task"
    assert result["task_id"] is None
    assert result["relationship"] is None
    assert result["needs_reference"] is False


def test_resolver_uses_latest_twelve_plus_direct_matches() -> None:
    state = _add_sql_task(_state(), label="Old", sql_artifact_id="sql-old", selection_artifact_id="sel-old", dataset_artifact_id="dataset-old")
    old_task_id = state["memory"]["last_task_id"]
    for idx in range(13):
        state = _add_sql_task(
            state,
            label=f"Recent {idx}",
            sql_artifact_id=f"sql-recent-{idx}",
            selection_artifact_id=f"sel-recent-{idx}",
            dataset_artifact_id=f"dataset-recent-{idx}",
        )

    payload_without_direct = build_resolver_payload(state, "show recent work", "hash-1", limit=12)
    payload_with_direct = build_resolver_payload(state, "show sql-old", "hash-2", limit=12)

    ids_without_direct = [card["task_id"] for card in payload_without_direct["completed_tasks"]]
    ids_with_direct = [card["task_id"] for card in payload_with_direct["completed_tasks"]]
    assert len(ids_without_direct) == 12
    assert old_task_id not in ids_without_direct
    assert len(ids_with_direct) == 13
    assert old_task_id in ids_with_direct


def test_resolver_prompt_excludes_artifact_payload_content() -> None:
    state = _add_sql_task(_state())
    llm = StaticLLM({"label": "new_task", "needs_reference": False, "reason": "Fresh request."})

    resolve_reference_for_turn(state, llm, "make a different subset", "hash-1")

    prompt = llm.prompts[0]
    assert "secret_payload" not in prompt
    assert "secret_column" not in prompt
    assert "secret_dataset_name" not in prompt
    assert "sql-art-1" in prompt


def test_no_completed_tasks_returns_and_caches_new_task_without_llm() -> None:
    state = _state()

    first = resolve_reference_for_turn(state, FailingLLM(), "start a fresh analysis", "hash-1")
    second = resolve_reference_for_turn(state, FailingLLM(), "start a fresh analysis", "hash-1")

    assert first == {
        "label": "new_task",
        "task_id": None,
        "relationship": None,
        "intended_action": None,
        "confidence": None,
        "needs_reference": False,
        "reason": "No completed tasks exist.",
    }
    assert second == first
    assert state["memory"]["last_reference_resolution"] == {
        "user_message_hash": "hash-1",
        "result": first,
    }


def test_cached_resolution_reused_for_same_user_hash() -> None:
    state = _add_sql_task(_state())
    task_id = state["memory"]["last_task_id"]
    llm = StaticLLM(
        {
            "label": "resolved",
            "task_id": task_id,
            "relationship": "inspect_artifact",
            "intended_action": "show_sql",
            "confidence": "high",
            "needs_reference": False,
            "reason": "Initial resolution.",
        }
    )

    first = resolve_reference_for_turn(state, llm, "show that SQL", "hash-1")
    second = resolve_reference_for_turn(state, FailingLLM(), "show that SQL", "hash-1")

    assert first == second
    assert len(llm.prompts) == 1
