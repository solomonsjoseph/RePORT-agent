from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph.memory import (
    build_user_intent_classifier_payload,
    classify_user_intent_reference,
    complete_task,
    validate_user_intent_reference,
    upsert_user_intent_from_db_rag_intent,
)


class StubClassifier:
    def __init__(self, result):
        self.result = result
        self.prompt = None

    def invoke(self, prompt):
        self.prompt = prompt
        return self.result


def _state_with_intent() -> dict:
    state = {
        "messages": [],
        "output": {},
        "artifacts": {"files": {}, "datasets": {}},
        "next_action": None,
        "last_action": None,
        "observations": [],
        "orchestrator": {},
        "planner": {},
        "agents": {"rag_db_qa": {"thread_status": "done", "active_thread": False}},
        "node_data": {},
        "meta": {},
    }
    return upsert_user_intent_from_db_rag_intent(
        state,
        active_intent={
            "intent_id": "rag-intent-1",
            "source_question": "Query my database for age",
            "goal_text": "Query age among index cases",
            "sql": "SELECT secret FROM payload",
            "raw_model_output": "do not expose",
        },
        source_message_hash="hash-1",
        status="cancelled",
    )


def _state_with_completed_task(
    *,
    artifact_refs: dict[str, str] | None = None,
) -> tuple[dict, str]:
    state = _state_with_intent()
    state = complete_task(
        state,
        kind="db_rag_sql_extraction",
        source_question="Query my database for age",
        goal_text="Query age among index cases",
        label="Age query",
        summary="Age query completed.",
        artifact_refs=artifact_refs or {"dataset_artifact_id": "dataset-1"},
        event_refs={"user_event_id": "evt-user-1", "completion_event_id": "evt-done-1"},
        provenance={"raw_model_output": "do not expose"},
    )
    return state, state["memory"]["last_task_id"]


def test_payload_contains_compact_intent_and_no_payload_artifacts() -> None:
    state = _state_with_intent()

    payload = build_user_intent_classifier_payload(
        state,
        user_message="continue previous query",
        user_message_hash="hash-2",
    )

    assert payload["latest_user_message"] == "continue previous query"
    assert payload["candidate_user_intents"][0]["source_question"] == "Query my database for age"
    assert set(payload["candidate_user_intents"][0]) == {
        "intent_id",
        "display_ordinal",
        "kind",
        "agent",
        "source_question",
        "goal_text",
        "status",
        "source_message_hash",
        "active_intent_id",
        "completed_task_id",
        "continued_from_intent_id",
        "created_at",
        "updated_at",
    }
    assert "sql" not in str(payload).lower()
    assert "raw_model_output" not in str(payload)


def test_validate_existing_user_intent_reference() -> None:
    state = _state_with_intent()
    intent_id = state["memory"]["last_user_intent_id"]

    result = validate_user_intent_reference(
        state,
        {
            "target": "existing_user_intent",
            "target_id": intent_id,
            "relationship": "continue",
            "confidence": "high",
            "reason": "User wants to continue the prior query.",
        },
    )

    assert result["target"] == "existing_user_intent"
    assert result["target_id"] == intent_id
    assert result["source_question"] == "Query my database for age"


def test_validate_rejects_invented_intent_id() -> None:
    state = _state_with_intent()

    with pytest.raises(ValueError, match="Resolved user intent does not exist"):
        validate_user_intent_reference(
            state,
            {
                "target": "existing_user_intent",
                "target_id": "intent_missing",
                "relationship": "continue",
                "confidence": "high",
                "reason": "bad",
            },
        )


def test_low_confidence_returns_ambiguous() -> None:
    state = _state_with_intent()
    intent_id = state["memory"]["last_user_intent_id"]

    result = validate_user_intent_reference(
        state,
        {
            "target": "existing_user_intent",
            "target_id": intent_id,
            "relationship": "continue",
            "confidence": "low",
            "reason": "unclear",
        },
    )

    assert result["target"] == "ambiguous"
    assert result["needs_clarification"] is True


def test_classifier_invokes_model_and_validates() -> None:
    state = _state_with_intent()
    intent_id = state["memory"]["last_user_intent_id"]
    classifier = StubClassifier(
        json.dumps(
            {
                "target": "existing_user_intent",
                "target_id": intent_id,
                "relationship": "continue",
                "confidence": "high",
                "reason": "User wants the prior DB-RAG query.",
            }
        )
    )

    result = classify_user_intent_reference(
        state,
        classifier,
        user_message="continue previous query",
        user_message_hash="hash-2",
    )

    assert result["target"] == "existing_user_intent"
    assert result["source_question"] == "Query my database for age"
    assert classifier.prompt is not None
    assert "candidate_user_intents" in classifier.prompt


def test_classifier_accepts_response_content_dict() -> None:
    state = _state_with_intent()
    intent_id = state["memory"]["last_user_intent_id"]

    class Response:
        content = {
            "target": "existing_user_intent",
            "target_id": intent_id,
            "relationship": "continue",
            "confidence": "high",
            "reason": "User wants the prior DB-RAG query.",
        }

    classifier = StubClassifier(Response())

    result = classify_user_intent_reference(
        state,
        classifier,
        user_message="continue previous query",
        user_message_hash="hash-2",
    )

    assert result["target"] == "existing_user_intent"
    assert result["source_question"] == "Query my database for age"
    assert classifier.prompt is not None
    assert "candidate_user_intents" in classifier.prompt


def test_completed_task_rejects_unsupported_relationship() -> None:
    state, task_id = _state_with_completed_task()

    with pytest.raises(ValueError, match="unsupported relationship"):
        validate_user_intent_reference(
            state,
            {
                "target": "completed_task",
                "target_id": task_id,
                "relationship": "continue",
                "confidence": "high",
                "reason": "User asks about the prior result.",
            },
        )


def test_completed_task_rejects_missing_inspectable_artifact_ref() -> None:
    state, task_id = _state_with_completed_task(artifact_refs={"other_artifact_id": "other-1"})

    with pytest.raises(ValueError, match="inspectable artifact"):
        validate_user_intent_reference(
            state,
            {
                "target": "completed_task",
                "target_id": task_id,
                "relationship": "inspect_result",
                "confidence": "high",
                "reason": "User asks about the prior result.",
            },
        )


def test_candidate_bound_user_intent_id_returns_ambiguous() -> None:
    state = _state_with_intent()
    intent_id = state["memory"]["last_user_intent_id"]

    result = validate_user_intent_reference(
        state,
        {
            "target": "existing_user_intent",
            "target_id": intent_id,
            "relationship": "continue",
            "confidence": "high",
            "reason": "User asks about a prior query.",
        },
        candidate_user_intent_ids={"intent_other"},
    )

    assert result["target"] == "ambiguous"
    assert result["needs_clarification"] is True


def test_candidate_bound_completed_task_id_returns_ambiguous() -> None:
    state, task_id = _state_with_completed_task()

    result = validate_user_intent_reference(
        state,
        {
            "target": "completed_task",
            "target_id": task_id,
            "relationship": "inspect_result",
            "confidence": "high",
            "reason": "User asks about the prior result.",
        },
        candidate_completed_task_ids={"task_other"},
    )

    assert result["target"] == "ambiguous"
    assert result["needs_clarification"] is True


def test_completed_task_payload_fields_are_whitelisted() -> None:
    state = _state_with_intent()

    payload = build_user_intent_classifier_payload(
        state,
        user_message="inspect that result",
        user_message_hash="hash-2",
        completed_task_cards=[
            {
                "task_id": "task_1",
                "kind": "db_rag_sql_extraction",
                "source_question": "Query my database for age",
                "goal_text": "Query age",
                "label": "Age query",
                "summary": "Completed.",
                "dataset_id": "dataset-1",
                "artifact_refs": {"sql_candidate_artifact_id": "sql-1"},
                "provenance": {"raw_model_output": "secret"},
                "raw_model_output": "secret",
                "sql": "SELECT secret",
            }
        ],
    )

    assert payload["candidate_completed_tasks"] == [
        {
            "task_id": "task_1",
            "kind": "db_rag_sql_extraction",
            "source_question": "Query my database for age",
            "goal_text": "Query age",
            "label": "Age query",
            "summary": "Completed.",
            "dataset_id": "dataset-1",
            "created_at": None,
            "updated_at": None,
        }
    ]
    assert "artifact_refs" not in str(payload["candidate_completed_tasks"])
    assert "raw_model_output" not in str(payload)
    assert "SELECT secret" not in str(payload)


def test_classifier_accepts_response_content_list_blocks() -> None:
    state = _state_with_intent()
    intent_id = state["memory"]["last_user_intent_id"]

    class Response:
        content = [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "target": "existing_user_intent",
                        "target_id": intent_id,
                        "relationship": "continue",
                        "confidence": "high",
                        "reason": "User wants the prior DB-RAG query.",
                    }
                ),
            }
        ]

    result = classify_user_intent_reference(
        state,
        StubClassifier(Response()),
        user_message="continue previous query",
        user_message_hash="hash-2",
    )

    assert result["target"] == "existing_user_intent"
    assert result["source_question"] == "Query my database for age"


def test_classifier_wraps_invalid_json_response() -> None:
    state = _state_with_intent()

    with pytest.raises(ValueError, match="valid JSON"):
        classify_user_intent_reference(
            state,
            StubClassifier("not json"),
            user_message="continue previous query",
            user_message_hash="hash-2",
        )
