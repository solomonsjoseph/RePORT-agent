from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph.memory import (
    build_user_intent_classifier_payload,
    classify_user_intent_reference,
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
