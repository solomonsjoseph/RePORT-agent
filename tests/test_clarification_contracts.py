from __future__ import annotations

from graph.nodes.clarification_contracts import (
    CLARIFICATION_KIND_DATASET_SELECTION,
    CLARIFICATION_KIND_GENERATE_CODE,
    CLARIFICATION_KIND_QA_TOOL,
    CLARIFICATION_KIND_RAG_DB_EXTRACTION_OPT_IN,
    build_expected,
    contract_for_kind,
    resolve_clarification_reply,
)
from graph.state import MetaKeys


def test_dataset_id_reply_is_valid_without_classifier() -> None:
    meta = {
        MetaKeys.CLARIFICATION_KIND: CLARIFICATION_KIND_DATASET_SELECTION,
        MetaKeys.CLARIFICATION_RETURN_NODE: "generate_code",
        MetaKeys.PENDING_QUESTION: "subset index cases",
        MetaKeys.CLARIFICATION_EXPECTED: build_expected(
            CLARIFICATION_KIND_DATASET_SELECTION,
            allowed_values=["uploaded-1", "uploaded-2"],
        ),
    }

    result = resolve_clarification_reply(meta=meta, reply="uploaded-2")

    assert result.decision == "valid"
    assert result.normalized_value == "uploaded-2"
    assert result.target_node == "generate_code"


def test_dataset_database_reply_reroutes_only_when_contract_allows(monkeypatch) -> None:
    def classify(**_kwargs):
        return {
            "decision": "reroute",
            "intent": "database_source",
            "target_node": "rag_db_qa",
            "normalized_value": None,
            "confidence": 0.91,
            "reason": "User asked for the database.",
        }

    monkeypatch.setattr(
        "graph.nodes.clarification_contracts.classify_clarification_reply",
        classify,
    )
    meta = {
        MetaKeys.CLARIFICATION_KIND: CLARIFICATION_KIND_DATASET_SELECTION,
        MetaKeys.CLARIFICATION_RETURN_NODE: "generate_code",
        MetaKeys.PENDING_QUESTION: "study loss to follow up",
        MetaKeys.CLARIFICATION_EXPECTED: build_expected(
            CLARIFICATION_KIND_DATASET_SELECTION,
            allowed_values=["uploaded-1"],
        ),
    }

    result = resolve_clarification_reply(meta=meta, reply="look at my databse")

    assert result.decision == "reroute"
    assert result.intent == "database_source"
    assert result.target_node == "rag_db_qa"


def test_generate_code_database_reply_reroutes_from_classifier(monkeypatch) -> None:
    monkeypatch.setattr(
        "graph.nodes.clarification_contracts.classify_clarification_reply",
        lambda **_kwargs: {
            "decision": "reroute",
            "intent": "database_source",
            "target_node": "rag_db_qa",
            "normalized_value": "database",
            "confidence": 0.92,
            "reason": "User wants the database source.",
        },
    )
    meta = {
        MetaKeys.CLARIFICATION_KIND: CLARIFICATION_KIND_GENERATE_CODE,
        MetaKeys.CLARIFICATION_RETURN_NODE: "generate_code",
        MetaKeys.PENDING_QUESTION: "study loss to follow up",
        MetaKeys.CLARIFICATION_EXPECTED: build_expected(CLARIFICATION_KIND_GENERATE_CODE),
    }

    for reply in (
        "query my database",
        "help me to retrivel from my database",
        "look at my databse",
    ):
        result = resolve_clarification_reply(meta=meta, reply=reply)

        assert result.decision == "reroute"
        assert result.intent == "database_source"
        assert result.target_node == "rag_db_qa"


def test_generate_code_reroutable_free_text_reasks_when_classifier_unclear(monkeypatch) -> None:
    monkeypatch.setattr(
        "graph.nodes.clarification_contracts.classify_clarification_reply",
        lambda **_kwargs: {
            "decision": "unclear",
            "intent": "unknown",
            "target_node": None,
            "normalized_value": None,
            "confidence": 0.0,
            "reason": "classifier unavailable",
        },
    )
    meta = {
        MetaKeys.CLARIFICATION_KIND: CLARIFICATION_KIND_GENERATE_CODE,
        MetaKeys.CLARIFICATION_RETURN_NODE: "generate_code",
        MetaKeys.PENDING_QUESTION: "study loss to follow up",
        MetaKeys.CLARIFICATION_EXPECTED: build_expected(CLARIFICATION_KIND_GENERATE_CODE),
    }

    result = resolve_clarification_reply(meta=meta, reply="the database I am looking at is xyz")

    assert result.decision == "unclear"
    assert result.target_node is None


def test_classifier_cannot_invent_dataset_id(monkeypatch) -> None:
    def classify(**_kwargs):
        return {
            "decision": "valid",
            "intent": "uploaded_dataset_choice",
            "target_node": "generate_code",
            "normalized_value": "uploaded-404",
            "confidence": 0.99,
            "reason": "Bad invented ID.",
        }

    monkeypatch.setattr(
        "graph.nodes.clarification_contracts.classify_clarification_reply",
        classify,
    )
    meta = {
        MetaKeys.CLARIFICATION_KIND: CLARIFICATION_KIND_DATASET_SELECTION,
        MetaKeys.CLARIFICATION_RETURN_NODE: "generate_code",
        MetaKeys.CLARIFICATION_EXPECTED: build_expected(
            CLARIFICATION_KIND_DATASET_SELECTION,
            allowed_values=["uploaded-1"],
        ),
    }

    result = resolve_clarification_reply(meta=meta, reply="the other data")

    assert result.decision == "unclear"
    assert result.target_node is None
    assert result.normalized_value is None


def test_yes_no_contract_accepts_yes_and_no() -> None:
    meta = {
        MetaKeys.CLARIFICATION_KIND: CLARIFICATION_KIND_RAG_DB_EXTRACTION_OPT_IN,
        MetaKeys.CLARIFICATION_RETURN_NODE: "rag_db_qa",
        MetaKeys.CLARIFICATION_EXPECTED: build_expected(
            CLARIFICATION_KIND_RAG_DB_EXTRACTION_OPT_IN
        ),
    }

    assert resolve_clarification_reply(meta=meta, reply="yes").normalized_value == "yes"
    assert resolve_clarification_reply(meta=meta, reply="no").normalized_value == "no"


def test_free_text_generate_code_reply_resumes_owner(monkeypatch) -> None:
    def classify(**_kwargs):
        return {
            "decision": "valid",
            "intent": "analysis_instruction",
            "target_node": "generate_code",
            "normalized_value": "Use marital status as x",
            "confidence": 0.9,
            "reason": "Substantive generate-code instruction.",
        }

    monkeypatch.setattr(
        "graph.nodes.clarification_contracts.classify_clarification_reply",
        classify,
    )
    meta = {
        MetaKeys.CLARIFICATION_KIND: CLARIFICATION_KIND_GENERATE_CODE,
        MetaKeys.CLARIFICATION_RETURN_NODE: "generate_code",
        MetaKeys.CLARIFICATION_EXPECTED: build_expected(CLARIFICATION_KIND_GENERATE_CODE),
    }

    result = resolve_clarification_reply(meta=meta, reply="Use marital status as x")

    assert result.decision == "valid"
    assert result.normalized_value == "Use marital status as x"
    assert result.target_node == "generate_code"


def test_qa_tool_requires_all_fields() -> None:
    meta = {
        MetaKeys.CLARIFICATION_KIND: CLARIFICATION_KIND_QA_TOOL,
        MetaKeys.CLARIFICATION_RETURN_NODE: "qa",
        MetaKeys.CLARIFICATION_EXPECTED: build_expected(
            CLARIFICATION_KIND_QA_TOOL,
            tool_name="query_weather",
            required_fields=["city"],
        ),
    }

    result = resolve_clarification_reply(meta=meta, reply="Boston")

    assert result.decision == "valid"
    assert result.normalized_value == {"city": "Boston"}
    assert result.target_node == "qa"


def test_contract_for_kind_rejects_removed_kinds() -> None:
    for stale_kind in ("data_source_selection", "planner_route_clarification", "qa_followup"):
        assert contract_for_kind(stale_kind) is None
