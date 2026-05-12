from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph.memory import upsert_user_intent_from_db_rag_intent
from graph.state import MetaKeys


class _HumanMessage:
    type = "human"

    def __init__(self, content: str, id: str | None = None):
        self.content = content
        self.id = id or ""


class _AIMessage:
    type = "ai"

    def __init__(self, content: str, additional_kwargs: dict | None = None):
        self.content = content
        self.additional_kwargs = dict(additional_kwargs or {})


class _FormattedPrompt:
    def __init__(self, rendered):
        self._rendered = rendered

    def to_messages(self):
        return self._rendered


class _PromptTemplate:
    def __init__(self, messages):
        self._messages = [m for m in messages if isinstance(m, tuple)]

    def format_prompt(self, **kwargs):
        rendered = []
        for role, template in self._messages:
            rendered.append({"role": role, "content": template.format(**kwargs)})
        return _FormattedPrompt(rendered)


class _ChatPromptTemplate:
    @staticmethod
    def from_messages(messages):
        return _PromptTemplate(messages)


class _MessagesPlaceholder:
    def __init__(self, variable_name: str, optional: bool = False) -> None:
        self.variable_name = variable_name
        self.optional = optional


class _FewShotChatMessagePromptTemplate:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _LLM:
    def invoke(self, _messages):
        return SimpleNamespace(
            content=json.dumps({"action": "qa", "thought": "test planner fallback"})
        )


class _ClarifyingPlannerLLM:
    def invoke(self, _messages):
        return SimpleNamespace(
            content=json.dumps(
                {
                    "action": "generate_code",
                    "thought": "Need dataset binding before code generation.",
                    "ranked_actions": ["generate_code", "rag_db_qa", "qa"],
                    "needs_clarification": True,
                    "clarification_question": "Which dataset or file should I use?",
                }
            )
        )


def _install_langchain_stubs() -> None:
    messages_mod = ModuleType("langchain_core.messages")
    messages_mod.BaseMessage = object
    messages_mod.HumanMessage = _HumanMessage
    messages_mod.AIMessage = _AIMessage

    prompts_mod = ModuleType("langchain_core.prompts")
    prompts_mod.ChatPromptTemplate = _ChatPromptTemplate
    prompts_mod.MessagesPlaceholder = _MessagesPlaceholder
    prompts_mod.FewShotChatMessagePromptTemplate = _FewShotChatMessagePromptTemplate

    langchain_core_mod = ModuleType("langchain_core")
    langchain_core_mod.messages = messages_mod
    langchain_core_mod.prompts = prompts_mod

    graph_message_mod = ModuleType("langgraph.graph.message")
    graph_message_mod.add_messages = lambda current, new: (current or []) + (new or [])

    sys.modules["langchain_core"] = langchain_core_mod
    sys.modules["langchain_core.messages"] = messages_mod
    sys.modules["langchain_core.prompts"] = prompts_mod
    sys.modules["langgraph.graph.message"] = graph_message_mod


def _fresh_node_module():
    _install_langchain_stubs()
    for mod in (
        "prompts.planner_prompt",
        "graph.nodes.orchestrator",
        "graph.nodes.orchestrator.node",
        "graph.nodes.orchestrator.planner",
        "graph.nodes.orchestrator.policy",
        "graph.nodes.orchestrator.action_mask",
        "graph.nodes.orchestrator.context_builder",
    ):
        sys.modules.pop(mod, None)
    return importlib.import_module("graph.nodes.orchestrator.node")


def _base_state(user_message: str = "continue previous query") -> dict[str, Any]:
    return {
        "messages": [_HumanMessage(user_message, id="turn-1")],
        "output": {},
        "artifacts": {
            "files": {},
            "datasets": {},
            "conversation_events": [],
            "conversation_events_version": 1,
            "artifact_manifest_version": 1,
        },
        "next_action": None,
        "last_action": None,
        "observations": [],
        "orchestrator": {},
        "planner": {},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {
                "before_run_decision": None,
                "after_error_decision": None,
                "final_decision": None,
            },
            "qa": {},
            "generate_code": {},
            "rag_db_qa": {"thread_status": "done", "active_thread": False},
        },
        "node_data": {},
        "meta": {"error_iterations": 0, "workflow_trace": []},
    }


def _state_with_cancelled_db_rag_intent(
    user_message: str = "continue previous query",
) -> dict[str, Any]:
    return upsert_user_intent_from_db_rag_intent(
        _base_state(user_message),
        active_intent={
            "intent_id": "rag-intent-1",
            "source_question": "Which diagnoses have the highest average age?",
            "goal_text": "Find diagnoses ranked by average age",
        },
        source_message_hash="original-user-hash",
        status="cancelled",
    )


def test_apply_resolved_user_intent_meta_uses_stored_source_question() -> None:
    node = _fresh_node_module()
    intent = {
        "intent_id": "intent-1",
        "kind": "db_rag_query",
        "source_question": "original database question",
    }
    resolution = {
        "target_id": "intent-1",
        "kind": "db_rag_query",
        "relationship": "continue",
        "source_question": "classifier paraphrase",
    }

    meta = node._apply_resolved_user_intent_meta({}, resolution, intent, "hash-1")

    assert meta[MetaKeys.RESOLVED_USER_INTENT_SOURCE_QUESTION] == "original database question"
    assert meta[MetaKeys.RESOLVED_USER_INTENT_ID] == "intent-1"
    assert meta[MetaKeys.RESOLVED_USER_INTENT_KIND] == "db_rag_query"
    assert meta[MetaKeys.RESOLVED_USER_INTENT_RELATIONSHIP] == "continue"
    assert meta[MetaKeys.RESOLVED_USER_INTENT_USER_MESSAGE_HASH] == "hash-1"


def test_route_from_resolved_user_intent_meta_routes_db_rag_once() -> None:
    node = _fresh_node_module()
    state = _state_with_cancelled_db_rag_intent()
    intent_id = state["memory"]["last_user_intent_id"]
    meta = {
        MetaKeys.RESOLVED_USER_INTENT_ID: intent_id,
        MetaKeys.RESOLVED_USER_INTENT_KIND: "db_rag_query",
        MetaKeys.RESOLVED_USER_INTENT_RELATIONSHIP: "refine",
        MetaKeys.RESOLVED_USER_INTENT_SOURCE_QUESTION: "BAD",
        MetaKeys.RESOLVED_USER_INTENT_USER_MESSAGE_HASH: "hash-1",
    }

    routed, updated_meta, observations = node._route_from_resolved_user_intent_meta(
        state,
        meta,
        {"rag_db_qa", "qa", "end"},
    )
    routed_again, cleared_meta, second_observations = node._route_from_resolved_user_intent_meta(
        state,
        updated_meta,
        {"rag_db_qa", "qa", "end"},
    )

    assert routed == "rag_db_qa"
    assert updated_meta[MetaKeys.RAG_DB_QUESTION_OVERRIDE] == (
        "Which diagnoses have the highest average age?"
    )
    assert updated_meta["resolved_user_intent_meta_consumed"] == "hash-1"
    assert observations == [
        f"user_intent_id={intent_id} relationship=refine routed_node=rag_db_qa"
    ]
    assert routed_again is None
    assert MetaKeys.RESOLVED_USER_INTENT_ID not in cleared_meta
    assert MetaKeys.RAG_DB_QUESTION_OVERRIDE not in cleared_meta
    assert second_observations == []


def test_route_from_resolved_user_intent_meta_clears_blank_memory_source_question() -> None:
    node = _fresh_node_module()
    state = _state_with_cancelled_db_rag_intent()
    intent_id = state["memory"]["last_user_intent_id"]
    state["memory"]["user_intents"][intent_id]["source_question"] = " "
    meta = {
        MetaKeys.RESOLVED_USER_INTENT_ID: intent_id,
        MetaKeys.RESOLVED_USER_INTENT_KIND: "db_rag_query",
        MetaKeys.RESOLVED_USER_INTENT_RELATIONSHIP: "continue",
        MetaKeys.RESOLVED_USER_INTENT_SOURCE_QUESTION: "meta should not win",
        MetaKeys.RESOLVED_USER_INTENT_USER_MESSAGE_HASH: "hash-1",
        MetaKeys.RAG_DB_QUESTION_OVERRIDE: "stale override",
        "keep": "value",
    }

    routed, updated_meta, observations = node._route_from_resolved_user_intent_meta(
        state,
        meta,
        {"rag_db_qa", "qa", "end"},
    )

    assert routed is None
    assert updated_meta == {"keep": "value"}
    assert observations == []


def test_route_from_resolved_user_intent_meta_clears_missing_intent() -> None:
    node = _fresh_node_module()
    meta = {
        MetaKeys.RESOLVED_USER_INTENT_ID: "intent-missing",
        MetaKeys.RESOLVED_USER_INTENT_KIND: "db_rag_query",
        MetaKeys.RESOLVED_USER_INTENT_RELATIONSHIP: "continue",
        MetaKeys.RESOLVED_USER_INTENT_SOURCE_QUESTION: "original database question",
        MetaKeys.RESOLVED_USER_INTENT_USER_MESSAGE_HASH: "hash-1",
        MetaKeys.RAG_DB_QUESTION_OVERRIDE: "original database question",
        "keep": "value",
    }

    routed, updated_meta, observations = node._route_from_resolved_user_intent_meta(
        _base_state(),
        meta,
        {"rag_db_qa", "qa", "end"},
    )

    assert routed is None
    assert updated_meta == {"keep": "value"}
    assert observations == []


def test_route_from_resolved_user_intent_meta_clears_unsupported_intent_kind() -> None:
    node = _fresh_node_module()
    state = _base_state()
    state["memory"] = {
        "user_intents": {
            "intent-1": {
                "intent_id": "intent-1",
                "kind": "other_kind",
                "source_question": "original database question",
            }
        },
        "last_user_intent_id": "intent-1",
    }
    meta = {
        MetaKeys.RESOLVED_USER_INTENT_ID: "intent-1",
        MetaKeys.RESOLVED_USER_INTENT_KIND: "db_rag_query",
        MetaKeys.RESOLVED_USER_INTENT_RELATIONSHIP: "continue",
        MetaKeys.RESOLVED_USER_INTENT_SOURCE_QUESTION: "original database question",
        MetaKeys.RESOLVED_USER_INTENT_USER_MESSAGE_HASH: "hash-1",
        MetaKeys.RAG_DB_QUESTION_OVERRIDE: "original database question",
        "keep": "value",
    }

    routed, updated_meta, observations = node._route_from_resolved_user_intent_meta(
        state,
        meta,
        {"rag_db_qa"},
    )

    assert routed is None
    assert updated_meta == {"keep": "value"}
    assert observations == []


def test_clear_consumed_resolved_user_intent_meta_removes_handoff_keys() -> None:
    node = _fresh_node_module()
    meta = {
        MetaKeys.RESOLVED_USER_INTENT_ID: "intent-1",
        MetaKeys.RESOLVED_USER_INTENT_KIND: "db_rag_query",
        MetaKeys.RESOLVED_USER_INTENT_RELATIONSHIP: "continue",
        MetaKeys.RESOLVED_USER_INTENT_SOURCE_QUESTION: "original database question",
        MetaKeys.RESOLVED_USER_INTENT_USER_MESSAGE_HASH: "hash-1",
        MetaKeys.RAG_DB_QUESTION_OVERRIDE: "original database question",
        "resolved_user_intent_meta_consumed": "hash-1",
        "keep": "value",
    }

    cleared = node._clear_consumed_resolved_user_intent_meta(meta)

    assert cleared == {"keep": "value"}


def test_orchestrator_routes_existing_user_intent_reference_to_db_rag(monkeypatch) -> None:
    node = _fresh_node_module()
    state = _state_with_cancelled_db_rag_intent()
    intent_id = state["memory"]["last_user_intent_id"]
    calls = []

    def classify(selected_state, classifier, *, user_message, user_message_hash, **_kwargs):
        calls.append((selected_state, classifier, user_message, user_message_hash))
        return {
            "target": "existing_user_intent",
            "target_id": intent_id,
            "kind": "db_rag_query",
            "relationship": "continue",
            "source_question": "classifier should not win",
            "needs_clarification": False,
        }

    monkeypatch.setattr(node, "classify_user_intent_reference", classify)

    result = node.orchestrator_node(state, _LLM(), ["rag_db_qa", "qa", "end"])

    assert len(calls) == 1
    assert result["next_action"] == "rag_db_qa"
    assert result["meta"][MetaKeys.RESOLVED_USER_INTENT_ID] == intent_id
    assert result["meta"][MetaKeys.RAG_DB_QUESTION_OVERRIDE] == (
        "Which diagnoses have the highest average age?"
    )


def test_orchestrator_new_user_intent_classification_does_not_set_db_rag_override(
    monkeypatch,
) -> None:
    node = _fresh_node_module()
    state = _state_with_cancelled_db_rag_intent("query the database for a new cohort")

    def classify(_state, _classifier, *, user_message, user_message_hash, **_kwargs):
        return {
            "target": "new_user_intent",
            "target_id": None,
            "kind": "db_rag_query",
            "relationship": None,
            "needs_clarification": False,
        }

    monkeypatch.setattr(node, "classify_user_intent_reference", classify)

    result = node.orchestrator_node(state, _LLM(), ["rag_db_qa", "qa", "end"])

    assert MetaKeys.RAG_DB_QUESTION_OVERRIDE not in result["meta"]
    assert MetaKeys.RESOLVED_USER_INTENT_ID not in result["meta"]
    assert not any(
        str(observation).startswith("user_intent_id=")
        for observation in result.get("observations", [])
    )


def test_orchestrator_no_prior_intent_workflow_reference_does_not_route_to_db_rag(
    monkeypatch,
) -> None:
    node = _fresh_node_module()
    state = _base_state("continue previous query")

    def classify(_state, _classifier, *, user_message, user_message_hash, **_kwargs):
        return {
            "turn_type": "workflow_reference",
            "target": "none",
            "target_id": None,
            "relationship": "continue",
            "needs_clarification": False,
            "reason": "No prior intent.",
        }

    monkeypatch.setattr(node, "classify_user_intent_reference", classify)

    result = node.orchestrator_node(state, _LLM(), ["rag_db_qa", "qa", "end"])

    assert result["next_action"] == "end"
    assert "previous DB-RAG query" in result["output"]["qa_response"]
    assert MetaKeys.LAST_USER_MESSAGE_HASH in result["meta"]
    assert MetaKeys.RAG_DB_QUESTION_OVERRIDE not in result["meta"]


def test_orchestrator_routes_active_clarification_reply_to_clarification_node() -> None:
    node = _fresh_node_module()
    state = _base_state("help me to subset from my database")
    state["meta"][MetaKeys.AWAITING_USER_CLARIFICATION] = True
    state["meta"][MetaKeys.CLARIFICATION_KIND] = "generate_code"
    state["meta"][MetaKeys.CLARIFICATION_RETURN_NODE] = "generate_code"
    state["meta"][MetaKeys.PENDING_QUESTION] = (
        "I am trying to study factors associated with loss to follow up among index case, "
        "help me to subset factors related to marriage status, alcohol usage, and diabetes"
    )

    result = node.orchestrator_node(
        state,
        _LLM(),
        ["clarification", "generate_code", "rag_db_qa", "qa", "end"],
    )

    assert result["next_action"] == "clarification"
    assert result["meta"][MetaKeys.PENDING_QUESTION] == (
        "I am trying to study factors associated with loss to follow up among index case, "
        "help me to subset factors related to marriage status, alcohol usage, and diabetes"
    )
    assert result["meta"][MetaKeys.AWAITING_USER_CLARIFICATION] is True


def test_planner_clarification_does_not_create_hard_clarification_state() -> None:
    node = _fresh_node_module()
    state = _base_state(
        "I am trying to study factors associated with loss to follow up among index case"
    )

    result = node.orchestrator_node(
        state,
        _ClarifyingPlannerLLM(),
        ["clarification", "generate_code", "rag_db_qa", "qa", "end"],
    )

    assert result["next_action"] == "end"
    assert MetaKeys.AWAITING_USER_CLARIFICATION not in result["meta"]
    assert MetaKeys.CLARIFICATION_KIND not in result["meta"]
    assert MetaKeys.PENDING_QUESTION not in result["meta"]
    assert result["output"]["qa_response"] == "Which dataset or file should I use?"
    assert "task_id=None relationship=None routed_node=end" in result["observations"]


def test_orchestrator_uses_source_classifier_for_no_dataset_database_request(monkeypatch) -> None:
    node = _fresh_node_module()
    policy_module = sys.modules["graph.nodes.orchestrator.policy"]
    state = _base_state(
        "I am trying to study factors associated with loss to follow up among index case, "
        "help me to subset factors related to marriage status, alcohol usage, and diabetes"
    )
    calls = {}

    def classify(**kwargs):
        calls["kwargs"] = kwargs
        return {"label": "database", "confidence": 0.9}

    monkeypatch.setattr(policy_module, "classify_database_source_intent", classify)

    result = node.orchestrator_node(
        state,
        _LLM(),
        ["clarification", "generate_code", "rag_db_qa", "qa", "end"],
    )

    assert calls["kwargs"]["user_reply"].startswith("i am trying to study factors")
    assert result["next_action"] == "rag_db_qa"
    assert MetaKeys.AWAITING_USER_CLARIFICATION not in result["meta"]


def test_orchestrator_completed_task_classification_does_not_hijack_user_intent(
    monkeypatch,
) -> None:
    node = _fresh_node_module()
    state = _state_with_cancelled_db_rag_intent("explain the completed extraction")

    def classify(_state, _classifier, *, user_message, user_message_hash, **_kwargs):
        return {
            "target": "completed_task",
            "target_id": "task-1",
            "kind": "db_rag_sql_extraction",
            "relationship": "explain",
            "needs_clarification": False,
        }

    monkeypatch.setattr(node, "classify_user_intent_reference", classify)

    result = node.orchestrator_node(state, _LLM(), ["rag_db_qa", "qa", "end"])

    assert MetaKeys.RAG_DB_QUESTION_OVERRIDE not in result["meta"]
    assert MetaKeys.RESOLVED_USER_INTENT_ID not in result["meta"]
    assert not any(
        str(observation).startswith("user_intent_id=")
        for observation in result.get("observations", [])
    )


def test_orchestrator_classifier_error_falls_through_to_deterministic_db_rag_routing(
    monkeypatch,
) -> None:
    node = _fresh_node_module()
    state = _state_with_cancelled_db_rag_intent("query the database for a new cohort")

    def classify(_state, _classifier, *, user_message, user_message_hash, **_kwargs):
        raise ValueError("bad classifier json")

    monkeypatch.setattr(node, "classify_user_intent_reference", classify)

    result = node.orchestrator_node(state, _LLM(), ["rag_db_qa", "qa", "end"])

    assert MetaKeys.RAG_DB_QUESTION_OVERRIDE not in result["meta"]
    assert MetaKeys.RESOLVED_USER_INTENT_ID not in result["meta"]
    assert result["next_action"] == "rag_db_qa"
    assert "user_intent_classifier_error=bad classifier json" in result["observations"]


def test_orchestrator_asks_normal_followup_for_ambiguous_user_intent(monkeypatch) -> None:
    node = _fresh_node_module()
    state = _state_with_cancelled_db_rag_intent("do that previous thing")

    def classify(_state, _classifier, *, user_message, user_message_hash, **_kwargs):
        return {
            "target": "ambiguous",
            "target_id": None,
            "relationship": None,
            "needs_clarification": True,
            "reason": "unclear",
        }

    monkeypatch.setattr(node, "classify_user_intent_reference", classify)

    result = node.orchestrator_node(
        state,
        _LLM(),
        ["clarification", "rag_db_qa", "qa", "end"],
    )

    assert result["next_action"] == "end"
    assert MetaKeys.AWAITING_USER_CLARIFICATION not in result["meta"]
    assert MetaKeys.CLARIFICATION_KIND not in result["meta"]
    assert result["output"]["qa_response"] == (
        "Which previous database query did you want to continue?"
    )
    clarification_events = [
        event
        for event in result["artifacts"]["conversation_events"]
        if event["type"] == "clarification"
    ]
    assert len(clarification_events) == 1
    assert clarification_events[0]["actor"] == "orchestrator"
    assert clarification_events[0]["text"] == result["output"]["qa_response"]
    assert clarification_events[0]["status"] == "active"


def test_rag_db_dispatch_passes_and_clears_question_override() -> None:
    _install_langchain_stubs()
    langgraph_mod = ModuleType("langgraph.graph")
    langgraph_mod.StateGraph = object
    langgraph_mod.START = "START"
    langgraph_mod.END = "END"
    sqlite_mod = ModuleType("langgraph.checkpoint.sqlite")
    sqlite_mod.SqliteSaver = object
    langgraph_pkg = ModuleType("langgraph")
    langgraph_types_mod = ModuleType("langgraph.types")
    langgraph_types_mod.interrupt = lambda payload=None: payload
    sys.modules["langgraph"] = langgraph_pkg
    sys.modules["langgraph.graph"] = langgraph_mod
    sys.modules["langgraph.types"] = langgraph_types_mod
    sys.modules["langgraph.checkpoint.sqlite"] = sqlite_mod
    mcp_pool_mod = ModuleType("tools.mcp_pool")
    mcp_pool_mod.call_mcp_tool_sync = lambda *_args, **_kwargs: None
    sys.modules["tools.mcp_pool"] = mcp_pool_mod
    sys.modules.pop("graph.builder", None)

    from graph.builder import _dispatch_rag_db_qa_with_override

    calls = {}
    state = {
        "meta": {
            MetaKeys.RAG_DB_QUESTION_OVERRIDE: "original database question",
            "keep": "value",
        }
    }

    def rag_db_qa_node(
        received_state,
        llm,
        *,
        provider,
        service,
        reranker_model=None,
        question_override=None,
    ):
        calls["question_override"] = question_override
        return {
            **received_state,
            "meta": {
                **received_state["meta"],
                MetaKeys.RAG_DB_QUESTION_OVERRIDE: "original database question",
            },
        }

    updated = _dispatch_rag_db_qa_with_override(
        state,
        llm=object(),
        provider="openai",
        service=object(),
        reranker_model="reranker",
        rag_db_qa_node_fn=rag_db_qa_node,
    )

    assert calls["question_override"] == "original database question"
    assert updated["meta"] == {"keep": "value"}
