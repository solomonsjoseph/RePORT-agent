from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _dataset_state(*, analysis_dataset_id: str | None = None) -> dict:
    meta = {
        "last_user_message_hash": "u1",
    }
    if analysis_dataset_id is not None:
        meta["analysis_dataset_id"] = analysis_dataset_id
    return {
        "messages": [],
        "output": {},
        "agents": {"generate_code": {}, "human_review": {}, "executor": {}},
        "meta": meta,
        "artifacts": {
            "datasets": {
                "uploaded-1": {
                    "id": "uploaded-1",
                    "kind": "uploaded",
                    "row_count": 10,
                    "column_count": 2,
                    "provenance": {"source": "upload"},
                },
                "subset-1": {
                    "id": "subset-1",
                    "kind": "subset",
                    "row_count": 4,
                    "column_count": 3,
                    "provenance": {"source": "db_rag_sql"},
                },
            },
            "active_dataset_id": "subset-1",
            "files": {},
            "conversation_events": [],
            "conversation_events_version": 1,
            "artifact_manifest_version": 1,
        },
    }


def test_choose_analysis_dataset_requires_explicit_selection_when_multiple_exist() -> None:
    from utils.dataset_artifacts import choose_analysis_dataset

    artifact, reason = choose_analysis_dataset(_dataset_state(), latest_user_message="analyze this")

    assert artifact is None
    assert reason == "ambiguous"


def test_choose_analysis_dataset_uses_explicit_analysis_dataset_id() -> None:
    from utils.dataset_artifacts import choose_analysis_dataset

    artifact, reason = choose_analysis_dataset(
        _dataset_state(analysis_dataset_id="uploaded-1"),
        latest_user_message="analyze this",
    )

    assert artifact is not None
    assert artifact["id"] == "uploaded-1"
    assert reason == "explicit"


def test_get_analysis_dataset_artifact_prefers_pinned_analysis_dataset_id() -> None:
    from utils.dataset_artifacts import get_analysis_dataset_artifact

    artifact = get_analysis_dataset_artifact(_dataset_state(analysis_dataset_id="uploaded-1"))

    assert artifact is not None
    assert artifact["id"] == "uploaded-1"


def test_build_active_dataset_artifacts_patch_sets_global_active_dataset() -> None:
    from utils.dataset_artifacts import build_active_dataset_artifacts_patch

    updated = build_active_dataset_artifacts_patch(_dataset_state()["artifacts"], "uploaded-1")

    assert updated["active_dataset_id"] == "uploaded-1"
    assert set(updated["datasets"]) == {"uploaded-1", "subset-1"}


def test_build_active_dataset_artifacts_patch_rejects_unknown_id() -> None:
    from utils.dataset_artifacts import build_active_dataset_artifacts_patch

    with pytest.raises(KeyError):
        build_active_dataset_artifacts_patch(_dataset_state()["artifacts"], "missing-id")


_STUBBED_GENERATE_CODE_MODULES = (
    "langchain_core",
    "langchain_core.messages",
    "langchain_core.prompts",
    "langgraph.graph.message",
    "prompts.generate_prompt",
    "utils.llm_response",
    "utils.message_window",
    "graph.nodes.code_guardrails",
    "graph.nodes.generate_code",
)


@pytest.fixture
def generate_code_module():
    original = {name: sys.modules.get(name) for name in _STUBBED_GENERATE_CODE_MODULES}
    try:
        langchain_core = ModuleType("langchain_core")
        messages = ModuleType("langchain_core.messages")

        class _BaseMessage:
            def __init__(self, content: str, id: str | None = None):
                self.content = content
                self.id = id
                self.additional_kwargs = {}

        class HumanMessage(_BaseMessage):
            pass

        class AIMessage(_BaseMessage):
            pass

        messages.BaseMessage = _BaseMessage
        messages.HumanMessage = HumanMessage
        messages.AIMessage = AIMessage
        langchain_core.messages = messages

        prompts = ModuleType("langchain_core.prompts")

        class ChatPromptTemplate:
            @classmethod
            def from_messages(cls, messages):
                return SimpleNamespace(invoke=lambda payload: payload)

        class MessagesPlaceholder:
            def __init__(self, variable_name: str):
                self.variable_name = variable_name

        prompts.ChatPromptTemplate = ChatPromptTemplate
        prompts.MessagesPlaceholder = MessagesPlaceholder
        langchain_core.prompts = prompts
        sys.modules["langchain_core"] = langchain_core
        sys.modules["langchain_core.messages"] = messages
        sys.modules["langchain_core.prompts"] = prompts
        graph_message_mod = ModuleType("langgraph.graph.message")
        graph_message_mod.add_messages = lambda current, new: (current or []) + (new or [])
        sys.modules["langgraph.graph.message"] = graph_message_mod

        prompt_module = ModuleType("prompts.generate_prompt")
        prompt_module.make_generate_code_prompt = lambda: SimpleNamespace(invoke=lambda payload: payload)
        sys.modules["prompts.generate_prompt"] = prompt_module

        llm_response = ModuleType("utils.llm_response")
        llm_response.coerce_text_content = lambda content: content
        sys.modules["utils.llm_response"] = llm_response

        message_window = ModuleType("utils.message_window")
        message_window.window_messages = lambda messages, max_turns: list(messages)
        sys.modules["utils.message_window"] = message_window

        code_guardrails = ModuleType("graph.nodes.code_guardrails")
        code_guardrails.code_fingerprint = lambda code: f"hash:{code}"
        code_guardrails.is_executable_python = lambda code: bool(code.strip())
        sys.modules["graph.nodes.code_guardrails"] = code_guardrails

        sys.modules.pop("graph.nodes.generate_code", None)
        module = importlib.import_module("graph.nodes.generate_code")

        def _append_event(state, event):
            artifacts = dict(state.get("artifacts") or {})
            events = list(artifacts.get("conversation_events") or [])
            event = dict(event)
            event.setdefault("event_id", f"evt-{len(events)+1}")
            events.append(event)
            artifacts["conversation_events"] = events
            return {**state, "artifacts": artifacts}

        def _store_artifact(state, record):
            artifacts = dict(state.get("artifacts") or {})
            files = dict(artifacts.get("files") or {})
            artifact_id = f"artifact-{len(files)+1}"
            files[artifact_id] = {"artifact_id": artifact_id, **record}
            artifacts["files"] = files
            return {**state, "artifacts": artifacts}

        module.append_conversation_event = _append_event
        module.store_thread_artifact = _store_artifact
        module.build_assistant_event = lambda **kwargs: kwargs
        module.build_code_event = lambda **kwargs: kwargs
        module.build_clarification_event = lambda **kwargs: kwargs
        module.build_dataset_context = lambda artifact: f"context::{artifact['id']}"
        yield module, HumanMessage, AIMessage
    finally:
        for name, module in original.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_generate_code_asks_for_exact_dataset_id_when_multiple_candidates_exist(generate_code_module) -> None:
    module, HumanMessage, _AIMessage = generate_code_module
    state = _dataset_state()
    state["messages"] = [HumanMessage("plot age by sex")]

    updated = module.generate_code_node(state, SimpleNamespace(invoke=lambda _prompt: None), {"runtime_datasets": True})

    assert updated["meta"]["clarification_kind"] == "generate_code_dataset_selection"
    assert updated["meta"]["analysis_dataset_candidate_ids"] == ["uploaded-1", "subset-1"]
    assert updated["meta"]["analysis_dataset_pending_request"] == "plot age by sex"
    assert "Reply with exactly one dataset ID" in updated["output"]["qa_response"]
    assert "uploaded-1" in updated["output"]["qa_response"]
    assert "subset-1" in updated["output"]["qa_response"]


def test_generate_code_accepts_exact_dataset_id_reply_and_pins_analysis_selection(generate_code_module) -> None:
    module, HumanMessage, AIMessage = generate_code_module

    class FakeLLM:
        def __init__(self):
            self.prompts = []

        def invoke(self, prompt):
            self.prompts.append(prompt)
            return SimpleNamespace(
                content='{"response_type":"code_result","summary":"done","assumptions":"","code":"print(1)"}'
            )

    llm = FakeLLM()
    state = _dataset_state()
    state["messages"] = [
        HumanMessage("plot age by sex"),
        AIMessage("Multiple datasets are available."),
        HumanMessage("subset-1"),
    ]
    state["meta"].update(
        {
            "analysis_dataset_candidate_ids": ["uploaded-1", "subset-1"],
            "analysis_dataset_pending_request": "plot age by sex",
        }
    )

    updated = module.generate_code_node(
        state,
        llm,
        {"runtime_datasets": True},
        question_override="subset-1",
    )

    assert "context::subset-1" in llm.prompts[0]["context"]
    assert 'datasets["subset-1"]' in llm.prompts[0]["context"]
    assert updated["meta"]["analysis_dataset_id"] == "subset-1"
    assert "analysis_dataset_candidate_ids" not in updated["meta"]
    assert "analysis_dataset_pending_request" not in updated["meta"]
    assert updated["output"]["generated_code"] == "print(1)"


def test_generate_code_records_selected_dataset_id_in_assumptions(generate_code_module) -> None:
    module, HumanMessage, _AIMessage = generate_code_module

    class FakeLLM:
        def invoke(self, _prompt):
            return SimpleNamespace(
                content=(
                    '{"response_type":"code_result","summary":"done",'
                    '"assumptions":"Gender values are cleaned before tabulation.",'
                    '"code":"print(1)"}'
                )
            )

    state = _dataset_state(analysis_dataset_id="subset-1")
    state["messages"] = [HumanMessage("tabulate gender")]

    updated = module.generate_code_node(state, FakeLLM(), {"runtime_datasets": True})

    assumptions = updated["output"]["code_assumptions"]
    assert "Gender values are cleaned before tabulation." in assumptions
    assert "Dataset used: subset-1" in assumptions


def test_generate_code_prompt_context_names_dataset_mapping(generate_code_module) -> None:
    module, HumanMessage, _AIMessage = generate_code_module

    class FakeLLM:
        def __init__(self):
            self.prompts = []

        def invoke(self, prompt):
            self.prompts.append(prompt)
            return SimpleNamespace(
                content='{"response_type":"code_result","summary":"done","assumptions":"","code":"print(1)"}'
            )

    llm = FakeLLM()
    state = _dataset_state(analysis_dataset_id="subset-1")
    state["messages"] = [HumanMessage("tabulate gender")]

    module.generate_code_node(state, llm, {"runtime_datasets": True})

    assert 'datasets["subset-1"]' in llm.prompts[0]["context"]


def test_generate_code_reprompts_when_dataset_reply_is_not_exact_id(generate_code_module) -> None:
    module, HumanMessage, AIMessage = generate_code_module
    state = _dataset_state()
    state["messages"] = [
        HumanMessage("plot age by sex"),
        AIMessage("Multiple datasets are available."),
        HumanMessage("the subset"),
    ]
    state["meta"].update(
        {
            "analysis_dataset_candidate_ids": ["uploaded-1", "subset-1"],
            "analysis_dataset_pending_request": "plot age by sex",
        }
    )

    updated = module.generate_code_node(
        state,
        SimpleNamespace(invoke=lambda _prompt: None),
        {"runtime_datasets": True},
        question_override="the subset",
    )

    assert updated["meta"]["clarification_kind"] == "generate_code_dataset_selection"
    assert updated["meta"]["analysis_dataset_pending_request"] == "plot age by sex"
    assert "Reply with exactly one dataset ID" in updated["output"]["qa_response"]


def test_clarification_node_routes_dataset_id_contract_to_generate_code(monkeypatch) -> None:
    from langchain_core.messages import HumanMessage
    from graph.nodes import clarification as module
    from graph.nodes.clarification_contracts import (
        CLARIFICATION_KIND_DATASET_SELECTION,
        build_expected,
    )
    from graph.state import MetaKeys

    called = {}

    def _generate_code_node(state, llm, context, question_override=None):
        called["question_override"] = question_override
        called["analysis_dataset_id"] = state["meta"].get(MetaKeys.ANALYSIS_DATASET_ID)
        return state

    monkeypatch.setattr(module, "generate_code_node", _generate_code_node)
    state = {
        "messages": [HumanMessage(content="uploaded-1")],
        "meta": {
            MetaKeys.CLARIFICATION_KIND: CLARIFICATION_KIND_DATASET_SELECTION,
            MetaKeys.CLARIFICATION_RETURN_NODE: "generate_code",
            MetaKeys.PENDING_QUESTION: "plot age by sex",
            MetaKeys.CLARIFICATION_EXPECTED: build_expected(
                CLARIFICATION_KIND_DATASET_SELECTION,
                allowed_values=["uploaded-1"],
            ),
            MetaKeys.CLARIFICATION_ATTEMPT_COUNT: 0,
        },
    }

    module.clarification_node(state, SimpleNamespace(), context={"runtime_datasets": True})

    assert called["question_override"] == "plot age by sex"
    assert called["analysis_dataset_id"] == "uploaded-1"


def test_clarification_node_reroutes_dataset_contract_database_reply_to_db_rag(monkeypatch) -> None:
    from langchain_core.messages import HumanMessage
    from graph.nodes import clarification as module
    from graph.nodes.clarification_contracts import (
        CLARIFICATION_KIND_DATASET_SELECTION,
        build_expected,
    )
    from graph.state import MetaKeys

    monkeypatch.setattr(
        "graph.nodes.clarification_contracts.classify_clarification_reply",
        lambda **_kwargs: {
            "decision": "reroute",
            "intent": "database_source",
            "target_node": "rag_db_qa",
            "normalized_value": None,
            "confidence": 0.9,
            "reason": "database source",
        },
    )
    called = {}

    def _rag_db_qa_node(state, llm, provider="", service=None, reranker_model=None, question_override=None):
        called["question_override"] = question_override
        called["source_hash"] = state["meta"].get(MetaKeys.RAG_DB_SOURCE_MESSAGE_HASH)
        return state

    monkeypatch.setattr(module, "rag_db_qa_node", _rag_db_qa_node)
    state = {
        "messages": [HumanMessage(content="use the study DB")],
        "meta": {
            MetaKeys.CLARIFICATION_KIND: CLARIFICATION_KIND_DATASET_SELECTION,
            MetaKeys.CLARIFICATION_RETURN_NODE: "generate_code",
            MetaKeys.PENDING_QUESTION: "study loss to follow-up among index cases",
            MetaKeys.PENDING_QUESTION_USER_MESSAGE_HASH: "original-hash",
            MetaKeys.CLARIFICATION_EXPECTED: build_expected(
                CLARIFICATION_KIND_DATASET_SELECTION,
                allowed_values=["uploaded-1"],
            ),
            MetaKeys.CLARIFICATION_ATTEMPT_COUNT: 0,
        },
    }

    module.clarification_node(
        state,
        SimpleNamespace(),
        context={"provider": "openai", "db_rag_service": object()},
    )

    assert called["question_override"] == "study loss to follow-up among index cases"
    assert called["source_hash"] == "original-hash"


def test_clarification_node_reasks_unclear_dataset_contract_reply(monkeypatch) -> None:
    from langchain_core.messages import HumanMessage
    from graph.nodes import clarification as module
    from graph.nodes.clarification_contracts import (
        CLARIFICATION_KIND_DATASET_SELECTION,
        build_expected,
    )
    from graph.state import MetaKeys

    monkeypatch.setattr(
        "graph.nodes.clarification_contracts.classify_clarification_reply",
        lambda **_kwargs: {
            "decision": "unclear",
            "intent": "unknown",
            "target_node": None,
            "normalized_value": None,
            "confidence": 0.0,
            "reason": "ambiguous",
        },
    )
    state = {
        "messages": [HumanMessage(content="that one")],
        "output": {},
        "artifacts": {"conversation_events": []},
        "meta": {
            MetaKeys.CLARIFICATION_KIND: CLARIFICATION_KIND_DATASET_SELECTION,
            MetaKeys.CLARIFICATION_RETURN_NODE: "generate_code",
            MetaKeys.PENDING_QUESTION: "plot age by sex",
            MetaKeys.CLARIFICATION_EXPECTED: build_expected(
                CLARIFICATION_KIND_DATASET_SELECTION,
                allowed_values=["uploaded-1", "uploaded-2"],
            ),
            MetaKeys.CLARIFICATION_ATTEMPT_COUNT: 0,
        },
    }

    updated = module.clarification_node(state, SimpleNamespace(), context={"runtime_datasets": True})

    assert "one listed dataset ID" in updated["output"]["qa_response"]
    assert updated["meta"][MetaKeys.CLARIFICATION_KIND] == CLARIFICATION_KIND_DATASET_SELECTION
    assert updated["meta"][MetaKeys.CLARIFICATION_ATTEMPT_COUNT] == 1


def test_clarification_node_passes_raw_db_rag_recoverable_error_reply() -> None:
    original = {
        name: sys.modules.get(name)
        for name in (
            "graph.nodes.generate_code",
            "graph.nodes.qa",
            "graph.nodes.rag_db_qa",
            "graph.nodes.tool_routing",
            "graph.nodes.clarification",
        )
    }
    try:
        called = {}

        generate_code_mod = ModuleType("graph.nodes.generate_code")
        generate_code_mod.generate_code_node = lambda state, llm, context, question_override=None: state
        qa_mod = ModuleType("graph.nodes.qa")
        qa_mod.qa_node = lambda state, llm, context="", question_override=None: state
        rag_mod = ModuleType("graph.nodes.rag_db_qa")

        def _rag_db_qa_node(state, llm, provider="", service=None, reranker_model=None, question_override=None):
            called["question_override"] = question_override
            return state

        rag_mod.rag_db_qa_node = _rag_db_qa_node
        tool_routing_mod = ModuleType("graph.nodes.tool_routing")
        tool_routing_mod.latest_user_message = lambda state: "Use SUBJID_PSEUDO as the join key."

        sys.modules["graph.nodes.generate_code"] = generate_code_mod
        sys.modules["graph.nodes.qa"] = qa_mod
        sys.modules["graph.nodes.rag_db_qa"] = rag_mod
        sys.modules["graph.nodes.tool_routing"] = tool_routing_mod
        sys.modules.pop("graph.nodes.clarification", None)
        module = importlib.import_module("graph.nodes.clarification")

        state = {
            "messages": [],
            "meta": {
                "clarification_kind": "db_rag_recoverable_error",
                "clarification_return_node": "rag_db_qa",
                "pending_question": "How should DB-RAG recover?",
            },
        }
        module.clarification_node(state, SimpleNamespace(), context={"provider": "openai", "db_rag_service": object()})

        assert called["question_override"] == "Use SUBJID_PSEUDO as the join key."
    finally:
        for name, module in original.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_clarification_node_does_not_override_db_rag_extraction_opt_in_reply() -> None:
    original = {
        name: sys.modules.get(name)
        for name in (
            "graph.nodes.generate_code",
            "graph.nodes.qa",
            "graph.nodes.rag_db_qa",
            "graph.nodes.tool_routing",
            "graph.nodes.clarification",
        )
    }
    try:
        called = {}

        generate_code_mod = ModuleType("graph.nodes.generate_code")
        generate_code_mod.generate_code_node = lambda state, llm, context, question_override=None: state
        qa_mod = ModuleType("graph.nodes.qa")
        qa_mod.qa_node = lambda state, llm, context="", question_override=None: state
        rag_mod = ModuleType("graph.nodes.rag_db_qa")

        def _rag_db_qa_node(state, llm, provider="", service=None, reranker_model=None, question_override=None):
            called["question_override"] = question_override
            return state

        rag_mod.rag_db_qa_node = _rag_db_qa_node
        tool_routing_mod = ModuleType("graph.nodes.tool_routing")
        tool_routing_mod.latest_user_message = lambda state: "yes"

        sys.modules["graph.nodes.generate_code"] = generate_code_mod
        sys.modules["graph.nodes.qa"] = qa_mod
        sys.modules["graph.nodes.rag_db_qa"] = rag_mod
        sys.modules["graph.nodes.tool_routing"] = tool_routing_mod
        sys.modules.pop("graph.nodes.clarification", None)
        module = importlib.import_module("graph.nodes.clarification")

        state = {
            "messages": [],
            "meta": {
                "clarification_kind": "rag_db_extraction_opt_in",
                "clarification_return_node": "rag_db_qa",
                "pending_question": "Would you like me to identify the tables and columns?",
            },
        }
        module.clarification_node(state, SimpleNamespace(), context={"provider": "openai", "db_rag_service": object()})

        assert called["question_override"] is None
    finally:
        for name, module in original.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
