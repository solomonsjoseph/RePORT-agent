from __future__ import annotations

import importlib
import json
import sys
from types import ModuleType, SimpleNamespace


def _install_stubs() -> None:
    class _MessagesPlaceholder:
        def __init__(self, variable_name: str, optional: bool = False) -> None:
            self.variable_name = variable_name
            self.optional = optional

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
                try:
                    rendered.append({"role": role, "content": template.format(**kwargs)})
                except KeyError:
                    rendered.append({"role": role, "content": template})
            return _FormattedPrompt(rendered)

    class _ChatPromptTemplate:
        @staticmethod
        def from_messages(messages):
            return _PromptTemplate(messages)

    messages_mod = ModuleType("langchain_core.messages")
    messages_mod.BaseMessage = object
    prompts_mod = ModuleType("langchain_core.prompts")
    prompts_mod.ChatPromptTemplate = _ChatPromptTemplate
    prompts_mod.MessagesPlaceholder = _MessagesPlaceholder

    langchain_core_mod = ModuleType("langchain_core")
    langchain_core_mod.messages = messages_mod
    langchain_core_mod.prompts = prompts_mod

    graph_message_mod = ModuleType("langgraph.graph.message")
    graph_message_mod.add_messages = lambda current, new: (current or []) + (new or [])

    sys.modules["langchain_core"] = langchain_core_mod
    sys.modules["langchain_core.messages"] = messages_mod
    sys.modules["langchain_core.prompts"] = prompts_mod
    sys.modules["langgraph.graph.message"] = graph_message_mod


def _fresh_orchestrator():
    _install_stubs()
    for mod in (
        "graph.state",
        "graph.nodes.tool_routing",
        "graph.nodes.node_registry",
        "graph.nodes.action_metadata",
        "graph.nodes.orchestrator.policy_contract",
        "graph.nodes.orchestrator.policy",
        "graph.nodes.orchestrator.action_mask",
        "graph.nodes.orchestrator.planner",
        "graph.nodes.orchestrator.node",
        "graph.nodes.orchestrator",
    ):
        sys.modules.pop(mod, None)
    return importlib.import_module("graph.nodes.orchestrator")


class _LLM:
    def __init__(self, content: str):
        self.content = content

    def invoke(self, _messages):
        return SimpleNamespace(content=self.content)


def test_orchestrator_falls_back_to_rag_db_qa_for_explicit_database_question() -> None:
    orchestrator = _fresh_orchestrator()

    state = {
        "messages": [SimpleNamespace(type="human", content="How many male participants are in the database cohort A?")],
        "output": {},
        "observations": [],
        "last_action": None,
        "orchestrator": {},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "final_decision": None},
            "rag_db_qa": {},
        },
        "meta": {"error_iterations": 0, "workflow_trace": []},
    }

    updated = orchestrator.orchestrator_node(
        state,
        _LLM("not-json"),
        ["qa", "rag_db_qa", "generate_code", "end"],
    )

    assert updated["next_action"] == "rag_db_qa"


def test_orchestrator_does_not_force_rag_db_qa_from_clinical_fields_alone() -> None:
    orchestrator = _fresh_orchestrator()

    state = {
        "messages": [
            SimpleNamespace(
                type="human",
                content="Help me to subset age, gender, diabetes status, and final outcome among index case",
            )
        ],
        "output": {},
        "artifacts": {"datasets": {}, "active_dataset_id": None},
        "observations": [],
        "last_action": None,
        "orchestrator": {},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "final_decision": None},
            "rag_db_qa": {},
        },
        "meta": {"error_iterations": 0, "workflow_trace": []},
    }

    updated = orchestrator.orchestrator_node(
        state,
        _LLM(json.dumps({"action": "qa", "thought": "generic clinical request"})),
        ["qa", "rag_db_qa", "generate_code", "clarification", "end"],
    )

    assert updated["next_action"] == "qa"


def test_orchestrator_routes_database_overview_to_rag_db_qa() -> None:
    orchestrator = _fresh_orchestrator()

    state = {
        "messages": [
            SimpleNamespace(
                type="human",
                content="Help me to query the database give me overview of the database, and what kind of analysis I can perform",
            )
        ],
        "output": {},
        "observations": [],
        "last_action": None,
        "orchestrator": {},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "final_decision": None},
            "rag_db_qa": {},
        },
        "meta": {"error_iterations": 0, "workflow_trace": []},
    }

    updated = orchestrator.orchestrator_node(
        state,
        _LLM(json.dumps({"action": "qa", "thought": "generic qa"})),
        ["qa", "rag_db_qa", "generate_code", "clarification", "end"],
    )

    assert updated["next_action"] == "rag_db_qa"


def test_orchestrator_prefers_pending_rag_db_column_review_over_rag_db_qa() -> None:
    orchestrator = _fresh_orchestrator()

    state = {
        "messages": [
            SimpleNamespace(
                type="human",
                content="How many male participants are in the database cohort A?",
            )
        ],
        "output": {},
        "observations": [],
        "last_action": None,
        "orchestrator": {},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "final_decision": None},
            "rag_db_qa": {
                "pending_column_review": {
                    "status": "awaiting_review",
                    "selection_id": "sel-1",
                }
            },
        },
        "meta": {"error_iterations": 0, "workflow_trace": []},
    }

    updated = orchestrator.orchestrator_node(
        state,
        _LLM(json.dumps({"action": "rag_db_qa", "thought": "database question"})),
        ["qa", "rag_db_qa", "human_review_rag_db_column_selection", "human_review_rag_db_sql_execution", "end"],
    )

    assert updated["next_action"] == "human_review_rag_db_column_selection"


def test_orchestrator_prefers_pending_rag_db_sql_review_over_rag_db_qa() -> None:
    orchestrator = _fresh_orchestrator()

    state = {
        "messages": [
            SimpleNamespace(
                type="human",
                content="How many male participants are in the database cohort A?",
            )
        ],
        "output": {},
        "observations": [],
        "last_action": None,
        "orchestrator": {},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "final_decision": None},
            "rag_db_qa": {
                "pending_sql_candidate": {
                    "status": "prepared",
                    "selection_id": "sel-1",
                }
            },
        },
        "meta": {"error_iterations": 0, "workflow_trace": []},
    }

    updated = orchestrator.orchestrator_node(
        state,
        _LLM(json.dumps({"action": "rag_db_qa", "thought": "database question"})),
        ["qa", "rag_db_qa", "human_review_rag_db_sql_execution", "end"],
    )

    assert updated["next_action"] == "human_review_rag_db_sql_execution"


def test_orchestrator_prioritizes_rag_db_qa_for_database_question_when_no_uploaded_dataset() -> None:
    orchestrator = _fresh_orchestrator()

    state = {
        "messages": [SimpleNamespace(type="human", content="Tell me about the database")],
        "output": {},
        "artifacts": {"datasets": {}, "active_dataset_id": None},
        "observations": [],
        "last_action": None,
        "orchestrator": {},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "final_decision": None},
            "rag_db_qa": {},
        },
        "meta": {"error_iterations": 0, "workflow_trace": []},
    }

    updated = orchestrator.orchestrator_node(
        state,
        _LLM(json.dumps({"action": "qa", "thought": "generic database explanation"})),
        ["qa", "rag_db_qa", "generate_code", "clarification", "end"],
    )

    assert updated["next_action"] == "rag_db_qa"


def test_orchestrator_exits_stale_qa_clarification_for_explicit_rag_database_request() -> None:
    orchestrator = _fresh_orchestrator()

    state = {
        "messages": [
            SimpleNamespace(
                type="human",
                content="Help me to query the database give me overview of the database, and what kind of analysis I can perform",
            ),
            SimpleNamespace(
                type="ai",
                content="Can you provide the database schema or describe the tables and their columns?",
            ),
            SimpleNamespace(type="human", content="No, I want to query the rag databse"),
        ],
        "output": {"qa_response": "Can you provide the database schema or describe the tables and their columns?"},
        "observations": ["orchestrator: next_action=qa", "qa: asked clarification (structured qa response)"],
        "last_action": "qa",
        "orchestrator": {},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "final_decision": None},
            "qa": {"awaiting_tool_clarification": True},
            "rag_db_qa": {},
        },
        "meta": {
            "error_iterations": 0,
            "workflow_trace": ["orchestrator", "qa"],
            "awaiting_user_clarification": True,
            "clarification_return_node": "qa",
            "clarification_kind": "qa_followup",
            "pending_question": "Help me to query the database give me overview of the database, and what kind of analysis I can perform",
        },
    }

    updated = orchestrator.orchestrator_node(
        state,
        _LLM(json.dumps({"action": "clarification", "thought": "resume qa clarification"})),
        ["qa", "rag_db_qa", "generate_code", "clarification", "end"],
    )

    assert updated["next_action"] == "rag_db_qa"


def test_orchestrator_reclassifies_qa_followup_with_planner_instead_of_hard_resuming_qa() -> None:
    orchestrator = _fresh_orchestrator()

    state = {
        "messages": [
            SimpleNamespace(type="human", content="Find the matching records"),
            SimpleNamespace(type="ai", content="Which source should I use?"),
            SimpleNamespace(type="human", content="Use the local metadata store"),
        ],
        "output": {"qa_response": "Which source should I use?"},
        "artifacts": {"datasets": {}, "active_dataset_id": None},
        "observations": ["orchestrator: next_action=qa", "qa: asked clarification (structured qa response)"],
        "last_action": "qa",
        "orchestrator": {},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "final_decision": None},
            "qa": {"awaiting_tool_clarification": True},
            "rag_db_qa": {},
        },
        "meta": {
            "error_iterations": 0,
            "workflow_trace": ["orchestrator", "qa"],
            "awaiting_user_clarification": True,
            "clarification_return_node": "qa",
            "clarification_kind": "qa_followup",
            "pending_question": "Find the matching records",
        },
    }

    updated = orchestrator.orchestrator_node(
        state,
        _LLM(json.dumps({"action": "rag_db_qa", "thought": "local metadata store means DB-RAG"})),
        ["qa", "rag_db_qa", "generate_code", "clarification", "end"],
    )

    assert updated["next_action"] == "rag_db_qa"


def test_orchestrator_routes_clinical_subset_request_from_stale_qa_clarification_to_rag_db_qa() -> None:
    orchestrator = _fresh_orchestrator()

    state = {
        "messages": [
            SimpleNamespace(type="human", content="Give me overview of the database"),
            SimpleNamespace(type="ai", content="Can you please specify which database you want an overview of?"),
            SimpleNamespace(type="human", content="the rag database in my system"),
            SimpleNamespace(type="ai", content="Can you provide details about the RAG database schema?"),
            SimpleNamespace(
                type="human",
                content="Help me to subset age, gender, diabetes status, and final outcome among index case",
            ),
        ],
        "output": {"qa_response": "Can you provide details about the RAG database schema?"},
        "artifacts": {"datasets": {}, "active_dataset_id": None},
        "observations": [],
        "last_action": "clarification",
        "orchestrator": {},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "final_decision": None},
            "qa": {"awaiting_tool_clarification": True},
            "rag_db_qa": {},
        },
        "meta": {
            "error_iterations": 0,
            "workflow_trace": ["orchestrator", "qa", "orchestrator", "clarification"],
            "awaiting_user_clarification": True,
            "clarification_return_node": "qa",
            "clarification_kind": "qa_followup",
            "pending_question": "the rag database in my system",
        },
    }

    updated = orchestrator.orchestrator_node(
        state,
        _LLM(json.dumps({"action": "clarification", "thought": "resume qa clarification"})),
        ["qa", "rag_db_qa", "generate_code", "clarification", "end"],
    )

    assert updated["next_action"] == "rag_db_qa"


def test_orchestrator_keeps_rag_db_thread_for_referential_followup() -> None:
    orchestrator = _fresh_orchestrator()

    state = {
        "messages": [
            SimpleNamespace(type="human", content="How many participants had positive cultures?"),
            SimpleNamespace(type="ai", content="I found the relevant tables and columns. Do you want SQL extraction?"),
            SimpleNamespace(type="human", content="what about females only?"),
        ],
        "output": {"qa_response": "I found the relevant tables and columns."},
        "observations": [],
        "last_action": "rag_db_qa",
        "orchestrator": {},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "final_decision": None},
            "rag_db_qa": {"active_thread": True},
        },
        "meta": {"error_iterations": 0, "workflow_trace": ["orchestrator", "rag_db_qa"]},
    }

    updated = orchestrator.orchestrator_node(
        state,
        _LLM(json.dumps({"action": "qa", "thought": "fallback"})),
        ["qa", "rag_db_qa", "generate_code", "end"],
    )

    assert updated["next_action"] == "rag_db_qa"
