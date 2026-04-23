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


def test_orchestrator_falls_back_to_rag_db_qa_for_database_question() -> None:
    orchestrator = _fresh_orchestrator()

    state = {
        "messages": [SimpleNamespace(type="human", content="How many male participants are in cohort A?")],
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
