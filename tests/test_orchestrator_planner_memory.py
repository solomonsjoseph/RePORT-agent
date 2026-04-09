from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace


def _install_langchain_and_langgraph_stubs() -> None:
    class _MessagesPlaceholder:
        def __init__(self, variable_name: str, optional: bool = False) -> None:
            self.variable_name = variable_name
            self.optional = optional

    class _FormattedPrompt:
        def __init__(self, rendered: list[dict[str, str]]) -> None:
            self._rendered = rendered

        def to_messages(self):
            return self._rendered

    class _PromptTemplate:
        def __init__(self, messages):
            self._messages = messages

        def format_prompt(self, **kwargs):
            rendered = []
            for role, template in self._messages:
                rendered.append({"role": role, "content": template.format(**kwargs)})
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


_install_langchain_and_langgraph_stubs()

from graph.nodes.orchestrator.state_logic import (  # noqa: E402
    build_planner_recent_turns,
    derive_planner_memory,
)
from graph.nodes.orchestrator.context_builder import build_planner_context  # noqa: E402
from graph.nodes.orchestrator import orchestrator_node  # noqa: E402


class _LLM:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self._content)


def test_derive_planner_memory_uses_latest_request_as_active_goal() -> None:
    state = {
        "messages": [SimpleNamespace(type="human", content="Explain the orchestrator design.")],
        "planner": {},
        "meta": {},
    }

    memory = derive_planner_memory(state)

    assert memory["active_user_goal"] == "Explain the orchestrator design."
    assert "orchestrator design" in memory["conversation_intent_summary"]
    assert memory["unresolved_user_constraints"] == []


def test_build_planner_recent_turns_returns_empty_when_no_pending_human_or_clarification() -> None:
    state = {
        "messages": [
            SimpleNamespace(type="human", content="Explain PCA"),
            SimpleNamespace(type="ai", content="PCA reduces dimensionality."),
        ],
        "meta": {"awaiting_user_clarification": False},
    }

    assert build_planner_recent_turns(state) == []


def test_build_planner_context_includes_memory_and_recent_turns() -> None:
    state = {
        "messages": [
            SimpleNamespace(type="ai", content="Which option do you want?"),
            SimpleNamespace(type="human", content="The second option."),
        ],
        "planner": {
            "memory": {
                "active_user_goal": "Explain the planner design",
                "conversation_intent_summary": "User is evaluating planner memory behavior.",
                "unresolved_user_constraints": ["Keep planner transcript-light"],
            },
            "decision_trace": [],
        },
        "meta": {"awaiting_user_clarification": True},
        "observations": [],
        "artifacts": {},
        "node_data": {"executor": {}, "human_review": {}},
    }

    context = build_planner_context(state, ["qa", "end"])

    assert context["planner_memory"]["active_user_goal"] == "Explain the planner design"
    assert context["recent_turns_for_planner"] == [
        {"role": "ai", "content": "Which option do you want?"},
        {"role": "human", "content": "The second option."},
    ]


def test_orchestrator_updates_planner_memory_before_planning() -> None:
    state = {
        "messages": [SimpleNamespace(type="human", content="Explain the planner tradeoff.")],
        "output": {},
        "observations": [],
        "last_action": None,
        "orchestrator": {},
        "planner": {"decision_trace": []},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "final_decision": None},
        },
        "meta": {"workflow_trace": []},
    }

    updated = orchestrator_node(
        state,
        _LLM('{"action":"qa","thought":"concept question"}'),
        ["qa", "end"],
    )

    assert updated["planner"]["memory"]["active_user_goal"] == "Explain the planner tradeoff."


def test_planner_recent_turns_capture_last_two_turns_when_waiting_on_user() -> None:
    state = {
        "messages": [
            SimpleNamespace(type="human", content="Compare two options."),
            SimpleNamespace(type="ai", content="Which option do you prefer?"),
            SimpleNamespace(type="human", content="The second option."),
        ],
        "meta": {"awaiting_user_clarification": False},
    }

    assert build_planner_recent_turns(state) == [
        {"role": "human", "content": "Compare two options."},
        {"role": "ai", "content": "Which option do you prefer?"},
        {"role": "human", "content": "The second option."},
    ]
