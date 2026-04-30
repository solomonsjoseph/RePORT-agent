from __future__ import annotations

import importlib
import json
import sys
from types import ModuleType, SimpleNamespace


class _HumanMessage:
    type = "human"

    def __init__(self, content: str, id: str | None = None):
        self.content = content
        self.id = id or ""


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


class _LLM:
    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self.response)


def _install_stubs() -> None:
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


def _fresh_orchestrator_module():
    _install_stubs()
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
    return importlib.import_module("graph.nodes.orchestrator")


def test_orchestrator_emits_user_event_once_for_fresh_turn() -> None:
    orchestrator = _fresh_orchestrator_module()
    state = {
        "messages": [_HumanMessage("Tell me about the dataset", id="turn-1")],
        "output": {},
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
        },
        "artifacts": {
            "conversation_events": [],
            "conversation_events_version": 1,
            "artifact_manifest_version": 1,
            "files": {},
        },
        "meta": {"error_iterations": 0, "workflow_trace": []},
    }

    llm = _LLM(json.dumps({"action": "qa", "thought": "answer the user"}))
    available_actions = ["qa", "end"]

    first = orchestrator.orchestrator_node(state, llm, available_actions)
    second = orchestrator.orchestrator_node(first, llm, available_actions)

    from graph.state_views import get_conversation_events

    events = get_conversation_events(second)
    user_events = [event for event in events if event["type"] == "user"]

    assert len(user_events) == 1
    assert user_events[0]["text"] == "Tell me about the dataset"
    assert user_events[0]["user_turn_hash"]
    assert [message.content for message in second["messages"]] == ["Tell me about the dataset"]
