from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace


class _AIMessage:
    type = "ai"

    def __init__(self, content: str):
        self.content = content


class _HumanMessage:
    type = "human"

    def __init__(self, content: str):
        self.content = content


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


def _install_stubs() -> None:
    messages_mod = ModuleType("langchain_core.messages")
    messages_mod.BaseMessage = object
    messages_mod.AIMessage = _AIMessage
    messages_mod.HumanMessage = _HumanMessage

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


def _fresh_clarification_module():
    _install_stubs()
    for mod in (
        "utils.message_window",
        "graph.nodes.tool_routing",
        "graph.nodes.qa",
        "graph.nodes.generate_code",
        "graph.nodes.error_handler",
        "graph.nodes.clarification",
    ):
        sys.modules.pop(mod, None)
    return importlib.import_module("graph.nodes.clarification")


def test_clarification_node_resumes_qa_tool_followup_and_enqueues_tool() -> None:
    clarification = _fresh_clarification_module()

    class _LLM:
        def invoke(self, _messages):
            return SimpleNamespace(
                content='{"tool_requests":[{"tool_name":"query_weather","payload":{"server":"weather","city":"Boston"}}]}'
            )

    state = {
        "messages": [
            _HumanMessage("What's weather today?"),
            _AIMessage("Which city would you like weather for?"),
            _HumanMessage("Boston"),
        ],
        "output": {"qa_response": "Which city would you like weather for?"},
        "meta": {
            "awaiting_user_clarification": True,
            "pending_question": "What's weather today?",
            "clarification_return_node": "qa",
            "clarification_kind": "qa_tool",
        },
        "observations": [],
        "agents": {"qa": {"awaiting_tool_clarification": True, "tool_requests": [], "tool_results": []}},
    }

    updated = clarification.clarification_node(state, _LLM(), context="")

    assert updated["agents"]["qa"]["status"] == "pending"
    assert updated["agents"]["qa"]["tool_requests"][0]["tool_name"] == "query_weather"
    assert "awaiting_user_clarification" not in updated["meta"]


def test_clarification_node_resumes_generic_qa_with_pending_question_context() -> None:
    clarification = _fresh_clarification_module()
    captured: dict[str, object] = {}

    def _fake_qa_node(state, llm, context="", question_override=None):
        captured["state"] = state
        captured["context"] = context
        captured["question_override"] = question_override
        return {
            **state,
            "output": {"qa_response": "Boston is the capital of Massachusetts."},
            "messages": list(state.get("messages", [])) + [_AIMessage("Boston is the capital of Massachusetts.")],
            "agents": {"qa": {"status": "done"}},
        }

    clarification.qa_node = _fake_qa_node

    state = {
        "messages": [
            _HumanMessage("Tell me about this place"),
            _AIMessage("Which place do you mean?"),
            _HumanMessage("Boston"),
        ],
        "output": {"qa_response": "Which place do you mean?"},
        "meta": {
            "awaiting_user_clarification": True,
            "pending_question": "Tell me about this place",
            "clarification_return_node": "qa",
            "clarification_kind": "qa_followup",
        },
        "observations": [],
        "agents": {"qa": {"awaiting_tool_clarification": True, "tool_requests": [], "tool_results": []}},
    }

    updated = clarification.clarification_node(state, object(), context="")

    assert captured["question_override"] == "Tell me about this place\n\nUser clarification: Boston"
    assert "awaiting_user_clarification" not in captured["state"]["meta"]
    assert updated["output"]["qa_response"] == "Boston is the capital of Massachusetts."
    assert updated["messages"][-1].content == "Boston is the capital of Massachusetts."
