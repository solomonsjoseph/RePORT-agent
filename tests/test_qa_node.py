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


class _LLM:
    def __init__(self):
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        return SimpleNamespace(content="ok")


def _install_stubs() -> None:
    messages_mod = ModuleType("langchain_core.messages")
    messages_mod.BaseMessage = object
    messages_mod.AIMessage = _AIMessage

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


def _fresh_qa_module():
    _install_stubs()
    for mod in ("utils.message_window", "graph.nodes.tool_routing", "graph.nodes.qa"):
        sys.modules.pop(mod, None)
    return importlib.import_module("graph.nodes.qa")


def test_qa_node_includes_context_in_prompt() -> None:
    qa = _fresh_qa_module()
    llm = _LLM()

    state = {
        "messages": [_HumanMessage("what variables inside the data")],
        "output": {},
        "meta": {"intent": "qa"},
        "observations": [],
        "agents": {"qa": {"tool_requests": [], "tool_results": []}},
    }

    qa.qa_node(state, llm, context="Available columns:\n- sex\n- time")

    assert llm.last_messages is not None
    rendered = "\n".join(m.get("content", "") for m in llm.last_messages)
    assert "Dataset context (if available):" in rendered
    assert "- sex" in rendered


def test_qa_node_routes_tools_after_clarification_followup() -> None:
    qa = _fresh_qa_module()

    class _LLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(content='{"clarification_question": "Which city in China would you like the weather for?"}')
            return SimpleNamespace(content='{"tool_requests":[{"tool_name":"query_weather","payload":{"server":"weather","city":"Shanghai"}}]}')

    llm = _LLM()
    state = {
        "messages": [_HumanMessage("What's weather in china today?")],
        "output": {},
        "meta": {"intent": "qa"},
        "observations": [],
        "agents": {"qa": {"tool_requests": [], "tool_results": []}},
    }

    first = qa.qa_node(state, llm, context="")
    assert first["agents"]["qa"]["awaiting_tool_clarification"] is True
    assert first["output"]["qa_response"].startswith("Which city")

    followup = {
        **first,
        "messages": list(first["messages"]) + [_HumanMessage("Shanghai")],
    }

    second = qa.qa_node(followup, llm, context="")
    assert second["agents"]["qa"]["status"] == "pending"
    assert second["agents"]["qa"]["awaiting_tool_clarification"] is False
    assert second["agents"]["qa"]["tool_requests"]
    assert second["agents"]["qa"]["tool_requests"][0]["tool_name"] == "query_weather"
    assert "pending_question" not in second.get("meta", {})


def test_qa_node_sets_awaiting_clarification_when_llm_asks_question() -> None:
    """When tool routing returns nothing (falls through) but the QA LLM naturally
    asks a clarifying question, the node must set AWAITING_USER_CLARIFICATION so
    the orchestrator routes back to qa on the user's follow-up answer (Case A)
    rather than treating it as a new independent turn (Case B / full reset).
    """
    qa = _fresh_qa_module()

    class _LLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            if self.calls == 1:
                # Tool-routing LLM returns no tool and no clarification — falls through.
                return SimpleNamespace(content='{"tool_requests": []}')
            # QA LLM naturally asks a clarifying question.
            return SimpleNamespace(content="Which city would you like weather for?")

    llm = _LLM()
    state = {
        "messages": [_HumanMessage("what's weather today")],
        "output": {},
        "meta": {"intent": "qa"},
        "observations": [],
        "agents": {"qa": {"tool_requests": [], "tool_results": []}},
    }

    result = qa.qa_node(state, llm, context="")

    assert result["meta"].get("awaiting_user_clarification") is True
    assert result["meta"].get("pending_question") == "what's weather today"
    assert result["meta"].get("clarification_return_node") == "qa"
    assert result["agents"]["qa"]["awaiting_tool_clarification"] is True
    assert result["output"]["qa_response"] == "Which city would you like weather for?"


def test_qa_node_routes_tools_after_generic_clarification_followup() -> None:
    qa = _fresh_qa_module()

    class _LLM:
        def __init__(self):
            self.calls = 0

        def invoke(self, _messages):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(content='{"clarification_question": "What expression should I calculate?"}')
            return SimpleNamespace(content='{"tool_requests":[{"tool_name":"calculate","payload":{"server":"calculator","expression":"(2 + 3) * 4"}}]}')

    llm = _LLM()
    state = {
        "messages": [_HumanMessage("Please calculate for me")],
        "output": {},
        "meta": {"intent": "qa"},
        "observations": [],
        "agents": {"qa": {"tool_requests": [], "tool_results": []}},
    }

    first = qa.qa_node(state, llm, context="")
    assert first["agents"]["qa"]["awaiting_tool_clarification"] is True
    assert first["output"]["qa_response"].startswith("What expression")

    followup = {
        **first,
        "messages": list(first["messages"]) + [_HumanMessage("(2 + 3) * 4")],
    }

    second = qa.qa_node(followup, llm, context="")
    assert second["agents"]["qa"]["status"] == "pending"
    assert second["agents"]["qa"]["awaiting_tool_clarification"] is False
    assert second["agents"]["qa"]["tool_requests"]
    assert second["agents"]["qa"]["tool_requests"][0]["tool_name"] == "calculate"
