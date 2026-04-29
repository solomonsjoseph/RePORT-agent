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


class _FakeAgent:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_input = None

    def invoke(self, agent_input):
        self.last_input = agent_input
        return {
            "messages": [
                _AIMessage(self.response_text),
            ]
        }


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


def test_build_qa_system_prompt_includes_context_and_math_contract() -> None:
    qa = _fresh_qa_module()
    rendered = qa._build_qa_system_prompt("Available columns:\n- sex\n- time")

    assert "Dataset context (if available):" in rendered
    assert "- sex" in rendered
    assert "inline math with $...$ and display math with $$...$$" in rendered
    assert "Do not use plain parentheses around LaTeX commands." in rendered


def test_qa_node_invokes_create_agent_adapter_with_replaced_question() -> None:
    qa = _fresh_qa_module()
    agent = _FakeAgent('{"answer":"ok","needs_clarification":false,"clarification_question":null}')
    captured = {}

    def _fake_create_qa_agent(llm, *, context):
        captured["llm"] = llm
        captured["context"] = context
        return agent

    qa._create_qa_agent = _fake_create_qa_agent

    state = {
        "messages": [_HumanMessage("old question")],
        "output": {},
        "meta": {},
        "observations": [],
        "agents": {"qa": {"tool_requests": ["stale"], "tool_results": ["stale"]}},
    }

    result = qa.qa_node(state, _LLM(), context="Dataset context", question_override="new question")

    assert captured["context"] == "Dataset context"
    assert agent.last_input["messages"][-1].content == "new question"
    assert result["output"]["qa_response"] == "ok"
    assert result["agents"]["qa"]["tool_requests"] == []
    assert result["agents"]["qa"]["tool_results"] == []


def test_qa_node_sets_tool_clarification_meta_from_agent_response() -> None:
    qa = _fresh_qa_module()
    qa._create_qa_agent = lambda llm, *, context: _FakeAgent(
        '{"answer":"","needs_clarification":true,"clarification_question":"Which city in China would you like the weather for?"}'
    )
    state = {
        "messages": [_HumanMessage("What's weather in china today?")],
        "output": {},
        "meta": {"intent": "qa"},
        "observations": [],
        "agents": {"qa": {"tool_requests": [], "tool_results": []}},
    }

    first = qa.qa_node(state, _LLM(), context="")
    assert first["agents"]["qa"]["awaiting_tool_clarification"] is True
    assert first["output"]["qa_response"].startswith("Which city")

    assert first["meta"]["awaiting_user_clarification"] is True
    assert first["meta"]["clarification_return_node"] == "qa"
    assert first["meta"]["clarification_kind"] == "qa_tool"


def test_qa_node_sets_awaiting_clarification_when_llm_returns_structured_signal() -> None:
    qa = _fresh_qa_module()
    qa._create_qa_agent = lambda llm, *, context: _FakeAgent(
        '{"answer":"","needs_clarification":true,"clarification_question":"Which city would you like weather for?"}'
    )
    state = {
        "messages": [_HumanMessage("what's weather today")],
        "output": {},
        "meta": {"intent": "qa"},
        "observations": [],
        "agents": {"qa": {"tool_requests": [], "tool_results": []}},
    }

    result = qa.qa_node(state, _LLM(), context="")

    assert result["meta"].get("awaiting_user_clarification") is True
    assert result["meta"].get("pending_question") == "what's weather today"
    assert result["meta"].get("clarification_return_node") == "qa"
    assert result["agents"]["qa"]["awaiting_tool_clarification"] is True
    assert result["output"]["qa_response"] == "Which city would you like weather for?"


def test_qa_node_sets_generic_clarification_meta_for_non_tool_question() -> None:
    qa = _fresh_qa_module()
    qa._create_qa_agent = lambda llm, *, context: _FakeAgent(
        '{"answer":"","needs_clarification":true,"clarification_question":"Which Boston do you mean?"}'
    )

    state = {
        "messages": [_HumanMessage("Tell me about Boston")],
        "output": {},
        "meta": {"intent": "qa"},
        "observations": [],
        "agents": {"qa": {"tool_requests": [], "tool_results": []}},
    }

    result = qa.qa_node(state, _LLM(), context="")

    assert result["meta"].get("awaiting_user_clarification") is True
    assert result["meta"].get("pending_question") == "Tell me about Boston"
    assert result["meta"].get("clarification_kind") == "qa_followup"
    assert result["output"]["qa_response"] == "Which Boston do you mean?"


def test_qa_node_does_not_set_clarification_for_plaintext_followup_question() -> None:
    qa = _fresh_qa_module()
    qa._create_qa_agent = lambda llm, *, context: _FakeAgent("Would you like sample code?")

    state = {
        "messages": [_HumanMessage("Explain survival analysis")],
        "output": {},
        "meta": {"intent": "qa"},
        "observations": [],
        "agents": {"qa": {"tool_requests": [], "tool_results": []}},
    }

    result = qa.qa_node(state, _LLM(), context="")

    assert result["meta"].get("awaiting_user_clarification") is None
    assert result["agents"]["qa"]["awaiting_tool_clarification"] is False
    assert result["output"]["qa_response"] == "Would you like sample code?"


def test_qa_node_routes_tools_after_generic_clarification_followup() -> None:
    qa = _fresh_qa_module()
    qa._create_qa_agent = lambda llm, *, context: _FakeAgent(
        '{"answer":"","needs_clarification":true,"clarification_question":"What expression should I calculate?"}'
    )
    state = {
        "messages": [_HumanMessage("Please calculate for me")],
        "output": {},
        "meta": {"intent": "qa"},
        "observations": [],
        "agents": {"qa": {"tool_requests": [], "tool_results": []}},
    }

    first = qa.qa_node(state, _LLM(), context="")
    assert first["agents"]["qa"]["awaiting_tool_clarification"] is True
    assert first["output"]["qa_response"].startswith("What expression")

    assert first["meta"]["awaiting_user_clarification"] is True
    assert first["meta"]["clarification_return_node"] == "qa"
    assert first["meta"]["clarification_kind"] == "qa_tool"


def test_qa_node_persists_executable_python_from_answer_for_later_run() -> None:
    qa = _fresh_qa_module()
    qa._create_qa_agent = lambda llm, *, context: _FakeAgent(
        "Use this:\n"
        "```python\n"
        "print('ready to run')\n"
        "```\n"
    )

    state = {
        "messages": [_HumanMessage("Give me example Python code")],
        "output": {},
        "meta": {"intent": "qa"},
        "observations": [],
        "agents": {"qa": {"tool_requests": [], "tool_results": []}},
    }

    result = qa.qa_node(state, _LLM(), context="")

    assert result["output"]["qa_response"].startswith("Use this:")
    assert result["output"]["generated_code"] == "print('ready to run')"
    assert result["meta"].get("current_code_hash")
