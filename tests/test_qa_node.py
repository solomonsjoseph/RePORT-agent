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
