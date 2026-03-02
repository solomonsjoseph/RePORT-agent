from __future__ import annotations

import importlib
import json
import sys
from types import ModuleType, SimpleNamespace


def _install_langchain_stubs() -> None:
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

    prompts_mod = ModuleType("langchain_core.prompts")
    prompts_mod.ChatPromptTemplate = _ChatPromptTemplate

    messages_mod = ModuleType("langchain_core.messages")
    messages_mod.BaseMessage = object

    langchain_core_mod = ModuleType("langchain_core")
    langchain_core_mod.prompts = prompts_mod
    langchain_core_mod.messages = messages_mod

    graph_message_mod = ModuleType("langgraph.graph.message")
    graph_message_mod.add_messages = lambda current, new: (current or []) + (new or [])

    sys.modules["langchain_core"] = langchain_core_mod
    sys.modules["langchain_core.prompts"] = prompts_mod
    sys.modules["langchain_core.messages"] = messages_mod
    sys.modules["langgraph.graph.message"] = graph_message_mod


class _LLM:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text

    def invoke(self, _messages):
        return SimpleNamespace(content=self.response_text)


def test_request_tools_for_question_returns_parsed_requests() -> None:
    _install_langchain_stubs()
    tool_routing = importlib.import_module("graph.nodes.tool_routing")

    llm = _LLM(
        json.dumps(
            {
                "tool_requests": [
                    {
                        "tool_name": "search",
                        "payload": {"server": "search", "query": "what is pca"},
                    }
                ]
            }
        )
    )

    result = tool_routing.request_tools_for_question(llm, "what is pca?")

    assert result == [
        {
            "tool_name": "search",
            "payload": {"server": "search", "query": "what is pca"},
        }
    ]


def test_request_tools_for_question_handles_no_tools_without_keyerror() -> None:
    _install_langchain_stubs()
    tool_routing = importlib.import_module("graph.nodes.tool_routing")

    llm = _LLM('{"tool_requests": []}')
    result = tool_routing.request_tools_for_question(llm, "what is pca?")

    assert result == []
