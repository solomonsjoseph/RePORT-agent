from __future__ import annotations

import importlib
import json
import sys
from types import ModuleType, SimpleNamespace


def _install_langchain_stubs() -> None:
    class _FormattedPrompt:
        def __init__(self, rendered: list) -> None:
            self._rendered = rendered

        def to_messages(self):
            return self._rendered

    class _MessagesPlaceholder:
        """Stub — renders as nothing when format_prompt is called."""
        def __init__(self, variable_name: str, optional: bool = False) -> None:
            self.variable_name = variable_name
            self.optional = optional

    class _PromptTemplate:
        def __init__(self, messages):
            # Keep only (role, template) tuples; skip placeholder objects.
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

    prompts_mod = ModuleType("langchain_core.prompts")
    prompts_mod.ChatPromptTemplate = _ChatPromptTemplate
    prompts_mod.MessagesPlaceholder = _MessagesPlaceholder

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


def _fresh_tool_routing():
    _install_langchain_stubs()
    sys.modules.pop("graph.nodes.tool_routing", None)
    return importlib.import_module("graph.nodes.tool_routing")


# ---------------------------------------------------------------------------
# Existing tests — updated to unwrap ToolRoutingResult
# ---------------------------------------------------------------------------

def test_request_tools_for_question_returns_parsed_requests() -> None:
    tool_routing = _fresh_tool_routing()

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

    assert result.clarification_question is None
    assert result.tool_requests == [
        {
            "tool_name": "search",
            "payload": {"server": "search", "query": "what is pca"},
        }
    ]


def test_request_tools_for_question_handles_no_tools_without_keyerror() -> None:
    tool_routing = _fresh_tool_routing()

    llm = _LLM('{"tool_requests": []}')
    result = tool_routing.request_tools_for_question(llm, "what is pca?")

    assert result.tool_requests == []
    assert result.clarification_question is None


# ---------------------------------------------------------------------------
# New tests
# ---------------------------------------------------------------------------

def test_request_tools_emits_clarification_question_when_required_field_missing() -> None:
    tool_routing = _fresh_tool_routing()

    llm = _LLM('{"clarification_question": "Which city would you like weather for?"}')
    result = tool_routing.request_tools_for_question(llm, "What is the weather like?")

    assert result.tool_requests == []
    assert result.clarification_question == "Which city would you like weather for?"


def test_request_tools_no_clarification_when_all_fields_present() -> None:
    tool_routing = _fresh_tool_routing()

    llm = _LLM(
        json.dumps(
            {
                "tool_requests": [
                    {
                        "tool_name": "query_weather",
                        "payload": {"server": "weather", "city": "Boston"},
                    }
                ]
            }
        )
    )
    result = tool_routing.request_tools_for_question(llm, "What is the weather in Boston?")

    assert result.clarification_question is None
    assert len(result.tool_requests) == 1
    assert result.tool_requests[0]["payload"]["city"] == "Boston"


def test_request_tools_returns_empty_on_invalid_json() -> None:
    tool_routing = _fresh_tool_routing()

    llm = _LLM("This is not JSON at all.")
    result = tool_routing.request_tools_for_question(llm, "some question")

    assert result.tool_requests == []
    assert result.clarification_question is None


def test_request_tools_handles_anthropic_style_content_blocks() -> None:
    tool_routing = _fresh_tool_routing()

    class _AnthropicLLM:
        def invoke(self, _messages):
            return SimpleNamespace(
                content=[
                    {
                        "type": "text",
                        "text": '{"clarification_question": "Which city would you like weather for?"}',
                    }
                ]
            )

    result = tool_routing.request_tools_for_question(_AnthropicLLM(), "what's the weather today")

    assert result.tool_requests == []
    assert result.clarification_question == "Which city would you like weather for?"


def test_format_tool_catalog_renders_required_and_optional_fields() -> None:
    tool_routing = _fresh_tool_routing()

    catalog = [
        {
            "tool_name": "query_weather",
            "server": "weather",
            "description": "Get weather.",
            "required_fields": {"city": "string"},
            "optional_fields": {"start_date": "YYYY-MM-DD"},
        }
    ]
    rendered = tool_routing.format_tool_catalog(catalog)

    assert "required" in rendered
    assert "optional" in rendered
    assert "city" in rendered
    assert "start_date" in rendered


def test_should_route_tools_false_for_dataset_analysis_question() -> None:
    tool_routing = _fresh_tool_routing()

    assert tool_routing.should_route_tools(
        "perform survival analysis for my attached data stratified by sex"
    ) is False


def test_should_route_tools_true_for_weather_question() -> None:
    tool_routing = _fresh_tool_routing()

    assert tool_routing.should_route_tools("What is the weather in Boston?") is True


def test_should_route_tools_true_for_imperative_search_request() -> None:
    tool_routing = _fresh_tool_routing()

    assert tool_routing.should_route_tools("Go search online for new CDC flu guidance") is True
