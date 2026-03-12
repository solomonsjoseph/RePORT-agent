from __future__ import annotations

import importlib
import json
import sys
from types import ModuleType, SimpleNamespace


class _HumanMessage:
    type = "human"

    def __init__(self, content: str):
        self.content = content


class _AIMessage:
    type = "ai"

    def __init__(self, content: str):
        self.content = content


class _Prompt:
    def invoke(self, payload):
        return payload


class _LLM:
    def __init__(self, content: str):
        self.content = content

    def invoke(self, _prompt):
        return SimpleNamespace(content=self.content)


def _install_stubs() -> None:
    messages_mod = ModuleType("langchain_core.messages")
    messages_mod.BaseMessage = object
    messages_mod.HumanMessage = _HumanMessage
    messages_mod.AIMessage = _AIMessage

    prompt_mod = ModuleType("prompts.generate_prompt")
    prompt_mod.make_generate_code_prompt = lambda: _Prompt()

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
    sys.modules["prompts.generate_prompt"] = prompt_mod


def _fresh_generate_code():
    """Install stubs and return a freshly imported generate_code module."""
    _install_stubs()
    for mod in ("utils.message_window", "graph.nodes.tool_routing", "graph.nodes.generate_code"):
        sys.modules.pop(mod, None)
    return importlib.import_module("graph.nodes.generate_code")


def test_generate_code_handles_non_code_response_without_execution_path() -> None:
    generate_code = _fresh_generate_code()

    state = {
        "messages": [_HumanMessage("write code for survival analysis")],
        "output": {},
        "observations": [],
        "meta": {},
        "agents": {
            "generate_code": {"tool_requests": [], "tool_results": []},
            "human_review": {"before_run_decision": None},
        },
    }

    # LLM returns non-code text on every call (tool routing + code gen).
    llm = _LLM(
        "I cannot write the code without duration/event columns. Please provide schema."
    )
    updated = generate_code.generate_code_node(state, llm, context="")

    assert updated["output"]["generated_code"] == ""
    assert updated["output"]["qa_response"].startswith("I cannot write the code")
    assert updated["meta"]["awaiting_user_clarification"] is True
    assert getattr(updated["messages"][-1], "type", None) == "ai"


def test_generate_code_resets_approval_and_sets_current_code_hash() -> None:
    generate_code = _fresh_generate_code()

    state = {
        "messages": [_HumanMessage("write python code")],
        "output": {},
        "observations": [],
        "meta": {},
        "agents": {
            "generate_code": {"tool_requests": [], "tool_results": []},
            "human_review": {"before_run_decision": "approve", "approved_code_hash": "old"},
        },
    }

    # First call (tool routing): non-JSON → empty ToolRoutingResult → falls through.
    # Second call (code gen): valid Python.
    llm = _LLM("```python\nprint(123)\n```")
    updated = generate_code.generate_code_node(state, llm, context="")

    assert updated["output"]["generated_code"] == "print(123)"
    assert updated["agents"]["human_review"]["before_run_decision"] is None
    assert updated["agents"]["human_review"]["approved_code_hash"] is None
    assert updated["meta"].get("current_code_hash")


def test_generate_code_asks_clarification_when_tool_field_missing() -> None:
    """When tool routing signals a missing required field, the node should emit
    a clarification question and set awaiting_user_clarification."""
    generate_code = _fresh_generate_code()

    state = {
        "messages": [_HumanMessage("What's the weather like?")],
        "output": {},
        "observations": [],
        "meta": {},
        "agents": {
            "generate_code": {"tool_requests": [], "tool_results": []},
            "human_review": {"before_run_decision": None},
        },
    }

    class _LLMClarification:
        """Returns clarification JSON on the first call (tool routing), then
        raises to verify no second LLM call is made."""
        def __init__(self):
            self._calls = 0

        def invoke(self, _prompt):
            self._calls += 1
            if self._calls == 1:
                return SimpleNamespace(
                    content=json.dumps(
                        {"clarification_question": "Which city would you like weather for?"}
                    )
                )
            raise AssertionError("LLM should not be called after clarification is returned")

    updated = generate_code.generate_code_node(state, _LLMClarification(), context="")

    assert updated["meta"]["awaiting_user_clarification"] is True
    assert updated["output"]["generated_code"] == ""
    assert updated["output"]["qa_response"] == "Which city would you like weather for?"
    last_msg = updated["messages"][-1]
    assert getattr(last_msg, "type", None) == "ai"
    assert last_msg.content == "Which city would you like weather for?"
    assert any("clarification" in obs for obs in updated.get("observations", []))
