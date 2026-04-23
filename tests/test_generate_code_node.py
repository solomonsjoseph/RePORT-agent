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


class _SeqLLM:
    def __init__(self, contents: list[str]):
        self.contents = list(contents)
        self.calls = []

    def invoke(self, prompt):
        self.calls.append(prompt)
        if not self.contents:
            raise AssertionError("LLM invoked more times than expected")
        return SimpleNamespace(content=self.contents.pop(0))


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

    llm = _LLM(
        json.dumps(
            {
                "response_type": "clarification",
                "question": "Which columns should be used as the duration and event variables?",
            }
        )
    )
    updated = generate_code.generate_code_node(state, llm, context="")

    assert updated["output"]["generated_code"] == ""
    assert updated["output"]["qa_response"].startswith("Which columns should be used")
    assert updated["meta"]["awaiting_user_clarification"] is True
    assert updated["meta"]["clarification_return_node"] == "generate_code"
    assert getattr(updated["messages"][-1], "type", None) == "ai"


def test_generate_code_accepts_structured_code_result_payload() -> None:
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

    llm = _LLM(
        json.dumps(
            {
                "response_type": "code_result",
                "summary": "Fits a Kaplan-Meier survival analysis stratified by sex.",
                "assumptions": "",
                "code": "print(123)",
            }
        )
    )
    updated = generate_code.generate_code_node(state, llm, context="")

    assert updated["output"]["generated_code"] == "print(123)"
    assert updated["output"]["code_summary"].startswith("Fits a Kaplan-Meier")
    assert updated["output"]["code_assumptions"] == ""


def test_generate_code_supports_callable_context_provider() -> None:
    generate_code = _fresh_generate_code()

    state = {
        "messages": [_HumanMessage("write code for descriptive analysis")],
        "output": {},
        "observations": [],
        "meta": {},
        "agents": {
            "generate_code": {"tool_requests": [], "tool_results": []},
            "human_review": {"before_run_decision": None},
        },
    }

    llm = _SeqLLM(
        [
            json.dumps(
                {
                    "response_type": "code_result",
                    "summary": "Summarizes the selected dataset.",
                    "assumptions": "",
                    "code": "print(df.shape)",
                }
            )
        ]
    )

    updated = generate_code.generate_code_node(
        state,
        llm,
        context=lambda current_state: f"context for {len(current_state['messages'])} messages",
    )

    assert updated["output"]["generated_code"] == "print(df.shape)"
    assert llm.calls[0]["context"] == "context for 1 messages"


def test_generate_code_retries_once_for_malformed_payload_then_accepts_code_result() -> None:
    generate_code = _fresh_generate_code()

    state = {
        "messages": [_HumanMessage("run df.head() for me")],
        "output": {},
        "observations": [],
        "meta": {},
        "agents": {
            "generate_code": {"tool_requests": [], "tool_results": []},
            "human_review": {"before_run_decision": None},
        },
    }

    llm = _SeqLLM(
        [
            "This is not valid JSON",
            json.dumps(
                {
                    "response_type": "code_result",
                    "summary": "Shows the first rows of df.",
                    "assumptions": "",
                    "code": "print(df.head())",
                }
            ),
        ]
    )
    updated = generate_code.generate_code_node(state, llm, context="")

    assert updated["output"]["generated_code"] == "print(df.head())"
    assert updated["output"]["code_summary"] == "Shows the first rows of df."
    assert len(llm.calls) == 2


def test_generate_code_accepts_structured_clarification_payload() -> None:
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

    llm = _LLM(
        json.dumps(
            {
                "response_type": "clarification",
                "question": "Which columns should be used as the time and event variables?",
            }
        )
    )
    updated = generate_code.generate_code_node(state, llm, context="")

    assert updated["output"]["generated_code"] == ""
    assert updated["output"]["qa_response"].startswith("Which columns should be used")
    assert updated["meta"]["awaiting_user_clarification"] is True
    assert updated["meta"]["clarification_return_node"] == "generate_code"


def test_generate_code_returns_deterministic_fallback_clarification_after_two_malformed_payloads() -> None:
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

    llm = _SeqLLM(
        [
            "I need the event column.",
            "Still not JSON.",
        ]
    )
    updated = generate_code.generate_code_node(state, llm, context="")

    assert updated["output"]["generated_code"] == ""
    assert "please restate your request" in updated["output"]["qa_response"].lower()
    assert updated["meta"]["awaiting_user_clarification"] is True
    assert updated["meta"]["clarification_return_node"] == "generate_code"
    assert len(llm.calls) == 2


def test_generate_code_resets_approval_and_sets_current_code_hash() -> None:
    generate_code = _fresh_generate_code()

    state = {
        "messages": [_HumanMessage("write python code")],
        "output": {},
        "observations": [],
        "meta": {"execution_ticket_hash": "old", "error_recovery_active": True},
        "agents": {
            "generate_code": {"tool_requests": [], "tool_results": []},
            "human_review": {
                "before_run_decision": "approve",
                "final_decision": "approve",
                "approved_code_hash": "old",
            },
        },
    }

    llm = _LLM(
        json.dumps(
            {
                "response_type": "code_result",
                "summary": "Prints 123.",
                "assumptions": "",
                "code": "print(123)",
            }
        )
    )
    updated = generate_code.generate_code_node(state, llm, context="")

    assert updated["output"]["generated_code"] == "print(123)"
    assert updated["agents"]["human_review"]["before_run_decision"] is None
    assert updated["agents"]["human_review"]["final_decision"] is None
    assert updated["agents"]["human_review"]["approved_code_hash"] is None
    assert updated["meta"].get("current_code_hash")
    assert updated["meta"].get("execution_ticket_hash") is None
    assert updated["meta"].get("error_recovery_active") is None


def test_generate_code_after_error_review_resets_failed_execution_state() -> None:
    generate_code = _fresh_generate_code()

    state = {
        "messages": [_HumanMessage("regenerate the analysis")],
        "output": {
            "generated_code": "print('old')",
            "error": {"category": "retryable_code", "type": "NameError", "message": "bad"},
            "text": "old output",
        },
        "observations": [],
        "meta": {
            "error_iterations": 5,
            "execution_ticket_hash": "old",
            "error_recovery_active": True,
        },
        "agents": {
            "generate_code": {"tool_requests": [], "tool_results": []},
            "human_review": {"before_run_decision": "approve", "approved_code_hash": "old"},
            "executor": {"run_status": "error", "status": "error"},
        },
    }

    llm = _LLM(
        json.dumps(
            {
                "response_type": "code_result",
                "summary": "Prints the new result.",
                "assumptions": "",
                "code": "print('new')",
            }
        )
    )
    updated = generate_code.generate_code_node(state, llm, context="")

    assert updated["output"]["generated_code"] == "print('new')"
    assert "error" not in updated["output"]
    assert "text" not in updated["output"]
    assert updated["meta"]["error_iterations"] == 0
    assert updated["agents"]["executor"]["run_status"] == "idle"
    assert updated["agents"]["executor"]["status"] == "idle"


def test_generate_code_does_not_run_tool_routing_clarification_path() -> None:
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
        def invoke(self, _prompt):
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "response_type": "clarification",
                        "question": "Which city would you like weather for?",
                    }
                )
            )

    updated = generate_code.generate_code_node(state, _LLMClarification(), context="")

    # No QA tool-routing clarification branch should execute in generate_code.
    assert updated["meta"]["awaiting_user_clarification"] is True
    assert updated["meta"]["clarification_return_node"] == "generate_code"
    assert updated["meta"]["clarification_kind"] == "generate_code"
    assert updated["output"]["generated_code"] == ""
