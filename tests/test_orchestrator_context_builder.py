from __future__ import annotations

import importlib
import json
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
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[list[dict[str, str]]] = []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self._content)


def test_build_planner_context_includes_history_observations_and_progress() -> None:
    _install_langchain_and_langgraph_stubs()
    build_planner_context = importlib.import_module(
        "graph.nodes.orchestrator.context_builder"
    ).build_planner_context

    state = {
        "messages": [],
        "artifacts": {"generated_code": "print(1)", "error": {"category": "retryable_code"}},
        "node_data": {
            "executor": {"run_status": "error"},
            "qa": {"status": "done"},
        },
        "planner": {
            "decision_trace": [
                {"action": "generate_code", "thought": "user asked for analysis"}
            ]
        },
        "observations": ["generate_code: code_generated", "execute_code: execution_failed_retryable"],
        "meta": {
            "workflow_trace": ["orchestrator", "generate_code", "orchestrator", "execute_code"],
            "progress_made_last_step": False,
            "stagnation_count": 2,
        },
        "last_action": "execute_code",
    }

    context = build_planner_context(
        state,
        available_actions=["qa", "generate_code", "execute_code"],
    )

    assert "recent_observations" in context
    assert "stagnation_count=2" in context["environment_summary"]
    assert "generate_code" in context["node_capabilities"]


def test_planner_prompt_uses_rich_context_and_normalized_affordances() -> None:
    _install_langchain_and_langgraph_stubs()
    llm_select_next_action = importlib.import_module(
        "graph.nodes.orchestrator.planner"
    ).llm_select_next_action

    state = {
        "messages": [],
        "artifacts": {"generated_code": "print(1)"},
        "output": {},
        "planner": {
            "decision_trace": [
                {"action": "generate_code", "thought": "initial analysis route"}
            ]
        },
        "observations": ["generate_code: code_generated"],
        "node_data": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "final_decision": None},
        },
        "meta": {"workflow_trace": ["orchestrator", "generate_code"]},
        "last_action": "generate_code",
    }

    llm = _LLM(json.dumps({"action": "human_review_before_run", "thought": "code exists"}))
    llm_select_next_action(
        state,
        llm,
        ["generate_code", "human_review_before_run", "execute_code", "end"],
    )

    rendered = llm.calls[0]
    combined = "\n".join(message["content"] for message in rendered)

    assert "recent_observations=['generate_code: code_generated']" in combined
    assert 'Recent observations:\n["generate_code: code_generated"]' in combined
    assert 'Planner decision trace:\n[{"action": "generate_code", "thought": "initial analysis route"}]' in combined
    assert "Allowed actions:\nend, execute_code, generate_code, human_review_before_run" in combined
    assert "Blocked actions:\nnone" in combined
