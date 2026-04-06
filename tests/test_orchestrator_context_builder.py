from __future__ import annotations

import importlib
import sys
from types import ModuleType


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
