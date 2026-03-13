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
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self._content)


def test_planner_prompt_formats_with_node_capabilities() -> None:
    _install_langchain_and_langgraph_stubs()
    planner_prompt = importlib.import_module("prompts.planner_prompt")

    prompt = planner_prompt.make_planner_prompt().format_prompt(
        actions="qa, generate_code",
        summary="generated_code_present=False",
        node_capabilities="- qa: answer directly",
    )
    rendered = prompt.to_messages()

    assert "Node capabilities:" in rendered[0]["content"]
    assert "- qa: answer directly" in rendered[0]["content"]


def test_orchestrator_fallback_prefers_qa_for_concept_questions() -> None:
    _install_langchain_and_langgraph_stubs()
    orchestrator = importlib.import_module("graph.nodes.orchestrator")

    state = {
        "messages": [SimpleNamespace(type="human", content="What's PCA in machine learning?")],
        "output": {},
        "observations": [],
        "last_action": None,
        "orchestrator": {},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "final_decision": None},
        },
        "meta": {"error_iterations": 0, "workflow_trace": []},
    }

    fallback = orchestrator.orchestrator_node(state, _LLM("not-json"), ["qa", "generate_code", "end"])

    assert fallback["next_action"] == "qa"
    assert fallback["meta"]["intent"] == "qa"
    
def test_orchestrator_uses_llm_action_and_fallback_policy() -> None:
    _install_langchain_and_langgraph_stubs()
    orchestrator = importlib.import_module("graph.nodes.orchestrator")

    state = {
        "messages": [SimpleNamespace(type="human", content="Can you explain this?")],
        "output": {},
        "observations": [],
        "last_action": "qa",
        "orchestrator": {},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "final_decision": None},
        },
        "meta": {"error_iterations": 0, "workflow_trace": ["qa"]},
    }
    available_actions = ["qa", "generate_code", "end"]

    llm_valid = _LLM(json.dumps({"action": "qa", "thought": "simple question"}))
    chosen = orchestrator.orchestrator_node(state, llm_valid, available_actions)
    assert chosen["next_action"] == "qa"
    assert chosen["last_action"] == "qa"
    assert chosen["orchestrator"]["thought"] == "simple question"
    assert chosen["meta"]["workflow_trace"][-1] == "orchestrator"

    llm_invalid = _LLM("not-json")
    fallback = orchestrator.orchestrator_node(state, llm_invalid, available_actions)
    assert fallback["next_action"] == "qa"

def test_orchestrator_routes_sample_code_request_to_qa_without_execution_flow() -> None:
    _install_langchain_and_langgraph_stubs()
    orchestrator = importlib.import_module("graph.nodes.orchestrator")

    state = {
        "messages": [SimpleNamespace(type="human", content="Give me sample code to run survival analysis")],
        "output": {},
        "observations": [],
        "last_action": None,
        "orchestrator": {},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "final_decision": None},
        },
        "meta": {"error_iterations": 0, "workflow_trace": []},
    }

    fallback = orchestrator.orchestrator_node(state, _LLM("not-json"), ["qa", "generate_code", "end"])

    assert fallback["next_action"] == "qa"
    assert fallback["meta"]["intent"] == "qa"


def test_orchestrator_keeps_qa_intent_for_tool_clarification_followup() -> None:
    _install_langchain_and_langgraph_stubs()
    orchestrator = importlib.import_module("graph.nodes.orchestrator")

    state = {
        "messages": [
            SimpleNamespace(type="human", content="what's weather in china"),
            SimpleNamespace(type="ai", content="Which city in China would you like the weather for?"),
            SimpleNamespace(type="human", content="Shanghai"),
        ],
        "output": {},
        "observations": [],
        "last_action": "qa",
        "orchestrator": {},
        "agents": {
            "qa": {"awaiting_tool_clarification": True},
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "final_decision": None},
        },
        "meta": {
            "error_iterations": 0,
            "workflow_trace": ["orchestrator", "qa", "orchestrator"],
            "awaiting_user_clarification": True,
        },
    }

    # Even if the planner suggests ending, fallback policy should keep QA flow.
    updated = orchestrator.orchestrator_node(state, _LLM(json.dumps({"action": "end"})), ["qa", "generate_code", "end"])

    assert updated["next_action"] == "qa"
    assert updated["meta"]["intent"] == "qa"
    assert "awaiting_user_clarification" not in updated["meta"]


def test_orchestrator_does_not_force_qa_for_non_qa_clarification_followup() -> None:
    _install_langchain_and_langgraph_stubs()
    orchestrator = importlib.import_module("graph.nodes.orchestrator")

    state = {
        "messages": [
            SimpleNamespace(type="ai", content="Please clarify constraints for code generation."),
            SimpleNamespace(type="human", content="Use only pandas and sklearn."),
        ],
        "output": {},
        "observations": [],
        "last_action": "generate_code",
        "orchestrator": {},
        "agents": {
            "qa": {"awaiting_tool_clarification": False},
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "final_decision": None},
        },
        "meta": {
            "error_iterations": 0,
            "workflow_trace": ["orchestrator", "generate_code", "orchestrator"],
            "awaiting_user_clarification": True,
        },
    }

    updated = orchestrator.orchestrator_node(state, _LLM("not-json"), ["qa", "generate_code", "end"])

    assert updated["next_action"] == "generate_code"
    assert "awaiting_user_clarification" not in updated["meta"]
