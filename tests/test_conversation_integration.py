from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pandas as pd


class _HumanMessage:
    type = "human"

    def __init__(self, content: str, id: str | None = None):
        self.content = content
        self.id = id or ""


class _AIMessage:
    type = "ai"

    def __init__(self, content: str, additional_kwargs: dict | None = None):
        self.content = content
        self.additional_kwargs = additional_kwargs or {}


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
        self._messages = messages

    def invoke(self, payload):
        return payload

    def format_prompt(self, **kwargs):
        rendered = []
        for item in self._messages:
            if isinstance(item, tuple):
                role, template = item
                try:
                    rendered.append({"role": role, "content": template.format(**kwargs)})
                except KeyError:
                    rendered.append({"role": role, "content": template})
            elif isinstance(item, _MessagesPlaceholder):
                value = kwargs.get(item.variable_name, [])
                rendered.extend(value if isinstance(value, list) else [])
            else:
                # Few-shot prompt blocks and similar helper objects are not
                # needed for these scenario tests.
                continue
        return _FormattedPrompt(rendered)


class _ChatPromptTemplate:
    @staticmethod
    def from_messages(messages):
        return _PromptTemplate(messages)


class _FewShotChatMessagePromptTemplate:
    def __init__(self, example_prompt=None, examples=None):
        self.example_prompt = example_prompt
        self.examples = examples or []


class _SeqLLM:
    def __init__(self, contents: list[str]):
        self._contents = list(contents)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if not self._contents:
            raise AssertionError("LLM invoked more times than expected")
        return SimpleNamespace(content=self._contents.pop(0))


def _install_stubs(review_decisions: list[dict]) -> None:
    langchain_messages = ModuleType("langchain_core.messages")
    langchain_messages.BaseMessage = object
    langchain_messages.HumanMessage = _HumanMessage
    langchain_messages.AIMessage = _AIMessage

    prompts_mod = ModuleType("langchain_core.prompts")
    prompts_mod.ChatPromptTemplate = _ChatPromptTemplate
    prompts_mod.MessagesPlaceholder = _MessagesPlaceholder
    prompts_mod.FewShotChatMessagePromptTemplate = _FewShotChatMessagePromptTemplate

    langchain_core_mod = ModuleType("langchain_core")
    langchain_core_mod.messages = langchain_messages
    langchain_core_mod.prompts = prompts_mod

    graph_message_mod = ModuleType("langgraph.graph.message")
    graph_message_mod.add_messages = lambda current, new: (current or []) + (new or [])

    interrupt_queue = list(review_decisions)

    langgraph_types = ModuleType("langgraph.types")

    def _interrupt(_payload):
        if not interrupt_queue:
            raise AssertionError("interrupt called without queued test decision")
        return interrupt_queue.pop(0)

    langgraph_types.interrupt = _interrupt

    lifelines_mod = ModuleType("lifelines")
    lifelines_mod.KaplanMeierFitter = object
    lifelines_mod.CoxPHFitter = object

    sys.modules["langchain_core"] = langchain_core_mod
    sys.modules["langchain_core.messages"] = langchain_messages
    sys.modules["langchain_core.prompts"] = prompts_mod
    sys.modules["langgraph.graph.message"] = graph_message_mod
    sys.modules["langgraph.types"] = langgraph_types
    sys.modules["lifelines"] = lifelines_mod


def _fresh_modules(review_decisions: list[dict]):
    _install_stubs(review_decisions)
    for mod in (
        "utils.message_window",
        "graph.nodes.tool_routing",
        "prompts.generate_prompt",
        "prompts.planner_prompt",
        "graph.nodes.generate_code",
        "graph.nodes.execute_code",
        "graph.nodes.qa",
        "graph.nodes.human_review_before_run",
        "graph.nodes.human_review_final",
        "graph.nodes.orchestrator",
        "graph.nodes.orchestrator.node",
        "graph.nodes.orchestrator.planner",
    ):
        sys.modules.pop(mod, None)

    return {
        "generate_code": importlib.import_module("graph.nodes.generate_code"),
        "execute_code": importlib.import_module("graph.nodes.execute_code"),
        "qa": importlib.import_module("graph.nodes.qa"),
        "human_review_before_run": importlib.import_module("graph.nodes.human_review_before_run"),
        "human_review_final": importlib.import_module("graph.nodes.human_review_final"),
        "orchestrator": importlib.import_module("graph.nodes.orchestrator"),
    }


def _initial_state(user_message: _HumanMessage) -> dict:
    return {
        "messages": [user_message],
        "output": {},
        "next_action": None,
        "last_action": None,
        "observations": [],
        "orchestrator": {},
        "planner": {},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {
                "before_run_decision": None,
                "after_error_decision": None,
                "final_decision": None,
            },
            "qa": {},
            "generate_code": {},
        },
        "meta": {"error_iterations": 0, "workflow_trace": []},
    }


def _run_action(state: dict, node_name: str, fn, *args):
    updated_state = fn(state, *args)
    if not isinstance(updated_state, dict):
        updated_state = {}
    merged_state = {**state, **updated_state}

    orchestrator_state = dict(merged_state.get("orchestrator", {}))
    orchestrator_state.pop("next_action", None)
    meta = dict(merged_state.get("meta", {}))
    workflow_trace = list(meta.get("workflow_trace", []))
    workflow_trace.append(node_name)
    meta["workflow_trace"] = workflow_trace[-100:]

    bypass_actions = [
        a for a in meta.get("loop_guard_bypass_actions", [])
        if a != node_name
    ]
    if bypass_actions:
        meta["loop_guard_bypass_actions"] = bypass_actions
    else:
        meta.pop("loop_guard_bypass_actions", None)

    return {
        **merged_state,
        "next_action": None,
        "last_action": node_name,
        "orchestrator": orchestrator_state,
        "planner": dict(merged_state.get("planner", {})),
        "meta": meta,
    }


def _count_obs(state: dict, needle: str) -> int:
    return sum(1 for item in state.get("observations", []) if needle in item)


def _successful_analysis_state(mods) -> dict:
    orchestrator = mods["orchestrator"]
    generate_code = mods["generate_code"]
    execute_code = mods["execute_code"]
    before_run = mods["human_review_before_run"]
    final_review = mods["human_review_final"]

    planner_llm = _SeqLLM(['{"action":"generate_code","thought":"user asked to run code"}'])
    state = _initial_state(_HumanMessage("run df.head() for me", id="u1"))

    available_actions = [
        "qa",
        "generate_code",
        "human_review_before_run",
        "execute_code",
        "human_review_final",
        "end",
    ]

    state = orchestrator.orchestrator_node(state, planner_llm, available_actions)
    assert state["next_action"] == "generate_code"

    code_llm = _SeqLLM(["```python\nprint(df.head())\n```"])
    state = _run_action(state, "generate_code", generate_code.generate_code_node, code_llm, "Available columns:\n- x")

    state = orchestrator.orchestrator_node(state, _SeqLLM([]), available_actions)
    assert state["next_action"] == "human_review_before_run"

    state = _run_action(state, "human_review_before_run", before_run.human_review_before_run_node)
    state = orchestrator.orchestrator_node(state, _SeqLLM([]), available_actions)
    assert state["next_action"] == "execute_code"

    def _fake_run_python_user(_code, _df):
        return None, "Empty DataFrame\nColumns: []\nIndex: []\n", b"", None

    execute_code.run_python_user = _fake_run_python_user
    state = _run_action(state, "execute_code", execute_code.execute_code_node, pd.DataFrame())

    state = orchestrator.orchestrator_node(state, _SeqLLM([]), available_actions)
    assert state["next_action"] == "human_review_final"

    state = _run_action(state, "human_review_final", final_review.human_review_final_node)
    state = orchestrator.orchestrator_node(state, _SeqLLM([]), available_actions)
    assert state["next_action"] == "end"
    return state


def test_completed_final_review_is_idempotent_on_checkpoint_like_reentry() -> None:
    mods = _fresh_modules(
        [{"action": "approve"}, {"action": "approve"}]
    )

    state = _successful_analysis_state(mods)
    initial_count = _count_obs(state, "consumed final approval; routing to end")

    rerun_state = mods["orchestrator"].orchestrator_node(
        state,
        _SeqLLM([]),
        ["qa", "generate_code", "human_review_final", "end"],
    )

    assert rerun_state["next_action"] == "end"
    assert _count_obs(rerun_state, "consumed final approval; routing to end") == initial_count


def test_new_user_turn_after_completed_analysis_routes_to_qa_without_reconsuming_final_review() -> None:
    mods = _fresh_modules(
        [{"action": "approve"}, {"action": "approve"}]
    )

    state = _successful_analysis_state(mods)
    initial_count = _count_obs(state, "consumed final approval; routing to end")

    state = {
        **state,
        "messages": [*state["messages"], _HumanMessage("Who are you", id="u2")],
    }

    updated = mods["orchestrator"].orchestrator_node(
        state,
        _SeqLLM(['{"action":"qa","thought":"identity question"}']),
        ["qa", "generate_code", "human_review_final", "end"],
    )

    assert updated["next_action"] == "qa"
    assert _count_obs(updated, "consumed final approval; routing to end") == initial_count
