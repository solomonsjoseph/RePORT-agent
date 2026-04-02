from __future__ import annotations

import importlib
import json
import os
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


class _SeqLLM:
    def __init__(self, contents: list[str]) -> None:
        self._contents = list(contents)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self._contents:
            content = self._contents.pop(0)
        else:
            content = ""
        return SimpleNamespace(content=content)


def test_planner_prompt_formats_with_node_capabilities() -> None:
    _install_langchain_and_langgraph_stubs()
    planner_prompt = importlib.import_module("prompts.planner_prompt")

    prompt = planner_prompt.make_planner_prompt().format_prompt(
        actions="qa, generate_code",
        summary="generated_code_present=False",
        node_capabilities="- qa: answer directly",
        ready_actions="qa",
        blocked_actions="- generate_code",
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
            "executor": {"run_status": "ok"},
            "human_review": {"before_run_decision": None, "final_decision": None},
        },
        "meta": {"error_iterations": 0, "workflow_trace": []},
    }

    fallback = orchestrator.orchestrator_node(state, _LLM("not-json"), ["qa", "generate_code", "end"])

    assert fallback["next_action"] == "qa"
    assert fallback["meta"]["intent"] == "qa"


def test_orchestrator_uses_inferred_qa_intent_in_same_pass() -> None:
    _install_langchain_and_langgraph_stubs()
    orchestrator = importlib.import_module("graph.nodes.orchestrator")

    state = {
        "messages": [SimpleNamespace(type="human", content="who are you")],
        "output": {},
        "observations": [],
        "last_action": None,
        "orchestrator": {},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "final_decision": None},
            "qa": {},
        },
        "meta": {"error_iterations": 0, "workflow_trace": []},
    }

    updated = orchestrator.orchestrator_node(state, _LLM('{"action":"qa","thought":"identity question"}'), ["qa", "clarification", "end"])

    assert updated["next_action"] == "qa"
    assert updated["meta"]["intent"] == "qa"


def test_orchestrator_prefers_ranked_planner_candidate_over_end_when_no_ready_actions() -> None:
    _install_langchain_and_langgraph_stubs()
    orchestrator = importlib.import_module("graph.nodes.orchestrator")

    state = {
        "messages": [SimpleNamespace(type="human", content="who are you")],
        "output": {},
        "observations": [],
        "last_action": None,
        "orchestrator": {},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "final_decision": None},
            "qa": {},
        },
        "meta": {"error_iterations": 0, "workflow_trace": []},
    }

    updated = orchestrator.orchestrator_node(
        state,
        _LLM('{"thought":"identity question","ranked_actions":["qa","clarification","end"]}'),
        ["qa", "clarification", "end"],
    )

    assert updated["next_action"] == "qa"
    
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
            "executor": {"run_status": "ok"},
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
            SimpleNamespace(type="human", content="what's weather in USA today?"),
            SimpleNamespace(type="ai", content="Which city in USA would you like the weather for?"),
            SimpleNamespace(type="human", content="New York"),
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

    # Clarification follow-ups should route through the dedicated clarification node.
    updated = orchestrator.orchestrator_node(
        state,
        _LLM(json.dumps({"action": "end"})),
        ["clarification", "qa", "generate_code", "end"],
    )

    assert updated["next_action"] == "clarification"
    assert updated["meta"]["intent"] == "qa"
    assert updated["meta"]["awaiting_user_clarification"] is True


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

    updated = orchestrator.orchestrator_node(
        state,
        _LLM("not-json"),
        ["clarification", "qa", "generate_code", "end"],
    )

    assert updated["next_action"] == "clarification"
    assert updated["meta"]["awaiting_user_clarification"] is True




def test_orchestrator_accepts_provider_block_content_for_planner_json() -> None:
    _install_langchain_and_langgraph_stubs()
    orchestrator = importlib.import_module("graph.nodes.orchestrator")

    class _BlockLLM:
        def invoke(self, messages):
            return SimpleNamespace(content=[{"type": "text", "text": json.dumps({"action": "qa", "thought": "provider block"})}])

    state = {
        "messages": [SimpleNamespace(type="human", content="Explain regularization")],
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

    updated = orchestrator.orchestrator_node(state, _BlockLLM(), ["qa", "generate_code", "end"])

    assert updated["next_action"] == "qa"
    assert updated["orchestrator"]["thought"] == "provider block"
def test_orchestrator_routes_attached_data_analysis_to_generate_code() -> None:
    _install_langchain_and_langgraph_stubs()
    orchestrator = importlib.import_module("graph.nodes.orchestrator")

    state = {
        "messages": [SimpleNamespace(type="human", content="Can you perform survival analysis on my attached data?")],
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

    assert fallback["next_action"] == "generate_code"
    assert fallback["meta"]["intent"] == "code"


def test_detect_two_node_cycle_ignores_orchestrator_ping_pong() -> None:
    _install_langchain_and_langgraph_stubs()
    orchestrator = importlib.import_module("graph.nodes.orchestrator")

    trace = ["orchestrator", "execute_code", "orchestrator", "execute_code", "orchestrator"]
    assert orchestrator._detect_two_node_cycle(trace) is False


def test_orchestrator_regenerate_before_run_routes_back_to_generate_code() -> None:
    _install_langchain_and_langgraph_stubs()
    orchestrator = importlib.import_module("graph.nodes.orchestrator")

    state = {
        "messages": [SimpleNamespace(type="human", content="please regenerate with smoking and TB terms")],
        "output": {"generated_code": "print('old')"},
        "observations": [],
        "last_action": "execute_code",
        "orchestrator": {"next_action": "execute_code"},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": "regenerate", "approved_code_hash": "abc", "final_decision": None},
        },
        "meta": {"error_iterations": 0, "workflow_trace": ["orchestrator", "human_review_before_run"], "current_code_hash": "abc"},
    }

    updated = orchestrator.orchestrator_node(
        state,
        _LLM("not-json"),
        ["qa", "generate_code", "execute_code", "human_review_before_run", "end"],
    )

    assert updated["next_action"] == "generate_code"
    assert updated["output"].get("generated_code") is None
    assert (updated["agents"].get("human_review") or {}).get("before_run_decision") is None
    assert updated["meta"].get("current_code_hash") is None


def test_detect_two_node_cycle_ignores_expected_execute_error_retry_pair() -> None:
    _install_langchain_and_langgraph_stubs()
    orchestrator = importlib.import_module("graph.nodes.orchestrator")

    trace = [
        "orchestrator", "execute_code", "orchestrator", "error_handler",
        "orchestrator", "execute_code", "orchestrator", "error_handler",
    ]
    assert orchestrator._detect_two_node_cycle(trace) is False


def test_apply_loop_guard_stops_repeated_execute_code_spam() -> None:
    _install_langchain_and_langgraph_stubs()
    orchestrator = importlib.import_module("graph.nodes.orchestrator")

    state = {
        "meta": {
            "workflow_trace": [
                "orchestrator", "execute_code", "orchestrator", "execute_code",
                "orchestrator", "execute_code", "orchestrator", "execute_code",
            ]
        }
    }

    action, observations, fired = orchestrator._apply_loop_guards(
        "execute_code", state, []
    )

    assert fired is True
    assert action == "end"
    assert observations and "execute_code" in observations[-1]


def test_apply_loop_guard_allows_human_requested_generate_code_regeneration() -> None:
    _install_langchain_and_langgraph_stubs()
    orchestrator = importlib.import_module("graph.nodes.orchestrator")

    state = {
        "meta": {
            "loop_guard_bypass_actions": ["generate_code"],
            "workflow_trace": [
                "orchestrator", "generate_code", "orchestrator", "generate_code",
                "orchestrator", "generate_code", "orchestrator", "generate_code",
            ],
        }
    }

    action, observations, fired = orchestrator._apply_loop_guards(
        "generate_code", state, []
    )

    assert fired is False
    assert action == "generate_code"
    assert observations and "bypass for human-requested action 'generate_code'" in observations[-1]


def test_orchestrator_final_review_regenerate_routes_back_to_generate_code() -> None:
    _install_langchain_and_langgraph_stubs()
    orchestrator = importlib.import_module("graph.nodes.orchestrator")

    state = {
        "messages": [SimpleNamespace(type="ai", content="Please review final output")],
        "output": {
            "generated_code": "print('old')",
            "text": "old output",
            "figure_png": b"old-png",
        },
        "observations": [],
        "last_action": "human_review_final",
        "orchestrator": {"next_action": "end"},
        "agents": {
            "executor": {"run_status": "ok"},
            "human_review": {
                "before_run_decision": "approve",
                "approved_code_hash": "abc",
                "final_decision": "regenerate",
            },
        },
        "meta": {
            "error_iterations": 0,
            "workflow_trace": ["orchestrator", "human_review_final"],
            "current_code_hash": "abc",
        },
    }

    updated = orchestrator.orchestrator_node(
        state,
        _LLM("not-json"),
        ["qa", "generate_code", "execute_code", "human_review_final", "end"],
    )

    assert updated["next_action"] == "generate_code"
    assert updated["output"].get("generated_code") is None
    assert updated["output"].get("text") is None
    assert updated["meta"].get("current_code_hash") is None
    assert (updated["agents"].get("human_review") or {}).get("final_decision") is None
    assert (updated["agents"].get("executor") or {}).get("run_status") == "idle"


def test_orchestrator_resets_workflow_trace_on_new_user_turn() -> None:
    _install_langchain_and_langgraph_stubs()
    orchestrator = importlib.import_module("graph.nodes.orchestrator")

    state = {
        "messages": [SimpleNamespace(type="human", content="perform survival analysis on my attached data")],
        "output": {},
        "observations": [],
        "last_action": "end",
        "orchestrator": {},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "final_decision": None},
        },
        "meta": {
            "error_iterations": 0,
            "last_user_message_hash": "old-hash",
            "workflow_trace": [
                "orchestrator", "execute_code", "orchestrator", "execute_code",
                "orchestrator", "execute_code", "orchestrator", "execute_code",
            ],
        },
    }

    updated = orchestrator.orchestrator_node(
        state,
        _LLM(json.dumps({"action": "generate_code"})),
        ["qa", "generate_code", "execute_code", "end"],
    )

    assert updated["next_action"] == "generate_code"
    assert updated["meta"]["workflow_trace"] == ["orchestrator"]


def test_orchestrator_treats_same_text_new_message_id_as_new_turn() -> None:
    _install_langchain_and_langgraph_stubs()
    orchestrator = importlib.import_module("graph.nodes.orchestrator")

    state = {
        "messages": [SimpleNamespace(type="human", id="new-msg-id", content="What's the weather today")],
        "output": {"generated_code": "print('stale')"},
        "observations": [],
        "last_action": "generate_code",
        "orchestrator": {},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "final_decision": None},
        },
        "meta": {
            "error_iterations": 0,
            "intent": "code",
            "last_user_message_hash": "old-hash-from-prior-message",
            "workflow_trace": ["orchestrator", "generate_code"],
        },
    }

    updated = orchestrator.orchestrator_node(
        state,
        _LLM("not-json"),
        ["qa", "generate_code", "execute_code", "end"],
    )

    assert updated["meta"].get("intent") == "qa"
    assert updated["output"].get("generated_code") is None

def test_planner_two_stage_selection_uses_ranked_ready_candidate() -> None:
    _install_langchain_and_langgraph_stubs()
    planner = importlib.import_module("graph.nodes.orchestrator.planner")

    state = {
        "messages": [SimpleNamespace(type="human", content="continue")],
        "output": {"generated_code": "print('x')"},
        "agents": {
            "executor": {"run_status": "ok"},
            "human_review": {
                "before_run_decision": "approve",
                "approved_code_hash": "h1",
                "final_decision": None,
            },
        },
        "meta": {"workflow_trace": ["orchestrator"], "error_iterations": 0, "current_code_hash": "h1"},
        "last_action": "execute_code",
    }

    llm = _SeqLLM([
        json.dumps({
            "action": "execute_code",
            "ranked_actions": ["execute_code", "human_review_final"],
            "thought": "ranking"
        }),
        json.dumps({"verdict": "accept", "reason": "ok"}),
    ])

    action, thought = planner.llm_select_next_action(
        state, llm, ["generate_code", "execute_code", "human_review_final", "end"]
    )

    assert action == "human_review_final"
    assert "ranked=" in thought


def test_planner_lightweight_action_critic_can_correct_action() -> None:
    _install_langchain_and_langgraph_stubs()
    planner = importlib.import_module("graph.nodes.orchestrator.planner")

    state = {
        "messages": [SimpleNamespace(type="human", content="continue")],
        "output": {"generated_code": "print('x')"},
        "agents": {
            "executor": {"run_status": "ok"},
            "human_review": {
                "before_run_decision": "approve",
                "approved_code_hash": "h1",
                "final_decision": None,
            },
        },
        "meta": {"workflow_trace": ["orchestrator"], "error_iterations": 0, "current_code_hash": "h1"},
        "last_action": "execute_code",
    }

    llm = _SeqLLM([
        json.dumps({
            "action": "execute_code",
            "ranked_actions": ["execute_code"],
            "thought": "execute"
        }),
        json.dumps({
            "verdict": "reject",
            "corrected_action": "human_review_final",
            "reason": "needs final approval"
        }),
    ])

    action, thought = planner.llm_select_next_action(
        state, llm, ["generate_code", "execute_code", "human_review_final", "end"]
    )

    assert action == "human_review_final"
    assert "critic_reason=needs final approval" in thought


def test_planner_can_skip_critic_in_fast_mode() -> None:
    _install_langchain_and_langgraph_stubs()
    planner = importlib.import_module("graph.nodes.orchestrator.planner")

    state = {
        "messages": [SimpleNamespace(type="human", content="continue")],
        "output": {"generated_code": "print('x')"},
        "agents": {
            "executor": {"run_status": "ok"},
            "human_review": {
                "before_run_decision": "approve",
                "approved_code_hash": "h1",
                "final_decision": None,
            },
        },
        "meta": {"workflow_trace": ["orchestrator"], "error_iterations": 0, "current_code_hash": "h1"},
        "last_action": "execute_code",
    }

    llm = _SeqLLM([
        json.dumps({
            "action": "execute_code",
            "ranked_actions": ["execute_code", "human_review_final"],
            "thought": "ranking"
        }),
    ])

    prev = os.environ.get("ORCH_CRITIC_MODE")
    os.environ["ORCH_CRITIC_MODE"] = "off"
    try:
        action, thought = planner.llm_select_next_action(
            state, llm, ["generate_code", "execute_code", "human_review_final", "end"]
        )
    finally:
        if prev is None:
            os.environ.pop("ORCH_CRITIC_MODE", None)
        else:
            os.environ["ORCH_CRITIC_MODE"] = prev

    assert action == "human_review_final"
    assert len(llm.calls) == 1

def test_orchestrator_overrides_stale_code_intent_with_latest_qa_intent() -> None:
    _install_langchain_and_langgraph_stubs()
    orchestrator = importlib.import_module("graph.nodes.orchestrator")

    state = {
        "messages": [SimpleNamespace(type="human", content="What's the weather today?")],
        "output": {},
        "observations": [],
        "last_action": "generate_code",
        "orchestrator": {},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "final_decision": None},
        },
        "meta": {
            "error_iterations": 0,
            "intent": "code",  # stale from previous turn
            "workflow_trace": ["orchestrator", "generate_code"],
        },
    }

    updated = orchestrator.orchestrator_node(
        state,
        _LLM("not-json"),
        ["qa", "generate_code", "execute_code", "end"],
    )

    assert updated["meta"].get("intent") == "qa"
    assert updated["next_action"] == "qa"


def test_orchestrator_prefers_policy_qa_when_planner_misroutes_tool_question() -> None:
    _install_langchain_and_langgraph_stubs()
    orchestrator = importlib.import_module("graph.nodes.orchestrator")

    state = {
        "messages": [SimpleNamespace(type="human", content="Search online for the latest FDA guidance on GLP-1 drugs")],
        "output": {},
        "observations": [],
        "last_action": "orchestrator",
        "orchestrator": {},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "final_decision": None},
        },
        "meta": {
            "error_iterations": 0,
            "workflow_trace": ["orchestrator"],
        },
    }

    updated = orchestrator.orchestrator_node(
        state,
        _LLM(json.dumps({"action": "generate_code", "thought": "misrouted"})),
        ["qa", "generate_code", "execute_code", "end"],
    )

    assert updated["meta"].get("intent") == "qa"
    assert updated["next_action"] == "qa"


def test_orchestrator_routes_whats_weather_today_to_qa() -> None:
    _install_langchain_and_langgraph_stubs()
    orchestrator = importlib.import_module("graph.nodes.orchestrator")

    state = {
        "messages": [SimpleNamespace(type="human", content="what's weather today?")],
        "output": {},
        "observations": [],
        "last_action": "orchestrator",
        "orchestrator": {},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "final_decision": None},
        },
        "meta": {
            "error_iterations": 0,
            "workflow_trace": ["orchestrator"],
        },
    }

    updated = orchestrator.orchestrator_node(
        state,
        _LLM(json.dumps({"action": "generate_code", "thought": "misrouted"})),
        ["qa", "generate_code", "execute_code", "end"],
    )

    assert updated["meta"].get("intent") == "qa"
    assert updated["next_action"] == "qa"


def test_orchestrator_prefers_policy_generate_code_for_explicit_code_intent_when_planner_says_qa() -> None:
    _install_langchain_and_langgraph_stubs()
    orchestrator = importlib.import_module("graph.nodes.orchestrator")

    state = {
        "messages": [SimpleNamespace(type="human", content="Write Python code to analyze my attached CSV and plot a Kaplan-Meier curve")],
        "output": {},
        "observations": [],
        "last_action": "orchestrator",
        "orchestrator": {},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "final_decision": None},
        },
        "meta": {
            "error_iterations": 0,
            "workflow_trace": ["orchestrator"],
        },
    }

    updated = orchestrator.orchestrator_node(
        state,
        _LLM(json.dumps({"action": "qa", "thought": "misrouted"})),
        ["qa", "generate_code", "execute_code", "end"],
    )

    assert updated["meta"].get("intent") == "code"
    assert updated["next_action"] == "generate_code"


def test_orchestrator_does_not_fallback_to_generate_code_for_bare_followup_with_stale_code_intent() -> None:
    _install_langchain_and_langgraph_stubs()
    orchestrator = importlib.import_module("graph.nodes.orchestrator")

    state = {
        "messages": [SimpleNamespace(type="human", content="Boston")],
        "output": {},
        "observations": [],
        "last_action": "generate_code",
        "orchestrator": {},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "final_decision": None},
        },
        "meta": {
            "error_iterations": 0,
            "intent": "code",
            "workflow_trace": ["orchestrator", "generate_code"],
        },
    }

    updated = orchestrator.orchestrator_node(
        state,
        _LLM("not-json"),
        ["qa", "generate_code", "execute_code", "end"],
    )

    assert updated["next_action"] != "generate_code"
