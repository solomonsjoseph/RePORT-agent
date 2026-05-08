from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.llm_response import coerce_text_content
from utils.performance import collect_timings, timing_stage


class _FakePrompt:
    def format_prompt(self, **_kwargs):
        return self

    def to_messages(self):
        return []


def _load_llm_plan_next_action():
    source = Path("graph/nodes/orchestrator/planner.py").read_text()
    tree = ast.parse(source)
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_raw_response_preview",
            "_parse_planner_response",
            "llm_plan_next_action",
        }
    ]
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "AgentState": dict,
        "Iterable": Iterable,
        "PLANNER_RAW_RESPONSE_PREVIEW_CHARS": 120,
        "build_planner_context": _build_planner_context,
        "build_planner_runtime_state": _build_planner_runtime_state,
        "coerce_text_content": coerce_text_content,
        "json": json,
        "make_planner_prompt": lambda: _FakePrompt(),
        "mask_actions": _mask_actions,
        "timing_stage": timing_stage,
    }
    exec(compile(module, "graph/nodes/orchestrator/planner.py", "exec"), namespace)
    return namespace["llm_plan_next_action"]


def _build_planner_runtime_state(state, _available_actions):
    return state


def _mask_actions(_planner_state, available_actions):
    return list(available_actions), {}


def _build_planner_context(_planner_state, _masked_actions):
    return {
        "environment_summary": "latest_user_message=What is this app?",
        "planner_environment": {},
        "planner_memory": {},
        "recent_observations": [],
        "planner_decision_trace": [],
        "node_capabilities": "- qa: answer questions",
        "recent_turns_for_planner": [],
    }


class _FakePlannerLlm:
    def invoke(self, _messages):
        payload = {
            "thought": "The user asked a normal question.",
            "action": "qa",
            "route_reason": "Conceptual question.",
            "referenced_task_id": None,
            "relationship": None,
            "dataset_id": None,
            "confidence": 0.9,
            "needs_clarification": False,
            "clarification_question": None,
            "ranked_actions": ["qa", "end"],
        }
        return SimpleNamespace(content=json.dumps(payload))


def test_llm_plan_next_action_records_internal_timing_stages() -> None:
    llm_plan_next_action = _load_llm_plan_next_action()
    state = {
        "messages": [SimpleNamespace(type="human", content="What is this app?")],
        "meta": {"workflow_trace": []},
        "planner": {"memory": {}, "decision_trace": []},
        "agents": {},
        "artifacts": {},
        "observations": [],
    }

    with collect_timings() as records:
        result = llm_plan_next_action(state, _FakePlannerLlm(), ["qa", "end"])

    assert result[0] == "qa"
    stages = [record["stage"] for record in records]
    assert "planner.total" in stages
    assert "planner.runtime_state" in stages
    assert "planner.action_mask" in stages
    assert "planner.context_build" in stages
    assert "planner.prompt_format" in stages
    assert "planner.llm_invoke" in stages
    assert "planner.parse_response" in stages
