from __future__ import annotations

import json
from typing import Iterable

from prompts.planner_prompt import make_planner_prompt

from ...state import AgentState
from ..node_registry import NODE_REGISTRY
from ..state_helpers import get_agent_state
from ..tool_routing import is_tool_requested
from .context_builder import build_planner_context
from utils.llm_response import coerce_text_content


def _action_affordances(state: AgentState, available_actions: Iterable[str]) -> tuple[list[str], list[str]]:
    """Return (ready_actions, blocked_actions_with_reasons)."""
    available = set(available_actions)
    ready: list[str] = []
    blocked: list[str] = []

    for nd in sorted(NODE_REGISTRY, key=lambda n: n.priority):
        if nd.name not in available:
            continue
        if nd.is_ready(state):
            ready.append(nd.name)
        else:
            blocked.append(nd.name)

    if "end" in available:
        blocked.append("end (prefer only when task complete or no ready action)")

    if not blocked:
        return ready, blocked

    # Add concise state-derived hints for common misroutes.
    executor_state = get_agent_state(state, "executor")
    review_state = get_agent_state(state, "human_review")
    run_status = executor_state.get("run_status")
    if (
        run_status == "ok"
        and review_state.get("final_decision") is None
        and "execute_code" in set(available_actions)
    ):
        blocked.append("execute_code (already succeeded; move to human_review_final)")

    if (
        bool((state.get("output") or {}).get("generated_code"))
        and review_state.get("before_run_decision") != "approve"
        and "execute_code" in set(available_actions)
    ):
        blocked.append("execute_code (requires fresh human approval)")

    return ready, blocked


def _parse_planner_response(
    content: str,
    available_actions: Iterable[str],
) -> tuple[str, str, list[str]]:
    content = content.strip()
    available = set(available_actions)
    if not content:
        return "end", "", []

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        action = content
        return (action if action in available else "end"), "", []

    if not isinstance(payload, dict):
        return "end", "", []

    action = payload.get("action", "")
    thought = str(payload.get("thought", "") or "")

    ranked_raw = payload.get("ranked_actions", [])
    ranked_actions: list[str] = []
    if isinstance(ranked_raw, list):
        for item in ranked_raw:
            candidate = str(item or "").strip()
            if candidate in available and candidate not in ranked_actions:
                ranked_actions.append(candidate)
            if len(ranked_actions) >= 3:
                break

    parsed_action = action if action in available else "end"
    return parsed_action, thought, ranked_actions


def llm_select_next_action(
    state: AgentState,
    llm,
    available_actions: Iterable[str],
) -> tuple[str, str]:
    available_action_list = sorted(set(available_actions))
    actions = ", ".join(available_action_list)
    ready_actions, blocked_actions = _action_affordances(state, available_actions)
    planner_context = build_planner_context(state, available_action_list)

    planner_prompt = make_planner_prompt().format_prompt(
        actions=actions,
        summary=(
            f"{planner_context['environment_summary']}\n"
            f"recent_observations={json.dumps(planner_context['recent_observations'], default=str, ensure_ascii=False)}\n"
            f"planner_decision_trace={json.dumps(planner_context['planner_decision_trace'], default=str, ensure_ascii=False)}"
        ),
        node_capabilities=planner_context["node_capabilities"],
        ready_actions=", ".join(ready_actions) if ready_actions else "none",
        blocked_actions="\n".join(f"- {a}" for a in blocked_actions) if blocked_actions else "none",
    )
    planner_response = llm.invoke(planner_prompt.to_messages())
    parsed_action, thought, ranked_actions = _parse_planner_response(
        coerce_text_content(getattr(planner_response, "content", "")),
        available_actions,
    )
    final_action = parsed_action
    if final_action == "end" and ranked_actions:
        final_action = ranked_actions[0]

    if ranked_actions:
        suffix = f"ranked={ranked_actions}"
        thought = f"{thought} {suffix}".strip() if thought else suffix

    return final_action, thought
