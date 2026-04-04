from __future__ import annotations

import json
from typing import Iterable

from prompts.planner_prompt import make_planner_prompt

from ...state import AgentState, MetaKeys
from ..node_registry import NODE_CAPABILITIES, NODE_REGISTRY
from ..state_helpers import get_agent_state
from ..tool_routing import is_tool_requested
from .constants import LOOP_GUARD_LOOKBACK
from .loop_guards import _count_action_in_recent_trace, _detect_two_node_cycle
from .state_logic import _latest_user_message
from utils.llm_response import coerce_text_content


def _format_tool_results(state: AgentState) -> str:
    agents = state.get("agents", {})
    formatted: list[str] = []
    for agent_name, agent_state in agents.items():
        results = agent_state.get("tool_results", [])
        if not results:
            continue
        for result in results[-3:]:
            formatted.append(
                f"{agent_name}: {json.dumps(result, default=str, ensure_ascii=False)}"
            )
    return "\n".join(formatted) if formatted else "none"


def _format_state_summary(state: AgentState) -> str:
    executor_state = get_agent_state(state, "executor")
    review_state = get_agent_state(state, "human_review")

    latest_user = _latest_user_message(state)
    trace = list((state.get("meta") or {}).get(MetaKeys.WORKFLOW_TRACE, []))
    trace_tail = trace[-LOOP_GUARD_LOOKBACK:]
    cycle_detected = _detect_two_node_cycle(trace)
    max_repeats = max(
        (_count_action_in_recent_trace(a, trace) for a in set(trace_tail)),
        default=0,
    )

    parts = [
        f"latest_user_message={latest_user}",
        f"generated_code_present={bool((state.get('output') or {}).get('generated_code'))}",
        f"executor_run_status={executor_state.get('run_status')}",
        f"before_run_decision={review_state.get('before_run_decision')}",
        f"final_decision={review_state.get('final_decision')}",
        f"error_iterations={(state.get('meta') or {}).get(MetaKeys.ERROR_ITERATIONS, 0)}",
        f"tool_requests_pending={is_tool_requested(state)}",
        f"last_action={state.get('last_action')}",
        f"workflow_trace_tail={trace_tail}",
        f"loop_cycle_detected={cycle_detected}",
        f"max_action_repeats_in_window={max_repeats}",
    ]

    agents = state.get("agents") or {}
    for nd in sorted(NODE_REGISTRY, key=lambda n: n.priority):
        agent_st = agents.get(nd.name, {})
        status = agent_st.get("status")
        if status and status not in ("idle", None):
            parts.append(f"{nd.name}_status={status}")

    parts.append("tool_results:\n" + _format_tool_results(state))
    return "\n".join(parts)


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


def _format_node_capabilities(available_actions: Iterable[str]) -> str:
    lines: list[str] = []
    for action in sorted(set(available_actions)):
        description = NODE_CAPABILITIES.get(action, "No description provided.")
        lines.append(f"- {action}: {description}")
    return "\n".join(lines)


def llm_select_next_action(
    state: AgentState,
    llm,
    available_actions: Iterable[str],
) -> tuple[str, str]:
    actions = ", ".join(sorted(set(available_actions)))
    ready_actions, blocked_actions = _action_affordances(state, available_actions)
    summary = _format_state_summary(state)

    planner_prompt = make_planner_prompt().format_prompt(
        actions=actions,
        summary=summary,
        node_capabilities=_format_node_capabilities(available_actions),
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
