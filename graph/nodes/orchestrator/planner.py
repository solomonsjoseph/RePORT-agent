from __future__ import annotations

import json
from typing import Iterable

from prompts.planner_prompt import make_planner_prompt

from ...state import AgentState, MetaKeys
from ..state_helpers import get_agent_state
from ..tool_routing import is_tool_requested
from ..node_registry import NODE_CAPABILITIES, NODE_REGISTRY
from .constants import LOOP_GUARD_LOOKBACK
from .intent import infer_intent_from_latest_user
from .loop_guards import _count_action_in_recent_trace, _detect_two_node_cycle
from .state_logic import _latest_user_message


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
    inferred_intent = infer_intent_from_latest_user(state)

    trace = list((state.get("meta") or {}).get(MetaKeys.WORKFLOW_TRACE, []))
    trace_tail = trace[-LOOP_GUARD_LOOKBACK:]
    cycle_detected = _detect_two_node_cycle(trace)
    max_repeats = max(
        (_count_action_in_recent_trace(a, trace) for a in set(trace_tail)),
        default=0,
    )

    parts = [
        f"latest_user_message={latest_user}",
        f"inferred_intent={inferred_intent}",
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


def _parse_planner_response(
    content: str,
    available_actions: Iterable[str],
) -> tuple[str, str]:
    content = content.strip()
    if not content:
        return "end", ""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        action = content
        return action if action in set(available_actions) else "end", ""
    if not isinstance(payload, dict):
        return "end", ""
    action = payload.get("action", "")
    thought = payload.get("thought", "")
    if action in set(available_actions):
        return action, str(thought or "")
    return "end", str(thought or "")


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
    prompt = make_planner_prompt().format_prompt(
        actions=actions,
        summary=_format_state_summary(state),
        node_capabilities=_format_node_capabilities(available_actions),
    )
    response = llm.invoke(prompt.to_messages())
    return _parse_planner_response(response.content, available_actions)
