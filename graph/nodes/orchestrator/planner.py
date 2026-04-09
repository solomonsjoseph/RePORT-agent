from __future__ import annotations

import json
from typing import Iterable

from prompts.planner_prompt import make_planner_prompt

from ...state import AgentState
from .action_mask import mask_actions
from .context_builder import build_planner_context, build_planner_runtime_state
from utils.llm_response import coerce_text_content

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


def llm_plan_next_action(
    state: AgentState,
    llm,
    available_actions: Iterable[str],
) -> tuple[str, str, str]:
    available_action_list = sorted(set(available_actions))
    planner_state = build_planner_runtime_state(state, available_action_list)
    masked_actions, mask_reasons = mask_actions(planner_state, available_action_list)
    planner_context = build_planner_context(planner_state, masked_actions)

    blocked_actions = [
        f"{action} ({mask_reasons[action]})"
        for action in available_action_list
        if action in mask_reasons
    ]

    planner_prompt = make_planner_prompt().format_prompt(
        actions=", ".join(masked_actions) if masked_actions else "none",
        environment_summary=planner_context["environment_summary"],
        planner_memory=json.dumps(
            planner_context["planner_memory"], default=str, ensure_ascii=False
        ),
        recent_observations=json.dumps(
            planner_context["recent_observations"], default=str, ensure_ascii=False
        ),
        planner_decision_trace=json.dumps(
            planner_context["planner_decision_trace"], default=str, ensure_ascii=False
        ),
        node_capabilities=planner_context["node_capabilities"],
        blocked_actions="\n".join(f"- {a}" for a in blocked_actions) if blocked_actions else "none",
        recent_turns_for_planner=(
            json.dumps(
                planner_context["recent_turns_for_planner"], default=str, ensure_ascii=False
            )
            if planner_context["recent_turns_for_planner"]
            else "none"
        ),
    )
    planner_response = llm.invoke(planner_prompt.to_messages())
    planner_action, thought, ranked_actions = _parse_planner_response(
        coerce_text_content(getattr(planner_response, "content", "")),
        available_action_list,
    )
    final_action = planner_action if planner_action in masked_actions else "end"
    masked_ranked_actions = [action for action in ranked_actions if action in masked_actions]
    if final_action == "end" and masked_ranked_actions:
        final_action = masked_ranked_actions[0]
    trace_action = planner_action
    if masked_ranked_actions and final_action == masked_ranked_actions[0] and final_action != planner_action:
        trace_action = final_action

    if masked_ranked_actions:
        suffix = f"ranked={masked_ranked_actions}"
        thought = f"{thought} {suffix}".strip() if thought else suffix

    return final_action, thought, trace_action


def llm_select_next_action(
    state: AgentState,
    llm,
    available_actions: Iterable[str],
) -> tuple[str, str]:
    final_action, thought, _planner_action = llm_plan_next_action(state, llm, available_actions)
    return final_action, thought
