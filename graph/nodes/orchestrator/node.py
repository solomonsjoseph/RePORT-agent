from __future__ import annotations

from typing import Iterable

from ...state import AgentState, MetaKeys
from ...state_views import get_planner_state, merge_state_patch
from .action_mask import mask_actions
from .loop_guards import _apply_loop_guards
from .planner import llm_plan_next_action
from .policy import (
    _next_tool_requester,
    _tool_request_queue,
    choose_invariant_action,
    choose_next_action,
)
from .progress import update_progress_tracking
from .state_logic import (
    _consume_final_review_regenerate,
    _consume_regenerate_before_run,
    _should_end_now,
    _user_message_hash,
)


def _record_planner_decision(
    state: AgentState,
    planner_action: str,
    thought: str,
    routed_action: str,
) -> AgentState:
    planner = get_planner_state(state)
    trace = list(planner.get("decision_trace") or [])
    decision = {
        "action": planner_action,
        "thought": thought,
        "after": state.get("last_action"),
        "routed_action": routed_action,
    }
    trace.append(decision)
    planner["last_decision"] = decision
    planner["decision_trace"] = trace[-20:]
    return merge_state_patch(state, {"planner": planner})


def orchestrator_node(state: AgentState, llm, available_actions: Iterable[str]) -> AgentState:
    state = update_progress_tracking(state)
    available_action_list = sorted(set(available_actions))
    available_action_set = set(available_action_list)
    orchestrator_state = dict(state.get("orchestrator", {}))
    next_action = orchestrator_state.get("next_action")
    thought = orchestrator_state.get("thought", "")
    meta = dict(state.get("meta") or {})

    output = dict(state.get("output") or {})
    agents = dict(state.get("agents") or {})
    current_hash = _user_message_hash(state)
    was_awaiting_clarification = bool(meta.get(MetaKeys.AWAITING_USER_CLARIFICATION))
    review_state = dict(agents.get("human_review") or {})
    has_pending_regenerate = (
        review_state.get("before_run_decision") == "regenerate"
        or review_state.get("final_decision") == "regenerate"
    )

    if (
        current_hash
        and current_hash != meta.get(MetaKeys.LAST_USER_MESSAGE_HASH)
        and not has_pending_regenerate
    ):
        if was_awaiting_clarification:
            meta[MetaKeys.LAST_USER_MESSAGE_HASH] = current_hash
            if "clarification" in available_action_set:
                next_action = "clarification"
            else:
                clarification_return = (
                    meta.get(MetaKeys.CLARIFICATION_RETURN_NODE)
                    or state.get("last_action")
                    or "qa"
                )
                if clarification_return in available_action_set:
                    next_action = clarification_return
                else:
                    next_action = "qa"
            orchestrator_state["next_action"] = next_action
        else:
            meta[MetaKeys.LAST_USER_MESSAGE_HASH] = current_hash
            next_action = None
            orchestrator_state.pop("next_action", None)

    output, agents, meta, regenerated = _consume_regenerate_before_run(output, agents, meta)
    if regenerated:
        next_action = None
        orchestrator_state.pop("next_action", None)
        observations = list(state.get("observations", []))
        observations.append("orchestrator: received regenerate request; routing back to generate_code")
        state = {**state, "observations": observations}

    output, agents, meta, regenerated_final = _consume_final_review_regenerate(output, agents, meta)
    if regenerated_final:
        next_action = None
        orchestrator_state.pop("next_action", None)
        observations = list(state.get("observations", []))
        observations.append(
            "orchestrator: received final-review regenerate request; routing back to generate_code"
        )
        state = {**state, "observations": observations}

    routing_state = {
        **state,
        "output": output,
        "agents": agents,
        "meta": meta,
    }

    if not next_action:
        if state.get("last_action") == "tool_handler":
            requester = _next_tool_requester(routing_state)
            if requester and requester in available_action_set:
                next_action = requester
                queue = _tool_request_queue(routing_state)
                meta[MetaKeys.TOOL_REQUEST_QUEUE] = [n for n in queue if n != requester]
                routing_state["meta"] = meta

        if not next_action:
            invariant_action = choose_invariant_action(routing_state, available_action_list)
            if invariant_action:
                next_action = invariant_action
            else:
                llm_choice, thought, planner_action = llm_plan_next_action(
                    routing_state, llm, available_action_list
                )
                masked_actions, _blocked = mask_actions(routing_state, available_action_list)
                fallback_action = choose_next_action(routing_state, available_action_list)
                if llm_choice != "end" and llm_choice in masked_actions and not _should_end_now(routing_state):
                    next_action = llm_choice
                else:
                    next_action = fallback_action
                routing_state = _record_planner_decision(
                    routing_state,
                    planner_action,
                    thought,
                    next_action,
                )

    state = routing_state
    orchestrator_state = dict(state.get("orchestrator", {}))
    meta = dict(state.get("meta") or {})

    observations = list(state.get("observations", []))
    next_action, observations, _guard_fired = _apply_loop_guards(
        next_action or "end", routing_state, observations
    )

    orchestrator_state["next_action"] = next_action
    if thought:
        orchestrator_state["thought"] = thought

    observations.append(f"orchestrator: next_action={next_action}")

    workflow_trace = list(meta.get(MetaKeys.WORKFLOW_TRACE, []))
    workflow_trace.append("orchestrator")
    meta[MetaKeys.WORKFLOW_TRACE] = workflow_trace[-100:]

    return {
        **state,
        "next_action": next_action,
        "last_action": state.get("last_action"),
        "orchestrator": orchestrator_state,
        "planner": dict(state.get("planner") or {}),
        "observations": observations,
        "output": output,
        "agents": agents,
        "meta": meta,
    }
