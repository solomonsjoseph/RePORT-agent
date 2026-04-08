from __future__ import annotations

from typing import Iterable

from ...state import AgentState, MetaKeys
from ...state_views import get_planner_state, merge_state_patch
from .action_mask import mask_actions
from .planner import llm_plan_next_action
from .policy import (
    _next_tool_requester,
    _tool_request_queue,
)
from .progress_controller import apply_recurrence_guard, update_recurrence_state
from .state_logic import (
    _consume_after_error_decision,
    _consume_before_run_approval,
    _consume_final_review_regenerate,
    _consume_final_review_approval,
    _consume_regenerate_before_run,
    _has_unanswered_human_message,
    _user_message_hash,
)
from .workflow_status import derive_workflow_status


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


def _planner_fallback_action(available_actions: Iterable[str]) -> str:
    """Return a stable semantic default when planner output is unusable."""
    available = set(available_actions)
    if "qa" in available:
        return "qa"
    if "generate_code" in available:
        return "generate_code"
    return "end"


def _resumed_from(state: AgentState, node_name: str) -> bool:
    if state.get("last_action") == node_name:
        return True
    workflow_trace = list((state.get("meta") or {}).get(MetaKeys.WORKFLOW_TRACE, []))
    return bool(workflow_trace and workflow_trace[-1] == node_name)


def _store_workflow_status(state: AgentState) -> AgentState:
    meta = dict(state.get("meta") or {})
    status = derive_workflow_status(state)
    meta[MetaKeys.WORKFLOW_MILESTONE] = status["milestone"]
    meta[MetaKeys.COMPLETION_STATUS] = status["completion_status"]
    meta[MetaKeys.BLOCKER_SIGNATURE] = status["blocker_signature"]
    return {**state, "meta": meta}


def _should_end_for_completion(state: AgentState) -> bool:
    meta = dict(state.get("meta") or {})
    completion = meta.get(MetaKeys.COMPLETION_STATUS)
    if completion == "complete":
        return True
    if completion != "blocked_waiting":
        return False
    if meta.get(MetaKeys.AWAITING_USER_CLARIFICATION) and not _has_unanswered_human_message(state):
        return True
    return state.get("last_action") in {
        "clarification",
        "human_review_before_run",
        "human_review_after_error",
        "human_review_final",
        "terminal_execution_error",
    }


def orchestrator_node(state: AgentState, llm, available_actions: Iterable[str]) -> AgentState:
    available_action_list = sorted(set(available_actions))
    available_action_set = set(available_action_list)
    orchestrator_state = dict(state.get("orchestrator", {}))
    next_action = orchestrator_state.get("next_action")
    thought = orchestrator_state.get("thought", "")
    meta = dict(state.get("meta") or {})
    transition_selected_from_resume = False

    output = dict(state.get("output") or {})
    agents = dict(state.get("agents") or {})
    if _resumed_from(state, "human_review_before_run"):
        output, agents, meta, regenerated = _consume_regenerate_before_run(output, agents, meta)
        if regenerated:
            next_action = "generate_code"
            transition_selected_from_resume = True
            orchestrator_state.pop("next_action", None)
            observations = list(state.get("observations", []))
            observations.append(
                "orchestrator: received regenerate request; routing back to generate_code"
            )
            state = {**state, "observations": observations}

        output, agents, meta, approved_before_run = _consume_before_run_approval(output, agents, meta)
        if approved_before_run:
            next_action = "execute_code"
            transition_selected_from_resume = True
            orchestrator_state.pop("next_action", None)
            observations = list(state.get("observations", []))
            observations.append("orchestrator: consumed before-run approval; routing to execute_code")
            state = {**state, "observations": observations}

    if _resumed_from(state, "human_review_after_error"):
        output, agents, meta, after_error_action = _consume_after_error_decision(output, agents, meta)
        if after_error_action:
            next_action = after_error_action
            transition_selected_from_resume = True
            orchestrator_state.pop("next_action", None)
            observations = list(state.get("observations", []))
            observations.append(
                "orchestrator: consumed after-error review decision; routing to generate_code"
            )
            state = {**state, "observations": observations}

    if _resumed_from(state, "human_review_final"):
        output, agents, meta, regenerated_final = _consume_final_review_regenerate(output, agents, meta)
        if regenerated_final:
            next_action = "generate_code"
            transition_selected_from_resume = True
            orchestrator_state.pop("next_action", None)
            observations = list(state.get("observations", []))
            observations.append(
                "orchestrator: received final-review regenerate request; routing back to generate_code"
            )
            state = {**state, "observations": observations}

        output, agents, meta, approved_final = _consume_final_review_approval(output, agents, meta)
        if approved_final:
            next_action = "end"
            transition_selected_from_resume = True
            orchestrator_state.pop("next_action", None)
            observations = list(state.get("observations", []))
            observations.append("orchestrator: consumed final approval; routing to end")
            state = {**state, "observations": observations}

    current_hash = _user_message_hash(state)
    if current_hash and transition_selected_from_resume and next_action in {"generate_code", "execute_code", "end"}:
        meta[MetaKeys.LAST_USER_MESSAGE_HASH] = current_hash

    was_awaiting_clarification = bool(meta.get(MetaKeys.AWAITING_USER_CLARIFICATION))
    review_state = dict(agents.get("human_review") or {})
    has_pending_regenerate = (
        review_state.get("before_run_decision") == "regenerate"
        or review_state.get("final_decision") == "regenerate"
    )

    if (
        current_hash
        and _has_unanswered_human_message(state)
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

    routing_state = _store_workflow_status(routing_state)
    routing_state = update_recurrence_state(routing_state)

    if not next_action and _should_end_for_completion(routing_state):
        next_action = "end"

    if not next_action:
        llm_choice, thought, planner_action = llm_plan_next_action(
            routing_state, llm, available_action_list
        )
        masked_actions, _blocked = mask_actions(routing_state, available_action_list)
        if llm_choice != "end" and llm_choice in masked_actions:
            next_action = llm_choice
        else:
            next_action = _planner_fallback_action(masked_actions)
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
    next_action, observations, _guard_fired = apply_recurrence_guard(
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
