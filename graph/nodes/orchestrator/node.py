from __future__ import annotations

from typing import Iterable

from ...state import AgentState, MetaKeys
from .intent import infer_intent_from_latest_user
from .loop_guards import _apply_loop_guards
from .planner import llm_select_next_action
from .policy import _next_tool_requester, _tool_request_queue, choose_next_action
from .state_logic import (
    _consume_final_review_regenerate,
    _consume_regenerate_before_run,
    _reset_for_new_turn,
    _should_end_now,
    _user_message_hash,
)


def orchestrator_node(state: AgentState, llm, available_actions: Iterable[str]) -> AgentState:
    orchestrator_state = dict(state.get("orchestrator", {}))
    next_action = orchestrator_state.get("next_action")
    thought = orchestrator_state.get("thought", "")
    meta = dict(state.get("meta") or {})

    output = dict(state.get("output") or {})
    agents = dict(state.get("agents") or {})
    current_hash = _user_message_hash(state)
    was_awaiting_clarification = bool(meta.get(MetaKeys.AWAITING_USER_CLARIFICATION))

    if current_hash and current_hash != meta.get(MetaKeys.LAST_USER_MESSAGE_HASH):
        if was_awaiting_clarification:
            meta.pop(MetaKeys.AWAITING_USER_CLARIFICATION, None)
            meta[MetaKeys.LAST_USER_MESSAGE_HASH] = current_hash
            clarification_return = (
                meta.get(MetaKeys.CLARIFICATION_RETURN_NODE)
                or state.get("last_action")
                or "qa"
            )
            if clarification_return in set(available_actions):
                next_action = clarification_return
            else:
                next_action = "qa"
            orchestrator_state["next_action"] = next_action
        else:
            output, agents, meta = _reset_for_new_turn(output, agents, meta)
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

    inferred_intent = infer_intent_from_latest_user(state)
    # Keep intent aligned with latest user message when it is confidently inferred.
    # This prevents stale intent (e.g., previous code turn) from forcing bad routes
    # on subsequent QA-style asks.
    if inferred_intent:
        meta[MetaKeys.INTENT] = inferred_intent

    if not next_action:
        if state.get("last_action") == "tool_handler":
            requester = _next_tool_requester(routing_state)
            if requester and requester in set(available_actions):
                next_action = requester
                queue = _tool_request_queue(routing_state)
                meta[MetaKeys.TOOL_REQUEST_QUEUE] = [n for n in queue if n != requester]

        if not next_action:
            llm_choice, thought = llm_select_next_action(routing_state, llm, available_actions)
            fallback_action = choose_next_action(routing_state, available_actions)

            if llm_choice != "end" and not _should_end_now(routing_state):
                next_action = llm_choice
                if not meta.get(MetaKeys.INTENT):
                    if llm_choice == "qa":
                        meta[MetaKeys.INTENT] = "qa"
                    elif llm_choice in ("generate_code", "execute_code", "error_handler"):
                        meta[MetaKeys.INTENT] = "code"
            else:
                next_action = fallback_action

    observations = list(state.get("observations", []))
    next_action, observations, _guard_fired = _apply_loop_guards(
        next_action or "end", routing_state, observations
    )

    orchestrator_state["next_action"] = next_action
    if thought:
        orchestrator_state["thought"] = thought
        thoughts = list(orchestrator_state.get("thoughts", []))
        thoughts.append(thought)
        orchestrator_state["thoughts"] = thoughts

    observations.append(f"orchestrator: next_action={next_action}")

    workflow_trace = list(meta.get(MetaKeys.WORKFLOW_TRACE, []))
    workflow_trace.append("orchestrator")
    meta[MetaKeys.WORKFLOW_TRACE] = workflow_trace[-100:]

    return {
        **state,
        "next_action": next_action,
        "last_action": state.get("last_action"),
        "orchestrator": orchestrator_state,
        "observations": observations,
        "output": output,
        "agents": agents,
        "meta": meta,
    }
