from __future__ import annotations

from typing import Any
from ..state import MetaKeys


def get_agent_state(state: dict, agent_name: str) -> dict[str, Any]:
    agents = dict(state.get("agents", {}))
    agent_state = dict(agents.get(agent_name, {}))
    agent_state.setdefault("status", "idle")
    agent_state.setdefault("run_status", "idle")
    agent_state.setdefault("tool_requests", [])
    agent_state.setdefault("tool_results", [])
    agent_state.setdefault("notes", [])
    agents[agent_name] = agent_state
    state["agents"] = agents
    return agent_state


def update_agent_state(state: dict, agent_name: str, updates: dict[str, Any]) -> dict[str, Any]:
    agent_state = get_agent_state(state, agent_name)
    agent_state.update(updates)
    agents = dict(state.get("agents", {}))
    agents[agent_name] = agent_state
    return {
        **state,
        "agents": agents,
    }


def clear_clarification_meta(meta: dict[str, Any] | None) -> dict[str, Any]:
    updated = dict(meta or {})
    updated.pop(MetaKeys.AWAITING_USER_CLARIFICATION, None)
    updated.pop(MetaKeys.PENDING_QUESTION, None)
    updated.pop(MetaKeys.PENDING_QUESTION_USER_MESSAGE_HASH, None)
    updated.pop(MetaKeys.CLARIFICATION_RETURN_NODE, None)
    updated.pop(MetaKeys.CLARIFICATION_KIND, None)
    updated.pop(MetaKeys.CLARIFICATION_EXPECTED, None)
    updated.pop(MetaKeys.CLARIFICATION_ATTEMPT_COUNT, None)
    updated.pop(MetaKeys.SEMANTIC_LAST_ACTION, None)
    return updated


def clear_analysis_dataset_meta(meta: dict[str, Any] | None) -> dict[str, Any]:
    updated = dict(meta or {})
    updated.pop(MetaKeys.ANALYSIS_DATASET_CANDIDATE_IDS, None)
    updated.pop(MetaKeys.ANALYSIS_DATASET_PENDING_REQUEST, None)
    return updated


def set_clarification_meta(
    meta: dict[str, Any] | None,
    *,
    return_node: str,
    kind: str,
    pending_question: str | None = None,
    pending_question_user_message_hash: str | None = None,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    updated = dict(meta or {})
    updated[MetaKeys.AWAITING_USER_CLARIFICATION] = True
    updated[MetaKeys.CLARIFICATION_RETURN_NODE] = return_node
    updated[MetaKeys.CLARIFICATION_KIND] = kind
    updated[MetaKeys.CLARIFICATION_ATTEMPT_COUNT] = 0
    if pending_question is not None:
        updated[MetaKeys.PENDING_QUESTION] = pending_question
    if pending_question_user_message_hash:
        updated[MetaKeys.PENDING_QUESTION_USER_MESSAGE_HASH] = pending_question_user_message_hash
    if expected is not None:
        updated[MetaKeys.CLARIFICATION_EXPECTED] = dict(expected)
    return updated
