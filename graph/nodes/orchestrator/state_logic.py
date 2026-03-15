from __future__ import annotations

import hashlib

from ...state import AgentState, MetaKeys
from ..node_registry import NODE_REGISTRY


def _latest_user_message(state: AgentState) -> str:
    messages = list(state.get("messages", []))
    for message in reversed(messages):
        if getattr(message, "type", None) == "human":
            return str(getattr(message, "content", "") or "").strip()
    return ""


def _has_unanswered_human_message(state: AgentState) -> bool:
    messages = list(state.get("messages", []))
    if not messages:
        return False
    last_human_index = -1
    last_ai_index = -1
    for idx, message in enumerate(messages):
        message_type = getattr(message, "type", None)
        if message_type == "human":
            last_human_index = idx
        elif message_type == "ai":
            last_ai_index = idx
    return last_human_index > last_ai_index


def _should_end_now(state: AgentState) -> bool:
    if (state.get("meta") or {}).get(MetaKeys.AWAITING_USER_CLARIFICATION):
        return True
    if state.get("last_action") == "qa" and not _has_unanswered_human_message(state):
        return True
    return False


def _user_message_hash(state: AgentState) -> str | None:
    msg = _latest_user_message(state)
    if not msg:
        return None
    return hashlib.sha256(msg.encode()).hexdigest()[:16]


def _reset_for_new_turn(output: dict, agents: dict, meta: dict) -> tuple[dict, dict, dict]:
    output = dict(output)
    output.pop("generated_code", None)
    output.pop("error", None)
    output.pop("tool_results", None)

    agents = dict(agents)
    keys_to_reset: set[str] = set()
    for nd in NODE_REGISTRY:
        keys_to_reset.update(nd.reset_agent_keys)

    for key in keys_to_reset:
        if key in agents:
            agents[key] = {}

    meta = dict(meta)
    meta.pop(MetaKeys.INTENT, None)
    meta.pop(MetaKeys.CURRENT_CODE_HASH, None)
    meta.pop(MetaKeys.AWAITING_USER_CLARIFICATION, None)
    meta.pop(MetaKeys.TOOL_REQUEST_QUEUE, None)
    meta.pop(MetaKeys.LOOP_GUARD_BYPASS_ACTIONS, None)
    meta[MetaKeys.ERROR_ITERATIONS] = 0

    return output, agents, meta


def _consume_regenerate_before_run(output: dict, agents: dict, meta: dict) -> tuple[dict, dict, dict, bool]:
    review = (agents.get("human_review") or {}) if isinstance(agents, dict) else {}
    if review.get("before_run_decision") != "regenerate":
        return output, agents, meta, False

    updated_output = dict(output)
    updated_output.pop("generated_code", None)

    updated_meta = dict(meta)
    updated_meta.pop(MetaKeys.CURRENT_CODE_HASH, None)
    bypass_actions = list(updated_meta.get(MetaKeys.LOOP_GUARD_BYPASS_ACTIONS, []))
    if "generate_code" not in bypass_actions:
        bypass_actions.append("generate_code")
    updated_meta[MetaKeys.LOOP_GUARD_BYPASS_ACTIONS] = bypass_actions

    updated_agents = dict(agents)
    updated_review = dict(review)
    updated_review["before_run_decision"] = None
    updated_review["approved_code_hash"] = None
    updated_agents["human_review"] = updated_review

    executor = dict((updated_agents.get("executor") or {}))
    if executor:
        executor["run_status"] = "idle"
        updated_agents["executor"] = executor

    return updated_output, updated_agents, updated_meta, True
