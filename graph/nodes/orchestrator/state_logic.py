from __future__ import annotations

import hashlib

from ...state import AgentState, MetaKeys


def _latest_user_message_obj(state: AgentState):
    messages = list(state.get("messages", []))
    for message in reversed(messages):
        if getattr(message, "type", None) == "human":
            return message
    return None


def _latest_user_message(state: AgentState) -> str:
    message = _latest_user_message_obj(state)
    if message is None:
        return ""
    return str(getattr(message, "content", "") or "").strip()


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


def _user_message_hash(state: AgentState) -> str | None:
    message = _latest_user_message_obj(state)
    if message is None:
        return None
    content = str(getattr(message, "content", "") or "").strip()
    msg_id = str(getattr(message, "id", "") or "").strip()
    # Use message id when available so repeated identical text still counts as
    # a fresh turn and stale intent/state does not bleed into new requests.
    hash_basis = f"{msg_id}:{content}" if msg_id else content
    if not hash_basis:
        return None
    return hashlib.sha256(hash_basis.encode()).hexdigest()[:16]




def _consume_regenerate_before_run(output: dict, agents: dict, meta: dict) -> tuple[dict, dict, dict, bool]:
    review = (agents.get("human_review") or {}) if isinstance(agents, dict) else {}
    if review.get("before_run_decision") != "regenerate":
        return output, agents, meta, False

    updated_output = dict(output)
    updated_output.pop("generated_code", None)

    updated_meta = dict(meta)
    updated_meta.pop(MetaKeys.CURRENT_CODE_HASH, None)
    updated_meta.pop(MetaKeys.EXECUTION_TICKET_HASH, None)
    updated_meta.pop(MetaKeys.ERROR_RECOVERY_ACTIVE, None)
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


def _consume_final_review_regenerate(
    output: dict,
    agents: dict,
    meta: dict,
) -> tuple[dict, dict, dict, bool]:
    review = (agents.get("human_review") or {}) if isinstance(agents, dict) else {}
    if review.get("final_decision") != "regenerate":
        return output, agents, meta, False

    updated_output = dict(output)
    updated_output.pop("generated_code", None)
    updated_output.pop("text", None)
    updated_output.pop("figure_png", None)
    updated_output.pop("error", None)

    updated_meta = dict(meta)
    updated_meta.pop(MetaKeys.CURRENT_CODE_HASH, None)
    updated_meta.pop(MetaKeys.EXECUTION_TICKET_HASH, None)
    updated_meta.pop(MetaKeys.ERROR_RECOVERY_ACTIVE, None)
    bypass_actions = list(updated_meta.get(MetaKeys.LOOP_GUARD_BYPASS_ACTIONS, []))
    if "generate_code" not in bypass_actions:
        bypass_actions.append("generate_code")
    updated_meta[MetaKeys.LOOP_GUARD_BYPASS_ACTIONS] = bypass_actions

    updated_agents = dict(agents)
    updated_review = dict(review)
    updated_review["final_decision"] = None
    updated_review["before_run_decision"] = None
    updated_review["approved_code_hash"] = None
    updated_agents["human_review"] = updated_review

    executor = dict((updated_agents.get("executor") or {}))
    if executor:
        executor["run_status"] = "idle"
        updated_agents["executor"] = executor

    return updated_output, updated_agents, updated_meta, True


def _consume_before_run_approval(
    output: dict,
    agents: dict,
    meta: dict,
) -> tuple[dict, dict, dict, bool]:
    review = (agents.get("human_review") or {}) if isinstance(agents, dict) else {}
    if review.get("before_run_decision") != "approve":
        return output, agents, meta, False

    updated_agents = dict(agents)
    updated_review = dict(review)
    updated_meta = dict(meta)
    current_hash = updated_meta.get(MetaKeys.CURRENT_CODE_HASH)

    updated_review["before_run_decision"] = None
    updated_review["approved_code_hash"] = current_hash if current_hash else None
    updated_agents["human_review"] = updated_review

    if current_hash:
        updated_meta[MetaKeys.EXECUTION_TICKET_HASH] = current_hash
    else:
        updated_meta.pop(MetaKeys.EXECUTION_TICKET_HASH, None)
    updated_meta.pop(MetaKeys.ERROR_RECOVERY_ACTIVE, None)

    return output, updated_agents, updated_meta, True


def _consume_after_error_decision(
    output: dict,
    agents: dict,
    meta: dict,
) -> tuple[dict, dict, dict, str | None]:
    review = (agents.get("human_review") or {}) if isinstance(agents, dict) else {}
    decision = review.get("after_error_decision")
    if not decision:
        return output, agents, meta, None

    updated_agents = dict(agents)
    updated_review = dict(review)
    updated_meta = dict(meta)
    updated_review["after_error_decision"] = None
    updated_review["before_run_decision"] = None
    updated_review["approved_code_hash"] = None
    updated_agents["human_review"] = updated_review
    updated_meta.pop(MetaKeys.EXECUTION_TICKET_HASH, None)
    updated_meta.pop(MetaKeys.ERROR_RECOVERY_ACTIVE, None)

    bypass_actions = list(updated_meta.get(MetaKeys.LOOP_GUARD_BYPASS_ACTIONS, []))
    if "generate_code" not in bypass_actions:
        bypass_actions.append("generate_code")
    updated_meta[MetaKeys.LOOP_GUARD_BYPASS_ACTIONS] = bypass_actions

    return output, updated_agents, updated_meta, "generate_code"


def _consume_final_review_approval(
    output: dict,
    agents: dict,
    meta: dict,
) -> tuple[dict, dict, dict, bool]:
    review = (agents.get("human_review") or {}) if isinstance(agents, dict) else {}
    if review.get("final_decision") != "approve":
        return output, agents, meta, False

    updated_agents = dict(agents)
    updated_review = dict(review)
    updated_meta = dict(meta)
    updated_review["final_decision"] = None
    updated_review["before_run_decision"] = None
    updated_review["approved_code_hash"] = None
    updated_agents["human_review"] = updated_review
    updated_meta.pop(MetaKeys.EXECUTION_TICKET_HASH, None)
    updated_meta.pop(MetaKeys.ERROR_RECOVERY_ACTIVE, None)

    return output, updated_agents, updated_meta, True
