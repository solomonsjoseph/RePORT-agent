from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..conversation_events import append_conversation_event, build_clarification_event
from ..memory import (
    cache_reference_resolution,
    ensure_memory_state,
    resolve_reference_for_turn,
    validate_reference_resolution,
)
from ..state import AgentState, MetaKeys
from .generate_code import generate_code_node
from .qa import qa_node
from .rag_db_qa import rag_db_qa_node
from .orchestrator.state_logic import _user_message_hash
from .state_helpers import clear_clarification_meta
from .tool_routing import latest_user_message

NODE_NAME = "clarification"
NODE_CAPABILITY = (
    "Resume an active clarification loop by interpreting the user's follow-up and handing "
    "control back to the relevant subworkflow such as QA tool routing or code generation."
)


def _with_semantic_last_action(state: AgentState, action: str) -> AgentState:
    meta = dict(state.get("meta") or {})
    meta[MetaKeys.SEMANTIC_LAST_ACTION] = action
    return {
        **state,
        "meta": meta,
    }


def _resume_qa_tool_clarification(state: AgentState, llm) -> AgentState:
    question = latest_user_message(state)
    pending_question = (state.get("meta") or {}).get(MetaKeys.PENDING_QUESTION)
    if pending_question:
        effective_question = f"{pending_question}\n\nUser clarification: {question}"
    else:
        effective_question = question

    resumed_state = {
        **state,
        "meta": clear_clarification_meta(state.get("meta") or {}),
    }
    return _with_semantic_last_action(
        qa_node(resumed_state, llm, question_override=effective_question),
        "qa",
    )


def _memory_reference_clarification_question(candidates: list[dict[str, Any]]) -> str:
    lines = ["Which prior task did you mean?"]
    for candidate in candidates:
        label = str(candidate.get("label") or candidate.get("task_id") or "prior task")
        ordinal = candidate.get("display_ordinal")
        if isinstance(ordinal, int):
            lines.append(f"Task {ordinal}: {label}")
        else:
            lines.append(f"{candidate.get('task_id')}: {label}")
    return "\n".join(lines)


def _candidate_matches_reply(candidate: dict[str, Any], reply: str) -> bool:
    normalized = reply.strip()
    normalized_folded = normalized.casefold()
    if not normalized:
        return False

    task_id = candidate.get("task_id")
    if isinstance(task_id, str) and normalized == task_id:
        return True

    ordinal = candidate.get("display_ordinal")
    if isinstance(ordinal, int):
        ordinal_text = str(ordinal)
        if normalized == ordinal_text or normalized_folded == f"task {ordinal}":
            return True
    elif isinstance(ordinal, str) and normalized == ordinal.strip():
        return True

    label = candidate.get("label")
    return isinstance(label, str) and normalized_folded == label.strip().casefold()


def _pending_resolution_field(pending: dict[str, Any], field: str) -> Any:
    value = pending.get(field)
    if value is not None:
        return value
    raw_result = pending.get("result")
    if isinstance(raw_result, dict):
        return raw_result.get(field)
    return None


def _resolved_relationship(pending: dict[str, Any]) -> str | None:
    relationship = _pending_resolution_field(pending, "relationship")
    if isinstance(relationship, str) and relationship:
        return relationship
    return None


def _active_memory_reference_meta(meta: dict[str, Any] | None) -> dict[str, Any]:
    updated = dict(meta or {})
    updated[MetaKeys.AWAITING_USER_CLARIFICATION] = True
    updated[MetaKeys.CLARIFICATION_KIND] = "memory_reference_resolution"
    updated[MetaKeys.CLARIFICATION_RETURN_NODE] = "orchestrator"
    return updated


def _apply_resolved_task_meta(
    meta: dict[str, Any],
    validated: dict[str, Any],
    user_message_hash: str,
    reply_hash: str | None,
) -> dict[str, Any]:
    updated = clear_clarification_meta(meta)
    updated[MetaKeys.RESOLVED_TASK_ID] = validated.get("task_id")
    updated[MetaKeys.RESOLVED_TASK_KIND] = validated.get("task_kind")
    updated[MetaKeys.RESOLVED_TASK_RELATIONSHIP] = validated.get("relationship")
    updated[MetaKeys.RESOLVED_TASK_INTENDED_ACTION] = validated.get("intended_action")
    updated[MetaKeys.RESOLVED_TASK_USER_MESSAGE_HASH] = user_message_hash
    if reply_hash:
        updated[MetaKeys.LAST_USER_MESSAGE_HASH] = reply_hash
    return updated


def _cache_validated_resolution(
    state: AgentState,
    pending: dict[str, Any],
    raw_result: dict[str, Any],
) -> AgentState:
    original_hash = pending.get("user_message_hash")
    if not isinstance(original_hash, str) or not original_hash:
        original_hash = ""
    validated = validate_reference_resolution(state, raw_result)
    updated = cache_reference_resolution(state, original_hash, validated)
    updated["memory"]["pending_reference_clarification"] = None
    return {
        **updated,
        "meta": _apply_resolved_task_meta(
            updated.get("meta") or {},
            validated,
            original_hash,
            _user_message_hash(state),
        ),
        "next_action": None,
    }


def _cache_selected_resolution(
    state: AgentState,
    pending: dict[str, Any],
    matched: dict[str, Any],
    relationship: str,
) -> AgentState:
    raw_result: dict[str, Any] = {
        "label": "resolved",
        "task_id": matched.get("task_id"),
        "relationship": relationship,
        "intended_action": _pending_resolution_field(pending, "intended_action"),
        "confidence": "high",
        "needs_reference": False,
        "reason": "Resolved by explicit memory clarification reply.",
    }
    return _cache_validated_resolution(state, pending, raw_result)


def _reask_memory_reference_clarification(
    state: AgentState,
    pending: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> AgentState:
    original_hash = pending.get("user_message_hash")
    prompt = _memory_reference_clarification_question(candidates)
    updated: AgentState = {
        **state,
        "meta": _active_memory_reference_meta(state.get("meta") or {}),
        "output": {**dict(state.get("output") or {}), "qa_response": prompt},
    }
    return append_conversation_event(
        updated,
        build_clarification_event(
            actor="clarification",
            user_turn_hash=original_hash if isinstance(original_hash, str) else None,
            text=prompt,
            status="active",
        ),
    )


def _selected_only_state(state: AgentState, selected_task_id: str) -> AgentState:
    selected_state: AgentState = {**state, "memory": deepcopy(dict(state.get("memory") or {}))}
    memory = selected_state["memory"]
    completed = dict(memory.get("completed_tasks") or {})
    selected_card = completed.get(selected_task_id)
    if not isinstance(selected_card, dict):
        raise ValueError(f"Selected memory clarification task does not exist: {selected_task_id}")
    memory["completed_tasks"] = {selected_task_id: deepcopy(selected_card)}
    memory["task_order"] = [selected_task_id]
    memory["last_task_id"] = selected_task_id
    kind = selected_card.get("kind")
    memory["last_task_id_by_kind"] = {kind: selected_task_id} if isinstance(kind, str) else {}
    memory["last_reference_resolution"] = None
    memory["pending_reference_clarification"] = None
    return selected_state


def _resolve_selected_candidate(
    state: AgentState,
    llm,
    pending: dict[str, Any],
    matched: dict[str, Any],
) -> AgentState:
    selected_task_id = matched.get("task_id")
    if not isinstance(selected_task_id, str) or not selected_task_id:
        raise ValueError("Selected memory clarification candidate is missing task_id")

    original_hash = pending.get("user_message_hash")
    if not isinstance(original_hash, str) or not original_hash:
        original_hash = ""
    original_message = pending.get("original_user_message")
    if not isinstance(original_message, str) or not original_message.strip():
        original_message = latest_user_message(state)
    selected_hash = f"{original_hash}:selected:{selected_task_id}"
    selected_state = _selected_only_state(state, selected_task_id)
    try:
        selected_result = resolve_reference_for_turn(
            selected_state,
            llm,
            original_message,
            selected_hash,
        )
    except ValueError:
        return _reask_memory_reference_clarification(state, pending, [matched])
    if selected_result.get("label") != "resolved" or selected_result.get("task_id") != selected_task_id:
        return _reask_memory_reference_clarification(state, pending, [matched])
    return _cache_validated_resolution(state, pending, selected_result)


def _resume_memory_reference_clarification(state: AgentState, llm) -> AgentState:
    state, memory = ensure_memory_state({**state})
    pending = memory.get("pending_reference_clarification")
    if not isinstance(pending, dict):
        return {
            **state,
            "meta": clear_clarification_meta(state.get("meta") or {}),
            "next_action": None,
        }

    meta = dict(state.get("meta") or {})
    original_hash = pending.get("user_message_hash")
    if isinstance(original_hash, str) and original_hash == meta.get(MetaKeys.LAST_USER_MESSAGE_HASH):
        return {
            **state,
            "meta": _active_memory_reference_meta(meta),
            "next_action": "end",
        }

    candidates = [
        candidate
        for candidate in list(pending.get("candidates") or [])
        if isinstance(candidate, dict)
    ]
    reply = latest_user_message(state)

    matched = next(
        (candidate for candidate in candidates if _candidate_matches_reply(candidate, reply)),
        None,
    )

    if matched is None:
        return _reask_memory_reference_clarification(state, pending, candidates)

    relationship = _resolved_relationship(pending)
    if relationship is None:
        return _resolve_selected_candidate(state, llm, pending, matched)

    return _cache_selected_resolution(state, pending, matched, relationship)


def clarification_node(state: AgentState, llm, context: str = "") -> AgentState:
    meta = dict(state.get("meta") or {})
    kind = str(meta.get(MetaKeys.CLARIFICATION_KIND) or "")

    if kind == "qa_tool":
        return _resume_qa_tool_clarification(state, llm)
    if kind == "memory_reference_resolution":
        return _resume_memory_reference_clarification(state, llm)

    question = latest_user_message(state)
    pending_question = meta.get(MetaKeys.PENDING_QUESTION)
    if kind in {"rag_db_extraction_opt_in", "generate_code_dataset_selection"}:
        effective_question = question
    else:
        effective_question = (
            f"{pending_question}\n\nUser clarification: {question}"
            if pending_question and question
            else question
        )
    resumed_state = {
        **state,
        "meta": clear_clarification_meta(meta),
    }

    if meta.get(MetaKeys.CLARIFICATION_RETURN_NODE) == "generate_code":
        if isinstance(context, dict):
            return _with_semantic_last_action(
                generate_code_node(
                    resumed_state,
                    llm,
                    context,
                    question_override=effective_question,
                ),
                "generate_code",
            )
        return _with_semantic_last_action(
            generate_code_node(
                resumed_state,
                llm,
                context,
                question_override=effective_question,
            ),
            "generate_code",
        )
    if meta.get(MetaKeys.CLARIFICATION_RETURN_NODE) == "rag_db_qa":
        if isinstance(context, dict):
            return _with_semantic_last_action(
                rag_db_qa_node(
                    resumed_state,
                    llm,
                    provider=str(context.get("provider") or ""),
                    service=context.get("db_rag_service"),
                    reranker_model=context.get("db_rag_reranker_model"),
                    question_override=effective_question,
                ),
                "rag_db_qa",
            )
        raise ValueError("rag_db_qa clarification resume requires a context mapping with provider and db_rag_service")

    return _with_semantic_last_action(
        qa_node(resumed_state, llm, context, question_override=effective_question),
        "qa",
    )
