from __future__ import annotations

from typing import Iterable

from ...conversation_events import (
    append_conversation_event,
    build_clarification_event,
    build_routing_decision_event,
    build_user_event,
    has_conversation_event,
)
from ...memory import (
    classify_user_intent_reference,
    ensure_memory_state,
    resolve_reference_for_turn,
)
from ...memory.schema import ALLOWED_RELATIONSHIPS
from ...memory.validation import ALLOWED_RELATIONSHIPS_BY_TASK_KIND
from ...state import AgentState, MetaKeys
from ...state_views import get_planner_state, merge_state_patch
from ..node_registry import NODE_REGISTRY_MAP
from ..tool_routing import is_tool_requested, latest_user_message
from .action_mask import mask_actions
from .planner import llm_plan_next_action
from .policy import (
    DETERMINISTIC_CONTROL_ACTIONS,
    _next_tool_requester,
    _tool_request_queue,
    select_planner_fallback_action,
    should_prefer_rag_db_qa,
)
from .progress_controller import apply_recurrence_guard, update_recurrence_state
from .state_logic import (
    _consume_after_error_decision,
    _consume_before_run_approval,
    _consume_before_run_cancel,
    _consume_final_review_cancel,
    _consume_final_review_regenerate,
    _consume_final_review_approval,
    _consume_regenerate_before_run,
    _has_unanswered_human_message,
    _user_message_hash,
    derive_planner_memory,
)
from .workflow_status import derive_workflow_status


__all__ = ["orchestrator_node", "resolve_reference_for_turn"]


_RESOLVED_TASK_META_KEYS = (
    MetaKeys.RESOLVED_TASK_ID,
    MetaKeys.RESOLVED_TASK_KIND,
    MetaKeys.RESOLVED_TASK_RELATIONSHIP,
    MetaKeys.RESOLVED_TASK_INTENDED_ACTION,
    MetaKeys.RESOLVED_TASK_USER_MESSAGE_HASH,
)
_RESOLVED_TASK_META_CONSUMED_KEY = "resolved_task_meta_consumed"
_RESOLVED_USER_INTENT_META_KEYS = (
    MetaKeys.RESOLVED_USER_INTENT_ID,
    MetaKeys.RESOLVED_USER_INTENT_KIND,
    MetaKeys.RESOLVED_USER_INTENT_RELATIONSHIP,
    MetaKeys.RESOLVED_USER_INTENT_SOURCE_QUESTION,
    MetaKeys.RESOLVED_USER_INTENT_USER_MESSAGE_HASH,
    MetaKeys.RAG_DB_QUESTION_OVERRIDE,
)
_RESOLVED_USER_INTENT_META_CONSUMED_KEY = "resolved_user_intent_meta_consumed"


def _clear_resolved_task_meta(meta: dict) -> dict:
    updated = dict(meta)
    for key in _RESOLVED_TASK_META_KEYS:
        updated.pop(key, None)
    updated.pop(_RESOLVED_TASK_META_CONSUMED_KEY, None)
    return updated


def _clear_resolved_user_intent_meta(meta: dict) -> dict:
    updated = dict(meta)
    for key in _RESOLVED_USER_INTENT_META_KEYS:
        updated.pop(key, None)
    updated.pop(_RESOLVED_USER_INTENT_META_CONSUMED_KEY, None)
    return updated


def _clear_consumed_resolved_task_meta(meta: dict) -> dict:
    resolved_hash = meta.get(MetaKeys.RESOLVED_TASK_USER_MESSAGE_HASH)
    consumed_hash = meta.get(_RESOLVED_TASK_META_CONSUMED_KEY)
    if isinstance(resolved_hash, str) and consumed_hash == resolved_hash:
        return _clear_resolved_task_meta(meta)
    return meta


def _clear_consumed_resolved_user_intent_meta(meta: dict) -> dict:
    resolved_hash = meta.get(MetaKeys.RESOLVED_USER_INTENT_USER_MESSAGE_HASH)
    consumed_hash = meta.get(_RESOLVED_USER_INTENT_META_CONSUMED_KEY)
    if isinstance(resolved_hash, str) and consumed_hash == resolved_hash:
        return _clear_resolved_user_intent_meta(meta)
    return meta


def _apply_resolved_task_meta(
    meta: dict,
    resolution: dict,
    task: dict,
    current_hash: str,
) -> dict:
    updated = dict(meta)
    updated[MetaKeys.RESOLVED_TASK_ID] = task.get("task_id")
    updated[MetaKeys.RESOLVED_TASK_KIND] = task.get("kind")
    updated[MetaKeys.RESOLVED_TASK_RELATIONSHIP] = resolution.get("relationship")
    updated[MetaKeys.RESOLVED_TASK_INTENDED_ACTION] = resolution.get("intended_action")
    updated[MetaKeys.RESOLVED_TASK_USER_MESSAGE_HASH] = current_hash
    updated[_RESOLVED_TASK_META_CONSUMED_KEY] = current_hash
    return updated


def _apply_resolved_user_intent_meta(
    meta: dict,
    resolution: dict,
    intent: dict,
    current_hash: str,
) -> dict:
    updated = dict(meta)
    updated[MetaKeys.RESOLVED_USER_INTENT_ID] = intent.get("intent_id")
    updated[MetaKeys.RESOLVED_USER_INTENT_KIND] = intent.get("kind")
    updated[MetaKeys.RESOLVED_USER_INTENT_RELATIONSHIP] = resolution.get("relationship")
    updated[MetaKeys.RESOLVED_USER_INTENT_SOURCE_QUESTION] = intent.get("source_question")
    updated[MetaKeys.RESOLVED_USER_INTENT_USER_MESSAGE_HASH] = current_hash
    updated.pop(_RESOLVED_USER_INTENT_META_CONSUMED_KEY, None)
    updated.pop(MetaKeys.RAG_DB_QUESTION_OVERRIDE, None)
    return updated


def _task_candidates_for_clarification(
    memory: dict,
    resolution: dict,
) -> list[dict]:
    raw_candidates = resolution.get("candidate_cards")
    if not isinstance(raw_candidates, list):
        completed = dict(memory.get("completed_tasks") or {})
        raw_candidates = [
            completed[task_id]
            for task_id in list(memory.get("task_order") or [])
            if task_id in completed
        ]

    candidates: list[dict] = []
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            continue
        task_id = raw.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            continue
        label = str(raw.get("label") or task_id).strip()
        summary = str(raw.get("summary") or "").strip()
        candidates.append(
            {
                "task_id": task_id,
                "display_ordinal": raw.get("display_ordinal"),
                "kind": raw.get("kind"),
                "label": label,
                "hint": summary,
            }
        )
    return candidates


def _memory_reference_clarification_question(candidates: list[dict]) -> str:
    lines = ["Which prior task did you mean?"]
    for candidate in candidates:
        ordinal = candidate.get("display_ordinal")
        label = str(candidate.get("label") or candidate.get("task_id") or "prior task")
        if isinstance(ordinal, int):
            lines.append(f"Task {ordinal}: {label}")
        else:
            lines.append(f"{candidate.get('task_id')}: {label}")
    return "\n".join(lines)


def _route_from_resolved_task_meta(
    routing_state: AgentState,
    meta: dict,
    available_action_set: set[str],
) -> tuple[str | None, dict, list[str]]:
    task_id = meta.get(MetaKeys.RESOLVED_TASK_ID)
    relationship = meta.get(MetaKeys.RESOLVED_TASK_RELATIONSHIP)
    resolved_hash = meta.get(MetaKeys.RESOLVED_TASK_USER_MESSAGE_HASH)
    consumed_hash = meta.get(_RESOLVED_TASK_META_CONSUMED_KEY)
    if isinstance(resolved_hash, str) and consumed_hash == resolved_hash:
        return None, _clear_resolved_task_meta(meta), []
    if not isinstance(task_id, str) or not isinstance(relationship, str):
        return None, meta, []

    _state, memory = ensure_memory_state(routing_state)
    task = dict(memory.get("completed_tasks") or {}).get(task_id)
    if not isinstance(task, dict):
        return None, _clear_resolved_task_meta(meta), []

    routed_node: str | None = None
    updated_meta = dict(meta)
    if task.get("kind") == "db_rag_sql_extraction":
        dataset_id = dict(task.get("artifact_refs") or {}).get("dataset_artifact_id")
        if relationship in {"use_as_input", "compare"} and "generate_code" in available_action_set:
            if isinstance(dataset_id, str) and dataset_id:
                updated_meta[MetaKeys.ANALYSIS_DATASET_ID] = dataset_id
                routed_node = "generate_code"
        elif relationship in {"revision", "rerun", "explain", "inspect_artifact"} and "rag_db_qa" in available_action_set:
            routed_node = "rag_db_qa"

    if not routed_node:
        return None, _clear_resolved_task_meta(updated_meta), []

    if isinstance(resolved_hash, str):
        updated_meta[_RESOLVED_TASK_META_CONSUMED_KEY] = resolved_hash
    return (
        routed_node,
        updated_meta,
        [
            f"task_id={task.get('task_id')} "
            f"relationship={relationship} "
            f"routed_node={routed_node}"
        ],
    )


def _route_from_resolved_user_intent_meta(
    routing_state: AgentState,
    meta: dict,
    available_action_set: set[str],
) -> tuple[str | None, dict, list[str]]:
    intent_id = meta.get(MetaKeys.RESOLVED_USER_INTENT_ID)
    kind = meta.get(MetaKeys.RESOLVED_USER_INTENT_KIND)
    relationship = meta.get(MetaKeys.RESOLVED_USER_INTENT_RELATIONSHIP)
    resolved_hash = meta.get(MetaKeys.RESOLVED_USER_INTENT_USER_MESSAGE_HASH)
    consumed_hash = meta.get(_RESOLVED_USER_INTENT_META_CONSUMED_KEY)
    if isinstance(resolved_hash, str) and consumed_hash == resolved_hash:
        return None, _clear_resolved_user_intent_meta(meta), []
    if (
        "rag_db_qa" not in available_action_set
        or not isinstance(intent_id, str)
        or kind != "db_rag_query"
        or relationship not in {"continue", "refine"}
    ):
        return None, _clear_resolved_user_intent_meta(meta), []

    _state, memory = ensure_memory_state(routing_state)
    intent = dict(memory.get("user_intents") or {}).get(intent_id)
    if not isinstance(intent, dict) or intent.get("kind") != "db_rag_query":
        return None, _clear_resolved_user_intent_meta(meta), []
    source_question = str(intent.get("source_question") or "").strip()
    if not source_question:
        return None, _clear_resolved_user_intent_meta(meta), []

    updated_meta = dict(meta)
    updated_meta[MetaKeys.RAG_DB_QUESTION_OVERRIDE] = source_question
    if isinstance(resolved_hash, str):
        updated_meta[_RESOLVED_USER_INTENT_META_CONSUMED_KEY] = resolved_hash
    return (
        "rag_db_qa",
        updated_meta,
        [
            f"user_intent_id={intent_id} "
            f"relationship={relationship} "
            "routed_node=rag_db_qa"
        ],
    )


def _dataset_exists(state: AgentState, dataset_id: object) -> bool:
    if not isinstance(dataset_id, str) or not dataset_id:
        return False
    datasets = (state.get("artifacts") or {}).get("datasets") or {}
    return isinstance(datasets, dict) and dataset_id in datasets


def _planner_binding_clarification(
    state: AgentState,
    meta: dict,
    output: dict,
    available_action_set: set[str],
    question: str,
    relationship: object = None,
) -> tuple[str, dict, dict, list[str]]:
    updated_meta = dict(meta)
    updated_meta[MetaKeys.AWAITING_USER_CLARIFICATION] = True
    _state, memory = ensure_memory_state(state)
    completed_tasks = dict(memory.get("completed_tasks") or {})
    if completed_tasks:
        candidates = _task_candidates_for_clarification(
            memory,
            {"candidate_cards": list(completed_tasks.values())},
        )
        question = _memory_reference_clarification_question(candidates)
        updated_meta[MetaKeys.CLARIFICATION_KIND] = "memory_reference_resolution"
        updated_meta[MetaKeys.CLARIFICATION_RETURN_NODE] = "orchestrator"
        memory["pending_reference_clarification"] = {
            "status": "awaiting_reply",
            "user_message_hash": _user_message_hash(state) or "",
            "original_user_message": latest_user_message(state),
            "relationship": relationship if isinstance(relationship, str) else None,
            "intended_action": None,
            "result": {
                "label": "ambiguous",
                "task_id": None,
                "relationship": relationship if isinstance(relationship, str) else None,
                "needs_reference": True,
                "reason": question,
            },
            "candidates": candidates,
        }
    else:
        updated_meta[MetaKeys.CLARIFICATION_KIND] = "planner_route_clarification"
        updated_meta[MetaKeys.CLARIFICATION_RETURN_NODE] = "qa"
    updated_meta[MetaKeys.PENDING_QUESTION] = question
    updated_output = dict(output)
    updated_output["qa_response"] = question
    routed_node = (
        "clarification"
        if "clarification" in available_action_set
        else ("qa" if "qa" in available_action_set else "end")
    )
    return routed_node, updated_meta, updated_output, [
        f"task_id=None relationship=None routed_node={routed_node}",
        f"planner_clarification={question}",
    ]


def _apply_planner_bindings(
    routing_state: AgentState,
    *,
    action: str,
    bindings: dict[str, object],
    meta: dict,
    output: dict,
    available_action_set: set[str],
    current_hash: str | None,
    allow_bindings: bool,
) -> tuple[str, dict, dict, list[str]]:
    if not allow_bindings:
        return action, _clear_resolved_task_meta(meta), output, []

    if bindings.get("needs_clarification") is True:
        question = str(
            bindings.get("clarification_question")
            or "Which prior task or dataset should I use for this request?"
        ).strip()
        return _planner_binding_clarification(
            routing_state,
            meta,
            output,
            available_action_set,
            question,
            bindings.get("relationship"),
        )

    updated_meta = dict(meta)
    observations: list[str] = []
    task: dict | None = None
    task_id = bindings.get("referenced_task_id")
    relationship = bindings.get("relationship")
    dataset_id = bindings.get("dataset_id")

    _state, memory = ensure_memory_state(routing_state)
    completed_tasks = dict(memory.get("completed_tasks") or {})
    if task_id is not None:
        if not isinstance(task_id, str) or task_id not in completed_tasks:
            return _planner_binding_clarification(
                routing_state,
                meta,
                output,
                available_action_set,
                "Which prior task should I use for this request?",
            )
        task = completed_tasks[task_id]

    if relationship is not None:
        if not isinstance(relationship, str) or relationship not in ALLOWED_RELATIONSHIPS:
            return _planner_binding_clarification(
                routing_state,
                meta,
                output,
                available_action_set,
                "What relationship should this request have to the prior task?",
            )
        if task is not None:
            allowed = ALLOWED_RELATIONSHIPS_BY_TASK_KIND.get(task.get("kind"), set())
            if relationship not in allowed:
                return _planner_binding_clarification(
                    routing_state,
                    meta,
                    output,
                    available_action_set,
                    "What should I do with the referenced prior task?",
                )

    if action == "generate_code":
        if not isinstance(dataset_id, str) and task is not None:
            task_dataset_id = dict(task.get("artifact_refs") or {}).get("dataset_artifact_id")
            if isinstance(task_dataset_id, str):
                dataset_id = task_dataset_id
        if isinstance(dataset_id, str):
            if not _dataset_exists(routing_state, dataset_id):
                return _planner_binding_clarification(
                    routing_state,
                    meta,
                    output,
                    available_action_set,
                    "Which registered dataset should I use for this analysis?",
                )
            updated_meta[MetaKeys.ANALYSIS_DATASET_ID] = dataset_id

    if task is not None and isinstance(relationship, str):
        updated_meta[MetaKeys.RESOLVED_TASK_ID] = task.get("task_id")
        updated_meta[MetaKeys.RESOLVED_TASK_KIND] = task.get("kind")
        updated_meta[MetaKeys.RESOLVED_TASK_RELATIONSHIP] = relationship
        updated_meta[MetaKeys.RESOLVED_TASK_INTENDED_ACTION] = bindings.get("route_reason")
        if current_hash:
            updated_meta[MetaKeys.RESOLVED_TASK_USER_MESSAGE_HASH] = current_hash
            updated_meta[_RESOLVED_TASK_META_CONSUMED_KEY] = current_hash
        observations.append(
            f"task_id={task.get('task_id')} relationship={relationship} routed_node={action}"
        )

    return action, updated_meta, output, observations


def _record_planner_decision(
    state: AgentState,
    planner_action: str,
    thought: str,
    routed_action: str,
    diagnostics: dict[str, str] | None = None,
) -> AgentState:
    planner = get_planner_state(state)
    trace = list(planner.get("decision_trace") or [])
    decision = {
        "action": planner_action,
        "thought": thought,
        "after": state.get("last_action"),
        "routed_action": routed_action,
    }
    if diagnostics:
        decision.update(diagnostics)
    trace.append(decision)
    planner["last_decision"] = decision
    planner["decision_trace"] = trace[-20:]
    return merge_state_patch(state, {"planner": planner})


def _latest_user_message_is_explicit_code_request(state: AgentState) -> bool:
    messages = list(state.get("messages", []))
    latest = ""
    for message in reversed(messages):
        if getattr(message, "type", None) == "human":
            latest = str(getattr(message, "content", "") or "").strip().lower()
            break

    if not latest:
        return False

    normalized = " ".join(latest.split())
    direct_markers = (
        "write code",
        "generate code",
        "show code",
        "sample code",
        "example code",
        "python code",
        "ready-to-run code",
        "ready to run code",
        "give me code",
        "give me python",
    )
    if any(marker in normalized for marker in direct_markers):
        return True

    return "python" in normalized and any(
        token in normalized for token in ("write", "generate", "show", "ready-to-run", "ready to run")
    )

def _ready_deterministic_action(
    state: AgentState,
    available_actions: Iterable[str],
    *,
    fresh_unanswered_user_turn: bool = False,
) -> str | None:
    """Return a ready control action that should not depend on planner output."""
    available = set(available_actions)
    for action in DETERMINISTIC_CONTROL_ACTIONS:
        if action not in available:
            continue
        if action == "human_review_before_run" and fresh_unanswered_user_turn:
            continue
        node = NODE_REGISTRY_MAP.get(action)
        if node and node.is_ready(state):
            return action
    return None


def _ready_rag_db_sql_generation_resume(
    state: AgentState,
    available_action_set: set[str],
) -> str | None:
    if "rag_db_qa" not in available_action_set:
        return None
    rag_state = dict((state.get("agents") or {}).get("rag_db_qa") or {})
    approved_selection_id = str(
        rag_state.get("approved_column_selection_artifact_id") or ""
    ).strip()
    if (
        approved_selection_id
        and rag_state.get("thread_status") == "awaiting_sql_generation"
    ):
        return "rag_db_qa"
    return None


def _resumed_from(state: AgentState, node_name: str) -> bool:
    if state.get("last_action") == node_name:
        return True
    workflow_trace = list((state.get("meta") or {}).get(MetaKeys.WORKFLOW_TRACE, []))
    return bool(workflow_trace and workflow_trace[-1] == node_name)


def _is_soft_qa_followup(meta: dict) -> bool:
    return (
        meta.get(MetaKeys.CLARIFICATION_KIND) == "qa_followup"
        and meta.get(MetaKeys.CLARIFICATION_RETURN_NODE) == "qa"
    )


def _store_workflow_status(state: AgentState) -> AgentState:
    meta = dict(state.get("meta") or {})
    status = derive_workflow_status(state)
    meta[MetaKeys.WORKFLOW_MILESTONE] = status["milestone"]
    meta[MetaKeys.COMPLETION_STATUS] = status["completion_status"]
    meta[MetaKeys.BLOCKER_SIGNATURE] = status["blocker_signature"]
    return {**state, "meta": meta}


def _refresh_planner_memory(state: AgentState) -> AgentState:
    planner = get_planner_state(state)
    planner["memory"] = derive_planner_memory(state)
    return merge_state_patch(state, {"planner": planner})


def _should_end_for_completion(state: AgentState) -> bool:
    if _has_unanswered_human_message(state):
        return False
    meta = dict(state.get("meta") or {})
    completion = meta.get(MetaKeys.COMPLETION_STATUS)
    if completion == "complete":
        return True
    if completion != "blocked_waiting":
        return False
    if meta.get(MetaKeys.AWAITING_USER_CLARIFICATION) and not _has_unanswered_human_message(state):
        return True
    return state.get("last_action") in {
        "human_review_before_run",
        "human_review_after_error",
        "human_review_before_output",
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
    defer_qa_followup_clarification_to_planner = False
    memory_clarification_selected = False
    memory_clarification_question: str | None = None

    output = dict(state.get("output") or {})
    agents = dict(state.get("agents") or {})
    if _resumed_from(state, "human_review_before_run"):
        output, agents, meta, before_run_cancelled = _consume_before_run_cancel(output, agents, meta)
        if before_run_cancelled:
            next_action = "end"
            transition_selected_from_resume = True
            orchestrator_state.pop("next_action", None)
            observations = list(state.get("observations", []))
            observations.append("orchestrator: consumed before-run cancel; ending workflow")
            state = {**state, "observations": observations}
        else:
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
            if after_error_action == "end":
                observations.append("orchestrator: consumed after-error cancel; ending workflow")
            else:
                observations.append(
                    "orchestrator: consumed after-error review decision; routing to generate_code"
                )
            state = {**state, "observations": observations}

    if _resumed_from(state, "human_review_before_output"):
        output, agents, meta, final_cancelled = _consume_final_review_cancel(output, agents, meta)
        if final_cancelled:
            next_action = "end"
            transition_selected_from_resume = True
            orchestrator_state.pop("next_action", None)
            observations = list(state.get("observations", []))
            observations.append("orchestrator: consumed final-review cancel; ending workflow")
            state = {**state, "observations": observations}
        else:
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

    if _resumed_from(state, "human_review_rag_db_column_selection"):
        rag_state = dict(agents.get("rag_db_qa") or {})
        if rag_state.get("thread_status") == "cancelled":
            rag_state["thread_status"] = "done"
            rag_state["active_thread"] = False
            agents = {
                **agents,
                "rag_db_qa": rag_state,
            }
            next_action = "end"
            transition_selected_from_resume = True
            orchestrator_state.pop("next_action", None)
            observations = list(state.get("observations", []))
            observations.append("orchestrator: consumed DB-RAG column-review cancel; ending workflow")
            state = {**state, "observations": observations}

    if _resumed_from(state, "human_review_rag_db_sql_execution"):
        rag_state = dict(agents.get("rag_db_qa") or {})
        if rag_state.get("thread_status") == "cancelled":
            rag_state["thread_status"] = "done"
            rag_state["active_thread"] = False
            agents = {
                **agents,
                "rag_db_qa": rag_state,
            }
            next_action = "end"
            transition_selected_from_resume = True
            orchestrator_state.pop("next_action", None)
            observations = list(state.get("observations", []))
            observations.append("orchestrator: consumed DB-RAG SQL-review cancel; ending workflow")
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

    fresh_unanswered_user_turn = bool(
        current_hash
        and _has_unanswered_human_message(state)
        and current_hash != meta.get(MetaKeys.LAST_USER_MESSAGE_HASH)
        and not has_pending_regenerate
    )

    if fresh_unanswered_user_turn:
        if was_awaiting_clarification:
            clarification_return_node = meta.get(MetaKeys.CLARIFICATION_RETURN_NODE)
            if _is_soft_qa_followup(meta):
                next_action = None
                defer_qa_followup_clarification_to_planner = True
                orchestrator_state.pop("next_action", None)
            elif (
                clarification_return_node != "rag_db_qa"
                and "rag_db_qa" in available_action_set
                and should_prefer_rag_db_qa({**state, "output": output, "agents": agents, "meta": meta})
            ):
                meta[MetaKeys.LAST_USER_MESSAGE_HASH] = current_hash
                next_action = "rag_db_qa"
            elif "clarification" in available_action_set:
                meta[MetaKeys.LAST_USER_MESSAGE_HASH] = current_hash
                next_action = "clarification"
            else:
                meta[MetaKeys.LAST_USER_MESSAGE_HASH] = current_hash
                clarification_return = (
                    clarification_return_node
                    or state.get("last_action")
                    or "qa"
                )
                if clarification_return in available_action_set:
                    next_action = clarification_return
                else:
                    next_action = "qa"
            orchestrator_state["next_action"] = next_action
        else:
            meta = _clear_resolved_task_meta(meta)
            meta = _clear_resolved_user_intent_meta(meta)
            meta[MetaKeys.LAST_USER_MESSAGE_HASH] = current_hash
            next_action = None
            orchestrator_state.pop("next_action", None)

    routing_state = {
        **state,
        "output": output,
        "agents": agents,
        "meta": meta,
    }

    if not fresh_unanswered_user_turn:
        meta = _clear_consumed_resolved_task_meta(meta)
        meta = _clear_consumed_resolved_user_intent_meta(meta)
        routing_state["meta"] = meta

    if not next_action:
        if "tool_handler" in available_action_set and is_tool_requested(routing_state):
            next_action = "tool_handler"
        elif state.get("last_action") == "tool_handler":
            requester = _next_tool_requester(routing_state)
            if requester and requester in available_action_set:
                next_action = requester
                queue = _tool_request_queue(routing_state)
                meta[MetaKeys.TOOL_REQUEST_QUEUE] = [n for n in queue if n != requester]
                routing_state["meta"] = meta

    routing_state = _refresh_planner_memory(routing_state)
    routing_state = _store_workflow_status(routing_state)
    routing_state = update_recurrence_state(routing_state)
    meta = dict(routing_state.get("meta") or {})

    if not next_action and _should_end_for_completion(routing_state):
        next_action = "end"

    if not next_action and not fresh_unanswered_user_turn:
        routed_from_meta, meta, resolved_observations = _route_from_resolved_task_meta(
            routing_state,
            meta,
            available_action_set,
        )
        if routed_from_meta:
            next_action = routed_from_meta
            observations = list(routing_state.get("observations", []))
            observations.extend(resolved_observations)
            routing_state = {
                **routing_state,
                "meta": meta,
                "observations": observations,
            }
        else:
            routing_state = {**routing_state, "meta": meta}

    if (
        not next_action
        and fresh_unanswered_user_turn
        and current_hash
        and dict((routing_state.get("memory") or {}).get("user_intents") or {})
    ):
        try:
            classification = classify_user_intent_reference(
                routing_state,
                llm,
                user_message=latest_user_message(routing_state),
                user_message_hash=current_hash,
            )
        except ValueError as exc:
            classification = None
            observations = list(routing_state.get("observations", []))
            observations.append(f"user_intent_classifier_error={exc}")
            routing_state = {**routing_state, "observations": observations}
        if (
            isinstance(classification, dict)
            and
            classification.get("target") == "existing_user_intent"
            and classification.get("needs_clarification") is not True
        ):
            _state, memory = ensure_memory_state(routing_state)
            intent_id = classification.get("target_id")
            intent = dict(memory.get("user_intents") or {}).get(intent_id)
            if isinstance(intent, dict):
                meta = _apply_resolved_user_intent_meta(
                    meta,
                    classification,
                    intent,
                    current_hash,
                )
                routing_state = {**routing_state, "meta": meta}
        elif isinstance(classification, dict) and (
            classification.get("target") == "ambiguous"
            or classification.get("needs_clarification") is True
        ):
            question = "Which previous database query did you want to continue?"
            meta = dict(meta)
            meta[MetaKeys.AWAITING_USER_CLARIFICATION] = True
            meta[MetaKeys.CLARIFICATION_KIND] = "qa_followup"
            meta[MetaKeys.CLARIFICATION_RETURN_NODE] = "qa"
            meta[MetaKeys.PENDING_QUESTION] = question
            output = dict(output)
            output["qa_response"] = question
            next_action = "end"
            memory_clarification_question = question
            observations = list(routing_state.get("observations", []))
            observations.append(f"user_intent_clarification={question}")
            routing_state = {
                **routing_state,
                "meta": meta,
                "output": output,
                "observations": observations,
            }

    if not next_action:
        routed_from_user_intent, meta, user_intent_observations = (
            _route_from_resolved_user_intent_meta(
                routing_state,
                meta,
                available_action_set,
            )
        )
        if routed_from_user_intent:
            next_action = routed_from_user_intent
            observations = list(routing_state.get("observations", []))
            observations.extend(user_intent_observations)
            routing_state = {
                **routing_state,
                "meta": meta,
                "observations": observations,
            }
        else:
            routing_state = {**routing_state, "meta": meta}

    if not next_action and not defer_qa_followup_clarification_to_planner:
        next_action = _ready_deterministic_action(
            routing_state,
            available_action_list,
            fresh_unanswered_user_turn=fresh_unanswered_user_turn,
        )

    if not next_action:
        next_action = _ready_rag_db_sql_generation_resume(
            routing_state,
            available_action_set,
        )

    if not next_action:
        llm_choice, thought, planner_action, diagnostics, bindings = llm_plan_next_action(
            routing_state, llm, available_action_list
        )
        masked_actions, _blocked = mask_actions(routing_state, available_action_list)
        if llm_choice != "end" and llm_choice in masked_actions:
            next_action, meta, output, planner_observations = _apply_planner_bindings(
                routing_state,
                action=llm_choice,
                bindings=bindings,
                meta=meta,
                output=output,
                available_action_set=available_action_set,
                current_hash=current_hash,
                allow_bindings=fresh_unanswered_user_turn,
            )
            observations = list(routing_state.get("observations", []))
            observations.extend(planner_observations)
            routing_state = {
                **routing_state,
                "meta": meta,
                "output": output,
                "observations": observations,
            }
            memory_clarification_selected = next_action == "clarification"
            memory_clarification_question = (
                output.get("qa_response") if memory_clarification_selected else None
            )
        else:
            if (
                _latest_user_message_is_explicit_code_request(routing_state)
                and "generate_code" in masked_actions
            ):
                next_action = "generate_code"
            else:
                next_action = select_planner_fallback_action(routing_state, set(masked_actions))
        if _is_soft_qa_followup(dict(routing_state.get("meta") or {})) and next_action == "qa":
            next_action = "clarification" if "clarification" in masked_actions else "qa"
        routing_state = _record_planner_decision(
            routing_state,
            planner_action,
            thought,
            next_action,
            diagnostics,
        )

    state = routing_state
    orchestrator_state = dict(state.get("orchestrator", {}))
    meta = dict(state.get("meta") or {})

    observations = list(state.get("observations", []))
    if not (memory_clarification_selected and next_action == "clarification"):
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

    state = {
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
    if current_hash and fresh_unanswered_user_turn and not has_conversation_event(
        state,
        event_type="user",
        user_turn_hash=current_hash,
        actor="user",
    ):
        user_text = latest_user_message(state)
        state = append_conversation_event(
            state,
            build_user_event(
                actor="user",
                user_turn_hash=current_hash,
                text=user_text,
            ),
        )

    if (
        current_hash
        and memory_clarification_question
        and not has_conversation_event(
            state,
            event_type="clarification",
            user_turn_hash=current_hash,
            actor="orchestrator",
        )
    ):
        state = append_conversation_event(
            state,
            build_clarification_event(
                actor="orchestrator",
                user_turn_hash=current_hash,
                text=memory_clarification_question,
                status="active",
            ),
        )

    if next_action:
        state = append_conversation_event(
            state,
            build_routing_decision_event(
                actor="orchestrator",
                user_turn_hash=current_hash,
                decision=next_action,
            ),
        )

    return state
