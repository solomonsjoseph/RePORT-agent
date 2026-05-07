from __future__ import annotations

import json

from ...memory.task_store import latest_task_cards
from ...state import MetaKeys
from ...state_views import get_artifacts, get_node_data, get_planner_state, merge_state_patch
from ...workflow_config import (
    PLANNER_DECISION_TRACE_LIMIT,
    RECENT_OBSERVATIONS_LIMIT,
    WORKFLOW_TRACE_TAIL,
)
from .policy import _latest_attachment_summary
from ..action_metadata import ACTION_CAPABILITIES
from .state_logic import _latest_user_message, build_planner_recent_turns


MAX_PLANNER_ENV_MESSAGES = 5
MAX_PLANNER_ENV_EVENTS = 12
MAX_PLANNER_ENV_TASKS = 12


def build_planner_runtime_state(state: dict, available_actions: list[str]) -> dict:
    node_names = set(available_actions)
    node_names.update({"executor", "human_review"})
    node_data = {name: get_node_data(state, name) for name in sorted(node_names)}
    planner_state = get_planner_state(state)

    return merge_state_patch(
        state,
        {
            "artifacts": get_artifacts(state),
            "node_data": node_data,
            "planner": planner_state,
        },
    )


def _message_content(message) -> str:
    return str(getattr(message, "content", "") or "").strip()


def _recent_messages_by_role(state: dict, role: str, *, limit: int) -> list[str]:
    values: list[str] = []
    for message in reversed(list(state.get("messages", []))):
        if getattr(message, "type", None) != role:
            continue
        text = _message_content(message)
        if text:
            values.append(text)
        if len(values) >= limit:
            break
    return list(reversed(values))


def _summarize_assistant_messages(state: dict) -> list[str]:
    summaries: list[str] = []
    for text in _recent_messages_by_role(state, "ai", limit=MAX_PLANNER_ENV_MESSAGES):
        normalized = " ".join(text.split())
        summaries.append(normalized[:500])
    return summaries


def _compact_conversation_events(artifacts: dict) -> list[dict[str, object]]:
    events = list(artifacts.get("conversation_events") or [])
    compacted: list[dict[str, object]] = []
    for event in events[-MAX_PLANNER_ENV_EVENTS:]:
        if not isinstance(event, dict):
            continue
        item: dict[str, object] = {
            "type": event.get("type"),
            "actor": event.get("actor"),
        }
        for field in ("text", "decision", "review_kind", "status", "artifact_id"):
            value = event.get(field)
            if isinstance(value, (str, int, float, bool)) or value is None:
                item[field] = value
        compacted.append(item)
    return compacted


def _dataset_source_task_ids(tasks: list[dict]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for task in tasks:
        artifact_refs = task.get("artifact_refs") if isinstance(task, dict) else {}
        if not isinstance(artifact_refs, dict):
            artifact_refs = {}
        dataset_id = task.get("dataset_id") or artifact_refs.get("dataset_artifact_id")
        task_id = task.get("task_id")
        if isinstance(dataset_id, str) and isinstance(task_id, str):
            mapping.setdefault(dataset_id, task_id)
    return mapping


def _compact_task_cards(state: dict, artifacts: dict) -> list[dict[str, object]]:
    tasks = latest_task_cards(state, limit=MAX_PLANNER_ENV_TASKS)
    datasets = dict(artifacts.get("datasets") or {})
    compacted: list[dict[str, object]] = []
    for task in tasks:
        artifact_refs = task.get("artifact_refs") if isinstance(task, dict) else {}
        dataset_id = artifact_refs.get("dataset_artifact_id") if isinstance(artifact_refs, dict) else None
        dataset = datasets.get(dataset_id) if isinstance(dataset_id, str) else None
        item: dict[str, object] = {
            "task_id": task.get("task_id"),
            "display_ordinal": task.get("display_ordinal"),
            "kind": task.get("kind"),
            "label": task.get("label"),
            "source_question": task.get("source_question"),
            "goal_text": task.get("goal_text"),
            "summary": task.get("summary"),
            "dataset_id": dataset_id,
        }
        if isinstance(dataset, dict):
            item["columns"] = list(dataset.get("columns") or [])
            item["row_count"] = dataset.get("row_count", dataset.get("rows"))
        compacted.append(item)
    return compacted


def _compact_dataset_cards(artifacts: dict, tasks: list[dict[str, object]]) -> list[dict[str, object]]:
    source_task_ids = _dataset_source_task_ids(tasks)
    datasets = dict(artifacts.get("datasets") or {})
    cards: list[dict[str, object]] = []
    for dataset_id, dataset in datasets.items():
        if not isinstance(dataset, dict):
            continue
        cards.append(
            {
                "dataset_id": dataset.get("id") or dataset_id,
                "kind": dataset.get("kind"),
                "columns": list(dataset.get("columns") or []),
                "row_count": dataset.get("row_count", dataset.get("rows")),
                "source_task_id": source_task_ids.get(str(dataset.get("id") or dataset_id)),
            }
        )
    return cards


def build_planner_environment(state: dict, available_actions: list[str]) -> dict[str, object]:
    artifacts = get_artifacts(state)
    planner_state = get_planner_state(state)
    planner_memory = dict(planner_state.get("memory") or {})
    candidate_tasks = _compact_task_cards(state, artifacts)
    return {
        "latest_user_message": _latest_user_message(state),
        "recent_user_messages": _recent_messages_by_role(
            state,
            "human",
            limit=MAX_PLANNER_ENV_MESSAGES,
        ),
        "recent_assistant_summaries": _summarize_assistant_messages(state),
        "active_goal": planner_memory.get("active_user_goal"),
        "workflow_state": {
            "last_action": state.get("last_action"),
            "workflow_trace_tail": list(
                (state.get("meta") or {}).get(MetaKeys.WORKFLOW_TRACE, [])
            )[-WORKFLOW_TRACE_TAIL:],
            "pending_review": bool((state.get("agents") or {}).get("human_review")),
            "pending_clarification": bool(
                (state.get("meta") or {}).get(MetaKeys.AWAITING_USER_CLARIFICATION)
            ),
            "workflow_milestone": (state.get("meta") or {}).get(MetaKeys.WORKFLOW_MILESTONE),
            "completion_status": (state.get("meta") or {}).get(MetaKeys.COMPLETION_STATUS),
        },
        "conversation_events": _compact_conversation_events(artifacts),
        "candidate_tasks": candidate_tasks,
        "candidate_datasets": _compact_dataset_cards(artifacts, candidate_tasks),
        "available_actions": list(available_actions),
    }


def build_planner_context(state: dict, available_actions: list[str]) -> dict:
    artifacts = get_artifacts(state)
    planner_state = get_planner_state(state)
    planner_memory = dict(planner_state.get("memory") or {})
    meta = dict(state.get("meta") or {})
    observations = list(state.get("observations") or [])
    workflow_trace = list(meta.get(MetaKeys.WORKFLOW_TRACE, []))
    executor = get_node_data(state, "executor")
    review = get_node_data(state, "human_review")
    recent_turns_for_planner = build_planner_recent_turns(state)
    planner_environment = build_planner_environment(state, available_actions)
    summary_lines = [
        f"latest_user_message={_latest_user_message(state)}",
        f"last_action={state.get('last_action')}",
        f"workflow_trace_tail={workflow_trace[-WORKFLOW_TRACE_TAIL:]}",
        f"recent_observations={observations[-RECENT_OBSERVATIONS_LIMIT:]}",
        f"generated_code_present={bool(artifacts.get('generated_code'))}",
        f"executor_run_status={executor.get('run_status')}",
        f"review_state={json.dumps(review, default=str, sort_keys=True)}",
        f"workflow_milestone={meta.get(MetaKeys.WORKFLOW_MILESTONE)}",
        f"completion_status={meta.get(MetaKeys.COMPLETION_STATUS)}",
        f"blocker_signature={meta.get(MetaKeys.BLOCKER_SIGNATURE)}",
        f"progress_class={meta.get(MetaKeys.PROGRESS_CLASS)}",
        f"progress_made_last_step={meta.get(MetaKeys.PROGRESS_MADE_LAST_STEP)}",
        f"stagnation_count={meta.get(MetaKeys.STAGNATION_COUNT, 0)}",
        f"weak_progress_count={meta.get(MetaKeys.WEAK_PROGRESS_COUNT, 0)}",
        f"repeated_failure_signature={meta.get(MetaKeys.REPEATED_FAILURE_SIGNATURE)}",
    ]
    node_caps = "\n".join(
        f"- {name}: {ACTION_CAPABILITIES[name]}"
        for name in available_actions
        if name in ACTION_CAPABILITIES
    )
    return {
        "environment_summary": "\n".join(summary_lines),
        "node_capabilities": node_caps,
        "recent_observations": observations[-RECENT_OBSERVATIONS_LIMIT:],
        "planner_decision_trace": list(planner_state.get("decision_trace") or [])[
            -PLANNER_DECISION_TRACE_LIMIT:
        ],
        "planner_memory": planner_memory,
        "planner_environment": planner_environment,
        "recent_turns_for_planner": recent_turns_for_planner,
        "latest_turn_attachments": _latest_attachment_summary(state),
        "artifacts": artifacts,
    }
