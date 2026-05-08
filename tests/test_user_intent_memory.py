from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph.memory import ensure_memory_state


def _old_state() -> dict:
    return {
        "messages": [],
        "output": {},
        "artifacts": {"files": {}, "datasets": {}},
        "next_action": None,
        "last_action": None,
        "observations": [],
        "orchestrator": {},
        "planner": {},
        "agents": {},
        "node_data": {},
        "meta": {},
    }


def test_ensure_memory_state_initializes_user_intent_keys() -> None:
    state, memory = ensure_memory_state(_old_state())

    assert memory["user_intents"] == {}
    assert memory["intent_order"] == []
    assert memory["last_user_intent_id"] is None
    assert memory["last_user_intent_id_by_kind"] == {}
    assert state["memory"] is memory


def test_ensure_memory_state_prunes_stale_user_intent_indexes() -> None:
    state = _old_state()
    state["memory"] = {
        "completed_tasks": {},
        "failed_tasks": {},
        "task_order": [],
        "last_task_id": None,
        "last_task_id_by_kind": {},
        "last_failed_task_id_by_kind": {},
        "last_reference_resolution": None,
        "pending_reference_clarification": None,
        "user_intents": {
            "intent_keep": {
                "intent_id": "intent_keep",
                "kind": "db_rag_query",
                "source_question": "Query age",
                "goal_text": "Query age",
                "status": "cancelled",
                "created_at": "2026-05-08T00:00:00+00:00",
                "updated_at": "2026-05-08T00:00:00+00:00",
            }
        },
        "intent_order": ["intent_missing", "intent_keep"],
        "last_user_intent_id": "intent_missing",
        "last_user_intent_id_by_kind": {"db_rag_query": "intent_missing"},
    }

    _state, memory = ensure_memory_state(state)

    assert memory["intent_order"] == ["intent_keep"]
    assert memory["last_user_intent_id"] == "intent_keep"
    assert memory["last_user_intent_id_by_kind"] == {}
