from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph.memory import (
    compact_user_intent_cards,
    ensure_memory_state,
    latest_user_intent,
    link_user_intent_completed_task,
    update_user_intent_status,
    upsert_user_intent_from_db_rag_intent,
)


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


def test_upsert_user_intent_from_db_rag_intent_creates_compact_card() -> None:
    state = upsert_user_intent_from_db_rag_intent(
        _old_state(),
        active_intent={
            "intent_id": "rag-1",
            "source_question": "Which forms contain age?",
            "goal_text": "Find forms that contain age fields.",
            "sql": "SELECT secret FROM payload",
        },
        source_message_hash="hash-1",
        status="active",
    )
    memory = state["memory"]
    intent_id = memory["intent_order"][0]
    card = memory["user_intents"][intent_id]

    assert card["intent_id"] == intent_id
    assert card["display_ordinal"] == 1
    assert card["kind"] == "db_rag_query"
    assert card["agent"] == "rag_db_qa"
    assert card["source_question"] == "Which forms contain age?"
    assert card["goal_text"] == "Find forms that contain age fields."
    assert card["status"] == "active"
    assert card["active_intent_id"] == "rag-1"
    assert card["source_message_hash"] == "hash-1"
    assert card["completed_task_id"] is None
    assert "sql" not in card
    assert memory["last_user_intent_id"] == intent_id
    assert memory["last_user_intent_id_by_kind"] == {"db_rag_query": intent_id}
    assert compact_user_intent_cards(state) == [card]


def test_upsert_same_db_rag_active_intent_updates_without_reordering_or_source_change() -> None:
    state = upsert_user_intent_from_db_rag_intent(
        _old_state(),
        active_intent={
            "intent_id": "rag-1",
            "source_question": "Initial age question",
            "goal_text": "Find age fields.",
        },
        source_message_hash="hash-1",
        status="active",
    )
    intent_id = state["memory"]["last_user_intent_id"]
    created_at = state["memory"]["user_intents"][intent_id]["created_at"]

    state = upsert_user_intent_from_db_rag_intent(
        state,
        active_intent={
            "intent_id": "rag-1",
            "source_question": "Refine to baseline age only",
            "goal_text": "Find baseline age fields.",
        },
        source_message_hash="hash-2",
        status="awaiting_column_review",
    )

    memory = state["memory"]
    card = memory["user_intents"][intent_id]
    assert memory["intent_order"] == [intent_id]
    assert memory["last_user_intent_id"] == intent_id
    assert memory["last_user_intent_id_by_kind"] == {"db_rag_query": intent_id}
    assert card["source_question"] == "Initial age question"
    assert card["goal_text"] == "Find baseline age fields."
    assert card["status"] == "awaiting_column_review"
    assert card["source_message_hash"] == "hash-2"
    assert card["created_at"] == created_at
    assert card["updated_at"] >= created_at


def test_update_user_intent_status_by_active_intent_id_does_not_reorder_or_move_last() -> None:
    state = upsert_user_intent_from_db_rag_intent(
        _old_state(),
        active_intent={
            "intent_id": "rag-1",
            "source_question": "First question",
            "goal_text": "Find first fields.",
        },
        source_message_hash="hash-1",
        status="active",
    )
    first_id = state["memory"]["last_user_intent_id"]
    state = upsert_user_intent_from_db_rag_intent(
        state,
        active_intent={
            "intent_id": "rag-2",
            "source_question": "Second question",
            "goal_text": "Find second fields.",
        },
        source_message_hash="hash-2",
        status="active",
        force_new=True,
    )
    second_id = state["memory"]["last_user_intent_id"]

    state = update_user_intent_status(state, active_intent_id="rag-1", status="cancelled")

    memory = state["memory"]
    assert memory["intent_order"] == [first_id, second_id]
    assert memory["last_user_intent_id"] == second_id
    assert memory["last_user_intent_id_by_kind"] == {"db_rag_query": second_id}
    assert memory["user_intents"][first_id]["status"] == "cancelled"


def test_link_user_intent_completed_task_marks_completed_and_stores_task_id() -> None:
    state = upsert_user_intent_from_db_rag_intent(
        _old_state(),
        active_intent={
            "intent_id": "rag-1",
            "source_question": "Extract diabetes cohort",
            "goal_text": "Create a diabetes subset.",
        },
        source_message_hash="hash-1",
        status="awaiting_sql_review",
    )
    intent_id = state["memory"]["last_user_intent_id"]

    state = link_user_intent_completed_task(state, intent_id=intent_id, task_id="task_1234abcd")

    card = state["memory"]["user_intents"][intent_id]
    assert card["status"] == "completed"
    assert card["completed_task_id"] == "task_1234abcd"


def test_latest_user_intent_returns_latest_card_by_kind() -> None:
    state = upsert_user_intent_from_db_rag_intent(
        _old_state(),
        active_intent={
            "intent_id": "rag-1",
            "source_question": "First question",
            "goal_text": "Find first fields.",
        },
        source_message_hash="hash-1",
        status="active",
    )
    first_id = state["memory"]["last_user_intent_id"]
    state = upsert_user_intent_from_db_rag_intent(
        state,
        active_intent={
            "intent_id": "rag-2",
            "source_question": "Second question",
            "goal_text": "Find second fields.",
        },
        source_message_hash="hash-2",
        status="active",
        force_new=True,
    )
    second_id = state["memory"]["last_user_intent_id"]

    latest = latest_user_intent(state, kind="db_rag_query")
    latest["goal_text"] = "mutated copy"

    assert latest["intent_id"] == second_id
    assert state["memory"]["user_intents"][second_id]["goal_text"] == "Find second fields."
    assert first_id != second_id
