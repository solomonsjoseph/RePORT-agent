# User Intent History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add current-thread DB-RAG user-intent history so cancelled or incomplete database queries remain semantically referenceable without polluting completed task memory.

**Architecture:** Extend the existing `graph/memory` package with compact user-intent cards and helpers. Add a narrow semantic classifier for user-intent references, validate classifier output in code, and hand validated references through one-turn orchestrator meta keys. DB-RAG writes and updates intent records at existing active-intent lifecycle points, and completed DB-RAG SQL extraction links back to the originating intent only after the existing SQL execution completion gate fires.

**Tech Stack:** Python, LangGraph state dicts, existing `graph/memory` task-store patterns, pytest.

---

## File Map

- Modify `graph/memory/schema.py`
  - Add user-intent constants and `UserIntentCard` typed shape.
- Modify `graph/memory/task_store.py`
  - Extend memory initialization/cleanup with `user_intents`, `intent_order`, and last-intent pointers.
- Create `graph/memory/user_intent_store.py`
  - Own all user-intent CRUD, compact cards, classifier payload building, classifier output validation.
- Modify `graph/memory/__init__.py`
  - Export user-intent helpers.
- Modify `graph/state.py`
  - Add `MetaKeys` constants for validated user-intent routing handoff.
- Modify `graph/nodes/orchestrator/node.py`
  - Clear/consume resolved user-intent meta, run semantic classifier on fresh turns, route validated continuation to DB-RAG.
- Modify `graph/nodes/orchestrator/context_builder.py`
  - Include capped recent user-intent cards and validated reference in planner environment.
- Modify `graph/nodes/db_rag_qa/node.py`
  - Upsert DB-RAG user intents on fresh questions, opt-in/refinement paths, and continuation handoff.
- Modify `graph/nodes/db_rag_qa/helpers.py`
  - Link completed `db_rag_sql_extraction` task to originating user intent after successful reviewed SQL execution.
- Modify `graph/nodes/human_review_rag_db_column_selection.py`
  - Mark originating user intent cancelled when column review cancel succeeds.
- Modify `graph/nodes/human_review_rag_db_sql_execution.py`
  - Mark originating user intent cancelled when SQL review cancel succeeds.
- Add/modify tests:
  - `tests/test_user_intent_memory.py`
  - `tests/test_user_intent_reference_classifier.py`
  - `tests/test_orchestrator_user_intent_resolution.py`
  - `tests/test_rag_db_qa_node.py`
  - `tests/test_review_control_events.py`
  - `tests/test_orchestrator_planner_environment.py`

---

### Task 1: Extend Memory Schema And Initialization

**Files:**
- Modify: `graph/memory/schema.py`
- Modify: `graph/memory/task_store.py`
- Modify: `graph/memory/__init__.py`
- Test: `tests/test_user_intent_memory.py`
- Test: `tests/test_task_memory.py`

- [ ] **Step 1: Write failing memory initialization tests**

Create `tests/test_user_intent_memory.py`:

```python
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
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
pytest tests/test_user_intent_memory.py -q
```

Expected: fail with `KeyError: 'user_intents'` or missing exports.

- [ ] **Step 3: Add schema constants and typed card**

In `graph/memory/schema.py`, add:

```python
ALLOWED_USER_INTENT_KINDS = {"db_rag_query"}
ALLOWED_USER_INTENT_STATUSES = {
    "active",
    "awaiting_extraction_opt_in",
    "awaiting_column_review",
    "awaiting_sql_review",
    "cancelled",
    "completed",
    "superseded",
    "declined",
}


class UserIntentCard(TypedDict, total=False):
    intent_id: str
    display_ordinal: int
    kind: str
    agent: str
    source_question: str
    goal_text: str
    status: str
    source_message_hash: str | None
    active_intent_id: str | None
    completed_task_id: str | None
    continued_from_intent_id: str | None
    created_at: str
    updated_at: str
```

Extend `MemoryState`:

```python
class MemoryState(TypedDict):
    completed_tasks: dict[str, TaskCard]
    failed_tasks: dict[str, TaskCard]
    task_order: list[str]
    last_task_id: str | None
    last_task_id_by_kind: dict[str, str]
    last_failed_task_id_by_kind: dict[str, str]
    last_reference_resolution: dict[str, Any] | None
    pending_reference_clarification: dict[str, Any] | None
    user_intents: dict[str, UserIntentCard]
    intent_order: list[str]
    last_user_intent_id: str | None
    last_user_intent_id_by_kind: dict[str, str]
```

- [ ] **Step 4: Extend memory defaults and cleanup**

In `graph/memory/task_store.py`, extend `EMPTY_MEMORY_STATE`:

```python
EMPTY_MEMORY_STATE: MemoryState = {
    "completed_tasks": {},
    "failed_tasks": {},
    "task_order": [],
    "last_task_id": None,
    "last_task_id_by_kind": {},
    "last_failed_task_id_by_kind": {},
    "last_reference_resolution": None,
    "pending_reference_clarification": None,
    "user_intents": {},
    "intent_order": [],
    "last_user_intent_id": None,
    "last_user_intent_id_by_kind": {},
}
```

Extend `MEMORY_KEY_TYPES`:

```python
MEMORY_KEY_TYPES = {
    "completed_tasks": dict,
    "failed_tasks": dict,
    "task_order": list,
    "last_task_id": (str, type(None)),
    "last_task_id_by_kind": dict,
    "last_failed_task_id_by_kind": dict,
    "last_reference_resolution": (dict, type(None)),
    "pending_reference_clarification": (dict, type(None)),
    "user_intents": dict,
    "intent_order": list,
    "last_user_intent_id": (str, type(None)),
    "last_user_intent_id_by_kind": dict,
}
```

At the end of `ensure_memory_state`, before `return state, memory`, add:

```python
    user_intents = memory["user_intents"]
    intent_order = [
        intent_id
        for intent_id in memory["intent_order"]
        if isinstance(intent_id, str) and intent_id in user_intents
    ]
    memory["intent_order"] = intent_order
    if memory["last_user_intent_id"] not in user_intents:
        memory["last_user_intent_id"] = intent_order[-1] if intent_order else None
    memory["last_user_intent_id_by_kind"] = {
        kind: intent_id
        for kind, intent_id in memory["last_user_intent_id_by_kind"].items()
        if isinstance(kind, str) and isinstance(intent_id, str) and intent_id in user_intents
    }
```

- [ ] **Step 5: Export schema symbols**

In `graph/memory/__init__.py`, import and export `ALLOWED_USER_INTENT_KINDS`, `ALLOWED_USER_INTENT_STATUSES`, and `UserIntentCard`.

- [ ] **Step 6: Update existing expected empty memory test**

In `tests/test_task_memory.py`, update `EXPECTED_EMPTY_MEMORY` to include:

```python
    "user_intents": {},
    "intent_order": [],
    "last_user_intent_id": None,
    "last_user_intent_id_by_kind": {},
```

- [ ] **Step 7: Run tests**

Run:

```bash
pytest tests/test_user_intent_memory.py tests/test_task_memory.py -q
```

Expected: all selected tests pass.

- [ ] **Step 8: Commit**

```bash
git add graph/memory/schema.py graph/memory/task_store.py graph/memory/__init__.py tests/test_user_intent_memory.py tests/test_task_memory.py
git commit -m "Add user intent memory state"
```

---

### Task 2: Add User Intent Store Helpers

**Files:**
- Create: `graph/memory/user_intent_store.py`
- Modify: `graph/memory/__init__.py`
- Test: `tests/test_user_intent_memory.py`

- [ ] **Step 1: Add failing helper tests**

Append to `tests/test_user_intent_memory.py`:

```python
from graph.memory import (
    latest_user_intent,
    link_user_intent_completed_task,
    update_user_intent_status,
    upsert_user_intent_from_db_rag_intent,
)


def test_upsert_db_rag_user_intent_creates_compact_card() -> None:
    state = _old_state()

    updated = upsert_user_intent_from_db_rag_intent(
        state,
        active_intent={
            "intent_id": "rag-intent-1",
            "source_question": "Query my database for age",
            "goal_text": "Find age columns",
        },
        source_message_hash="hash-1",
        status="active",
    )

    memory = updated["memory"]
    intent_id = memory["last_user_intent_id"]
    card = memory["user_intents"][intent_id]
    assert card["kind"] == "db_rag_query"
    assert card["agent"] == "rag_db_qa"
    assert card["source_question"] == "Query my database for age"
    assert card["goal_text"] == "Find age columns"
    assert card["status"] == "active"
    assert card["active_intent_id"] == "rag-intent-1"
    assert card["source_message_hash"] == "hash-1"
    assert card["completed_task_id"] is None
    assert memory["intent_order"] == [intent_id]
    assert memory["last_user_intent_id_by_kind"]["db_rag_query"] == intent_id


def test_upsert_updates_matching_active_intent_without_reordering() -> None:
    state = upsert_user_intent_from_db_rag_intent(
        _old_state(),
        active_intent={
            "intent_id": "rag-intent-1",
            "source_question": "Query my database for age",
            "goal_text": "Find age columns",
        },
        source_message_hash="hash-1",
        status="active",
    )
    intent_id = state["memory"]["last_user_intent_id"]

    updated = upsert_user_intent_from_db_rag_intent(
        state,
        active_intent={
            "intent_id": "rag-intent-1",
            "source_question": "Query my database for age",
            "goal_text": "Find age and gender columns",
        },
        source_message_hash="hash-2",
        status="awaiting_column_review",
    )

    assert updated["memory"]["intent_order"] == [intent_id]
    card = updated["memory"]["user_intents"][intent_id]
    assert card["goal_text"] == "Find age and gender columns"
    assert card["status"] == "awaiting_column_review"
    assert card["source_question"] == "Query my database for age"


def test_cancel_status_update_does_not_reorder_intents() -> None:
    state = upsert_user_intent_from_db_rag_intent(
        _old_state(),
        active_intent={"intent_id": "rag-1", "source_question": "Query age", "goal_text": "Query age"},
        source_message_hash="hash-1",
        status="active",
    )
    first_id = state["memory"]["last_user_intent_id"]
    state = upsert_user_intent_from_db_rag_intent(
        state,
        active_intent={"intent_id": "rag-2", "source_question": "Query gender", "goal_text": "Query gender"},
        source_message_hash="hash-2",
        status="active",
    )
    second_id = state["memory"]["last_user_intent_id"]

    updated = update_user_intent_status(
        state,
        active_intent_id="rag-1",
        status="cancelled",
    )

    assert updated["memory"]["intent_order"] == [first_id, second_id]
    assert updated["memory"]["last_user_intent_id"] == second_id
    assert updated["memory"]["user_intents"][first_id]["status"] == "cancelled"


def test_link_user_intent_completed_task() -> None:
    state = upsert_user_intent_from_db_rag_intent(
        _old_state(),
        active_intent={"intent_id": "rag-1", "source_question": "Query age", "goal_text": "Query age"},
        source_message_hash="hash-1",
        status="awaiting_sql_review",
    )
    intent_id = state["memory"]["last_user_intent_id"]

    updated = link_user_intent_completed_task(state, intent_id=intent_id, task_id="task_1234abcd")

    card = updated["memory"]["user_intents"][intent_id]
    assert card["status"] == "completed"
    assert card["completed_task_id"] == "task_1234abcd"


def test_latest_user_intent_returns_latest_by_kind() -> None:
    state = upsert_user_intent_from_db_rag_intent(
        _old_state(),
        active_intent={"intent_id": "rag-1", "source_question": "Query age", "goal_text": "Query age"},
        source_message_hash="hash-1",
        status="cancelled",
    )

    card = latest_user_intent(state, kind="db_rag_query")

    assert card is not None
    assert card["active_intent_id"] == "rag-1"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
pytest tests/test_user_intent_memory.py -q
```

Expected: fail with import errors for user-intent helpers.

- [ ] **Step 3: Implement user-intent store**

Create `graph/memory/user_intent_store.py`:

```python
from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from graph.state import AgentState

from .schema import (
    ALLOWED_USER_INTENT_KINDS,
    ALLOWED_USER_INTENT_STATUSES,
    require_json_safe,
)
from .task_store import ensure_memory_state


LIVE_USER_INTENT_STATUSES = {
    "active",
    "awaiting_extraction_opt_in",
    "awaiting_column_review",
    "awaiting_sql_review",
}

COMPACT_USER_INTENT_FIELDS = (
    "intent_id",
    "display_ordinal",
    "kind",
    "agent",
    "source_question",
    "goal_text",
    "status",
    "source_message_hash",
    "active_intent_id",
    "completed_task_id",
    "continued_from_intent_id",
    "created_at",
    "updated_at",
)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _new_intent_id(memory: dict[str, Any]) -> str:
    intent_id = f"intent_{uuid4().hex[:8]}"
    while intent_id in memory["user_intents"]:
        intent_id = f"intent_{uuid4().hex[:8]}"
    return intent_id


def _validate_kind(kind: str) -> None:
    if kind not in ALLOWED_USER_INTENT_KINDS:
        raise ValueError(f"Unknown user intent kind: {kind}")


def _validate_status(status: str) -> None:
    if status not in ALLOWED_USER_INTENT_STATUSES:
        raise ValueError(f"Unknown user intent status: {status}")


def _validate_text(value: Any, field_name: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    require_json_safe(value, field_name)


def _matching_intent_id(memory: dict[str, Any], active_intent_id: str) -> str | None:
    for intent_id, card in memory["user_intents"].items():
        if isinstance(card, dict) and card.get("active_intent_id") == active_intent_id:
            return intent_id
    return None


def _latest_live_intent_id(memory: dict[str, Any], kind: str) -> str | None:
    candidate = memory["last_user_intent_id_by_kind"].get(kind)
    card = memory["user_intents"].get(candidate) if isinstance(candidate, str) else None
    if isinstance(card, dict) and card.get("status") in LIVE_USER_INTENT_STATUSES:
        return candidate
    return None


def compact_user_intent_cards(
    state: AgentState,
    *,
    limit: int = 12,
    kind: str | None = None,
) -> list[dict[str, Any]]:
    _state, memory = ensure_memory_state(state)
    ordered_ids = list(reversed(memory["intent_order"]))
    cards: list[dict[str, Any]] = []
    for intent_id in ordered_ids:
        card = memory["user_intents"].get(intent_id)
        if not isinstance(card, dict):
            continue
        if kind is not None and card.get("kind") != kind:
            continue
        cards.append({field: deepcopy(card.get(field)) for field in COMPACT_USER_INTENT_FIELDS})
        if len(cards) >= limit:
            break
    return cards


def latest_user_intent(state: AgentState, *, kind: str) -> dict[str, Any] | None:
    _validate_kind(kind)
    _state, memory = ensure_memory_state(state)
    candidate = memory["last_user_intent_id_by_kind"].get(kind)
    if isinstance(candidate, str) and candidate in memory["user_intents"]:
        return deepcopy(memory["user_intents"][candidate])
    for intent_id in reversed(memory["intent_order"]):
        card = memory["user_intents"].get(intent_id)
        if isinstance(card, dict) and card.get("kind") == kind:
            return deepcopy(card)
    return None


def upsert_user_intent_from_db_rag_intent(
    state: AgentState,
    *,
    active_intent: dict[str, Any],
    source_message_hash: str | None,
    status: str,
    continued_from_intent_id: str | None = None,
    force_new: bool = False,
) -> AgentState:
    state, memory = ensure_memory_state(state)
    _validate_status(status)
    active_intent_id = str(active_intent.get("intent_id") or "").strip()
    source_question = str(active_intent.get("source_question") or "").strip()
    goal_text = str(active_intent.get("goal_text") or source_question).strip()
    _validate_text(source_question, "source_question")
    _validate_text(goal_text, "goal_text")
    if source_message_hash is not None:
        _validate_text(source_message_hash, "source_message_hash")
    if continued_from_intent_id is not None and continued_from_intent_id not in memory["user_intents"]:
        raise ValueError(f"Unknown continued_from_intent_id: {continued_from_intent_id}")

    intent_id = None if force_new else _matching_intent_id(memory, active_intent_id)
    if intent_id is None and not force_new:
        intent_id = _latest_live_intent_id(memory, "db_rag_query")

    timestamp = _now_iso()
    if intent_id is None:
        intent_id = _new_intent_id(memory)
        card = {
            "intent_id": intent_id,
            "display_ordinal": len(memory["intent_order"]) + 1,
            "kind": "db_rag_query",
            "agent": "rag_db_qa",
            "source_question": source_question,
            "goal_text": goal_text,
            "status": status,
            "source_message_hash": source_message_hash,
            "active_intent_id": active_intent_id or None,
            "completed_task_id": None,
            "continued_from_intent_id": continued_from_intent_id,
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        require_json_safe(card, "user_intent")
        memory["user_intents"][intent_id] = card
        memory["intent_order"].append(intent_id)
    else:
        card = memory["user_intents"][intent_id]
        card["goal_text"] = goal_text
        card["status"] = status
        card["active_intent_id"] = active_intent_id or card.get("active_intent_id")
        card["updated_at"] = timestamp
        if source_message_hash is not None:
            card["source_message_hash"] = source_message_hash
        require_json_safe(card, "user_intent")

    memory["last_user_intent_id"] = intent_id
    memory["last_user_intent_id_by_kind"]["db_rag_query"] = intent_id
    return state


def update_user_intent_status(
    state: AgentState,
    *,
    status: str,
    intent_id: str | None = None,
    active_intent_id: str | None = None,
) -> AgentState:
    state, memory = ensure_memory_state(state)
    _validate_status(status)
    target_id = intent_id
    if target_id is None and active_intent_id:
        target_id = _matching_intent_id(memory, active_intent_id)
    if target_id is None or target_id not in memory["user_intents"]:
        return state
    card = memory["user_intents"][target_id]
    card["status"] = status
    card["updated_at"] = _now_iso()
    require_json_safe(card, "user_intent")
    return state


def link_user_intent_completed_task(
    state: AgentState,
    *,
    intent_id: str,
    task_id: str,
) -> AgentState:
    state, memory = ensure_memory_state(state)
    if intent_id not in memory["user_intents"]:
        raise ValueError(f"Unknown intent_id: {intent_id}")
    if task_id not in memory["completed_tasks"]:
        raise ValueError(f"Unknown task_id: {task_id}")
    card = memory["user_intents"][intent_id]
    card["status"] = "completed"
    card["completed_task_id"] = task_id
    card["updated_at"] = _now_iso()
    require_json_safe(card, "user_intent")
    return state
```

- [ ] **Step 4: Export helpers**

In `graph/memory/__init__.py`, import and export:

```python
from .user_intent_store import (
    compact_user_intent_cards,
    latest_user_intent,
    link_user_intent_completed_task,
    update_user_intent_status,
    upsert_user_intent_from_db_rag_intent,
)
```

Add those names to `__all__`.

- [ ] **Step 5: Run tests**

Run:

```bash
pytest tests/test_user_intent_memory.py tests/test_task_memory.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add graph/memory/user_intent_store.py graph/memory/__init__.py tests/test_user_intent_memory.py
git commit -m "Add user intent store helpers"
```

---

### Task 3: Add Semantic Classifier Contract And Validation

**Files:**
- Modify: `graph/memory/user_intent_store.py`
- Test: `tests/test_user_intent_reference_classifier.py`

- [ ] **Step 1: Write failing classifier tests**

Create `tests/test_user_intent_reference_classifier.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph.memory import (
    build_user_intent_classifier_payload,
    classify_user_intent_reference,
    validate_user_intent_reference,
    upsert_user_intent_from_db_rag_intent,
)


class StubClassifier:
    def __init__(self, result):
        self.result = result
        self.prompt = None

    def invoke(self, prompt):
        self.prompt = prompt
        return self.result


def _state_with_intent() -> dict:
    state = {
        "messages": [],
        "output": {},
        "artifacts": {"files": {}, "datasets": {}},
        "next_action": None,
        "last_action": None,
        "observations": [],
        "orchestrator": {},
        "planner": {},
        "agents": {"rag_db_qa": {"thread_status": "done", "active_thread": False}},
        "node_data": {},
        "meta": {},
    }
    return upsert_user_intent_from_db_rag_intent(
        state,
        active_intent={
            "intent_id": "rag-intent-1",
            "source_question": "Query my database for age",
            "goal_text": "Query age among index cases",
        },
        source_message_hash="hash-1",
        status="cancelled",
    )


def test_payload_contains_compact_intent_and_no_payload_artifacts() -> None:
    state = _state_with_intent()

    payload = build_user_intent_classifier_payload(
        state,
        user_message="continue previous query",
        user_message_hash="hash-2",
    )

    assert payload["latest_user_message"] == "continue previous query"
    assert payload["candidate_user_intents"][0]["source_question"] == "Query my database for age"
    assert "sql" not in str(payload).lower()
    assert "raw_model_output" not in str(payload)


def test_validate_existing_user_intent_reference() -> None:
    state = _state_with_intent()
    intent_id = state["memory"]["last_user_intent_id"]

    result = validate_user_intent_reference(
        state,
        {
            "target": "existing_user_intent",
            "target_id": intent_id,
            "relationship": "continue",
            "confidence": "high",
            "reason": "User wants to continue the prior query.",
        },
    )

    assert result["target"] == "existing_user_intent"
    assert result["target_id"] == intent_id
    assert result["source_question"] == "Query my database for age"


def test_validate_rejects_invented_intent_id() -> None:
    state = _state_with_intent()

    with pytest.raises(ValueError, match="Resolved user intent does not exist"):
        validate_user_intent_reference(
            state,
            {
                "target": "existing_user_intent",
                "target_id": "intent_missing",
                "relationship": "continue",
                "confidence": "high",
                "reason": "bad",
            },
        )


def test_low_confidence_returns_ambiguous() -> None:
    state = _state_with_intent()
    intent_id = state["memory"]["last_user_intent_id"]

    result = validate_user_intent_reference(
        state,
        {
            "target": "existing_user_intent",
            "target_id": intent_id,
            "relationship": "continue",
            "confidence": "low",
            "reason": "unclear",
        },
    )

    assert result["target"] == "ambiguous"
    assert result["needs_clarification"] is True


def test_classifier_invokes_model_and_validates() -> None:
    state = _state_with_intent()
    intent_id = state["memory"]["last_user_intent_id"]
    classifier = StubClassifier(
        {
            "target": "existing_user_intent",
            "target_id": intent_id,
            "relationship": "continue",
            "confidence": "high",
            "reason": "User wants the prior DB-RAG query.",
        }
    )

    result = classify_user_intent_reference(
        state,
        classifier,
        user_message="continue previous query",
        user_message_hash="hash-2",
    )

    assert result["target"] == "existing_user_intent"
    assert result["source_question"] == "Query my database for age"
    assert classifier.prompt is not None
    assert "candidate_user_intents" in classifier.prompt
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
pytest tests/test_user_intent_reference_classifier.py -q
```

Expected: fail with import errors.

- [ ] **Step 3: Implement payload builder, parser, validator, classifier**

Append to `graph/memory/user_intent_store.py`:

```python
import json


ALLOWED_USER_INTENT_REFERENCE_TARGETS = {
    "new_user_intent",
    "existing_user_intent",
    "completed_task",
    "ambiguous",
}
ALLOWED_USER_INTENT_RELATIONSHIPS = {
    "continue",
    "refine",
    "inspect_result",
    "new_request",
}

CLASSIFIER_INSTRUCTIONS = (
    "Classify whether the latest user message refers to a prior DB-RAG user intent, "
    "a completed task/result, a fresh request, or an ambiguous reference. "
    "Choose only IDs present in candidate_user_intents or candidate_completed_tasks. "
    "Return JSON with target, target_id, relationship, confidence, reason."
)


def build_user_intent_classifier_payload(
    state: AgentState,
    *,
    user_message: str,
    user_message_hash: str,
    recent_turns: list[dict[str, str]] | None = None,
    completed_task_cards: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    agents = dict(state.get("agents") or {})
    rag_state = dict(agents.get("rag_db_qa") or {})
    return {
        "latest_user_message": str(user_message or ""),
        "user_message_hash": user_message_hash,
        "recent_turns": list(recent_turns or [])[-4:],
        "active_workflow": {
            "rag_db_thread_status": rag_state.get("thread_status"),
            "rag_db_active_thread": bool(rag_state.get("active_thread")),
            "pending_review": bool((state.get("agents") or {}).get("human_review")),
            "pending_clarification": bool((state.get("meta") or {}).get("awaiting_user_clarification")),
        },
        "candidate_user_intents": compact_user_intent_cards(state, kind="db_rag_query", limit=8),
        "candidate_completed_tasks": list(completed_task_cards or [])[:8],
    }


def _parse_classifier_response(response: Any) -> dict[str, Any]:
    if isinstance(response, dict):
        return response
    content = getattr(response, "content", response)
    if isinstance(content, dict):
        return content
    text = str(content)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("User-intent classifier response must be a JSON object")
    return parsed


def _confidence_is_low(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() == "low"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value < 0.5
    return value is None


def _ambiguous_result(reason: str) -> dict[str, Any]:
    return {
        "target": "ambiguous",
        "target_id": None,
        "relationship": None,
        "confidence": "low",
        "needs_clarification": True,
        "reason": reason,
    }


def validate_user_intent_reference(state: AgentState, raw_result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw_result, dict):
        raise ValueError("User-intent reference result must be a dict")
    require_json_safe(raw_result, "user_intent_reference")
    target = raw_result.get("target")
    relationship = raw_result.get("relationship")
    if target not in ALLOWED_USER_INTENT_REFERENCE_TARGETS:
        raise ValueError(f"Unknown user-intent reference target: {target}")
    if relationship is not None and relationship not in ALLOWED_USER_INTENT_RELATIONSHIPS:
        raise ValueError(f"Unknown user-intent relationship: {relationship}")
    if target == "ambiguous" or _confidence_is_low(raw_result.get("confidence")):
        return _ambiguous_result(str(raw_result.get("reason") or "The reference is ambiguous."))
    if target == "new_user_intent":
        return {
            "target": "new_user_intent",
            "target_id": None,
            "relationship": "new_request",
            "confidence": raw_result.get("confidence"),
            "needs_clarification": False,
            "reason": str(raw_result.get("reason") or ""),
        }

    _state, memory = ensure_memory_state(state)
    target_id = raw_result.get("target_id")
    if not isinstance(target_id, str) or not target_id:
        return _ambiguous_result("The classifier did not select a target ID.")
    if target == "existing_user_intent":
        card = memory["user_intents"].get(target_id)
        if not isinstance(card, dict):
            raise ValueError(f"Resolved user intent does not exist: {target_id}")
        if card.get("kind") != "db_rag_query":
            raise ValueError(f"Resolved user intent has unsupported kind: {card.get('kind')}")
        source_question = str(card.get("source_question") or "").strip()
        if not source_question:
            raise ValueError("Resolved user intent has no source_question")
        if relationship not in {"continue", "refine"}:
            return _ambiguous_result("The requested relationship is not supported for user intents.")
        return {
            "target": "existing_user_intent",
            "target_id": target_id,
            "kind": "db_rag_query",
            "relationship": relationship,
            "source_question": source_question,
            "goal_text": str(card.get("goal_text") or source_question),
            "confidence": raw_result.get("confidence"),
            "needs_clarification": False,
            "reason": str(raw_result.get("reason") or ""),
        }
    if target == "completed_task":
        completed = memory["completed_tasks"]
        if target_id not in completed:
            raise ValueError(f"Resolved completed task does not exist: {target_id}")
        return {
            "target": "completed_task",
            "target_id": target_id,
            "relationship": relationship,
            "confidence": raw_result.get("confidence"),
            "needs_clarification": False,
            "reason": str(raw_result.get("reason") or ""),
        }
    return _ambiguous_result("The classifier result is ambiguous.")


def classify_user_intent_reference(
    state: AgentState,
    classifier: Any,
    *,
    user_message: str,
    user_message_hash: str,
    recent_turns: list[dict[str, str]] | None = None,
    completed_task_cards: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    payload = build_user_intent_classifier_payload(
        state,
        user_message=user_message,
        user_message_hash=user_message_hash,
        recent_turns=recent_turns,
        completed_task_cards=completed_task_cards,
    )
    if not payload["candidate_user_intents"]:
        return {
            "target": "new_user_intent",
            "target_id": None,
            "relationship": "new_request",
            "confidence": "high",
            "needs_clarification": False,
            "reason": "No user intents exist.",
        }
    if not hasattr(classifier, "invoke"):
        raise ValueError("User-intent classifier must expose invoke")
    prompt = f"{CLASSIFIER_INSTRUCTIONS}\n\nPayload:\n{json.dumps(payload, sort_keys=True)}"
    raw_result = _parse_classifier_response(classifier.invoke(prompt))
    return validate_user_intent_reference(state, raw_result)
```

- [ ] **Step 4: Export helpers**

In `graph/memory/__init__.py`, import/export:

```python
build_user_intent_classifier_payload
classify_user_intent_reference
validate_user_intent_reference
```

- [ ] **Step 5: Run tests**

Run:

```bash
pytest tests/test_user_intent_reference_classifier.py tests/test_user_intent_memory.py -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add graph/memory/user_intent_store.py graph/memory/__init__.py tests/test_user_intent_reference_classifier.py
git commit -m "Add user intent reference classifier"
```

---

### Task 4: Add Orchestrator User-Intent Routing Handoff

**Files:**
- Modify: `graph/state.py`
- Modify: `graph/nodes/orchestrator/node.py`
- Test: `tests/test_orchestrator_user_intent_resolution.py`

- [ ] **Step 1: Write failing orchestrator tests**

Create `tests/test_orchestrator_user_intent_resolution.py` with focused unit tests for pure helpers:

```python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph.memory import upsert_user_intent_from_db_rag_intent
from graph.nodes.orchestrator.node import (
    _apply_resolved_user_intent_meta,
    _clear_consumed_resolved_user_intent_meta,
    _route_from_resolved_user_intent_meta,
)
from graph.state import MetaKeys


def _state_with_cancelled_intent() -> dict:
    state = {
        "messages": [],
        "output": {},
        "artifacts": {"files": {}, "datasets": {}},
        "next_action": None,
        "last_action": None,
        "observations": [],
        "orchestrator": {},
        "planner": {},
        "agents": {"rag_db_qa": {"thread_status": "done", "active_thread": False}},
        "node_data": {},
        "meta": {},
    }
    return upsert_user_intent_from_db_rag_intent(
        state,
        active_intent={
            "intent_id": "rag-1",
            "source_question": "Query my database for age",
            "goal_text": "Query age",
        },
        source_message_hash="hash-1",
        status="cancelled",
    )


def test_apply_resolved_user_intent_meta_copies_source_question_from_card() -> None:
    state = _state_with_cancelled_intent()
    intent_id = state["memory"]["last_user_intent_id"]
    card = state["memory"]["user_intents"][intent_id]

    meta = _apply_resolved_user_intent_meta(
        {},
        {
            "target": "existing_user_intent",
            "target_id": intent_id,
            "relationship": "continue",
            "source_question": "BAD CLASSIFIER TEXT",
        },
        card,
        "hash-2",
    )

    assert meta[MetaKeys.RESOLVED_USER_INTENT_ID] == intent_id
    assert meta[MetaKeys.RESOLVED_USER_INTENT_KIND] == "db_rag_query"
    assert meta[MetaKeys.RESOLVED_USER_INTENT_RELATIONSHIP] == "continue"
    assert meta[MetaKeys.RESOLVED_USER_INTENT_SOURCE_QUESTION] == "Query my database for age"
    assert meta[MetaKeys.RESOLVED_USER_INTENT_USER_MESSAGE_HASH] == "hash-2"


def test_route_from_resolved_user_intent_meta_routes_to_rag_db_qa_once() -> None:
    state = _state_with_cancelled_intent()
    intent_id = state["memory"]["last_user_intent_id"]
    card = state["memory"]["user_intents"][intent_id]
    meta = _apply_resolved_user_intent_meta({}, {"relationship": "continue"}, card, "hash-2")

    action, updated_meta, observations = _route_from_resolved_user_intent_meta(
        state,
        meta,
        {"rag_db_qa"},
    )

    assert action == "rag_db_qa"
    assert updated_meta[MetaKeys.RAG_DB_QUESTION_OVERRIDE] == "Query my database for age"
    assert updated_meta["resolved_user_intent_meta_consumed"] == "hash-2"
    assert observations == [f"intent_id={intent_id} relationship=continue routed_node=rag_db_qa"]


def test_consumed_resolved_user_intent_meta_clears_on_next_cycle() -> None:
    meta = {
        MetaKeys.RESOLVED_USER_INTENT_ID: "intent_1",
        MetaKeys.RESOLVED_USER_INTENT_USER_MESSAGE_HASH: "hash-1",
        "resolved_user_intent_meta_consumed": "hash-1",
    }

    assert _clear_consumed_resolved_user_intent_meta(meta) == {}
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
pytest tests/test_orchestrator_user_intent_resolution.py -q
```

Expected: fail with missing meta keys and helper functions.

- [ ] **Step 3: Add MetaKeys constants**

In `graph/state.py`, add:

```python
    RESOLVED_USER_INTENT_ID = "resolved_user_intent_id"
    RESOLVED_USER_INTENT_KIND = "resolved_user_intent_kind"
    RESOLVED_USER_INTENT_RELATIONSHIP = "resolved_user_intent_relationship"
    RESOLVED_USER_INTENT_SOURCE_QUESTION = "resolved_user_intent_source_question"
    RESOLVED_USER_INTENT_USER_MESSAGE_HASH = "resolved_user_intent_user_message_hash"
    RAG_DB_QUESTION_OVERRIDE = "rag_db_question_override"
```

- [ ] **Step 4: Add orchestrator helper functions**

In `graph/nodes/orchestrator/node.py`, add after `_RESOLVED_TASK_META_CONSUMED_KEY`:

```python
_RESOLVED_USER_INTENT_META_KEYS = (
    MetaKeys.RESOLVED_USER_INTENT_ID,
    MetaKeys.RESOLVED_USER_INTENT_KIND,
    MetaKeys.RESOLVED_USER_INTENT_RELATIONSHIP,
    MetaKeys.RESOLVED_USER_INTENT_SOURCE_QUESTION,
    MetaKeys.RESOLVED_USER_INTENT_USER_MESSAGE_HASH,
)
_RESOLVED_USER_INTENT_META_CONSUMED_KEY = "resolved_user_intent_meta_consumed"
```

Add helpers:

```python
def _clear_resolved_user_intent_meta(meta: dict) -> dict:
    updated = dict(meta)
    for key in _RESOLVED_USER_INTENT_META_KEYS:
        updated.pop(key, None)
    updated.pop(_RESOLVED_USER_INTENT_META_CONSUMED_KEY, None)
    updated.pop(MetaKeys.RAG_DB_QUESTION_OVERRIDE, None)
    return updated


def _clear_consumed_resolved_user_intent_meta(meta: dict) -> dict:
    resolved_hash = meta.get(MetaKeys.RESOLVED_USER_INTENT_USER_MESSAGE_HASH)
    consumed_hash = meta.get(_RESOLVED_USER_INTENT_META_CONSUMED_KEY)
    if isinstance(resolved_hash, str) and consumed_hash == resolved_hash:
        return _clear_resolved_user_intent_meta(meta)
    return meta


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
    return updated


def _route_from_resolved_user_intent_meta(
    routing_state: AgentState,
    meta: dict,
    available_action_set: set[str],
) -> tuple[str | None, dict, list[str]]:
    intent_id = meta.get(MetaKeys.RESOLVED_USER_INTENT_ID)
    relationship = meta.get(MetaKeys.RESOLVED_USER_INTENT_RELATIONSHIP)
    source_question = meta.get(MetaKeys.RESOLVED_USER_INTENT_SOURCE_QUESTION)
    resolved_hash = meta.get(MetaKeys.RESOLVED_USER_INTENT_USER_MESSAGE_HASH)
    consumed_hash = meta.get(_RESOLVED_USER_INTENT_META_CONSUMED_KEY)
    if isinstance(resolved_hash, str) and consumed_hash == resolved_hash:
        return None, _clear_resolved_user_intent_meta(meta), []
    if (
        not isinstance(intent_id, str)
        or relationship not in {"continue", "refine"}
        or not isinstance(source_question, str)
        or not source_question.strip()
    ):
        return None, meta, []
    _state, memory = ensure_memory_state(routing_state)
    intent = dict(memory.get("user_intents") or {}).get(intent_id)
    if not isinstance(intent, dict) or intent.get("kind") != "db_rag_query":
        return None, _clear_resolved_user_intent_meta(meta), []
    if "rag_db_qa" not in available_action_set:
        return None, _clear_resolved_user_intent_meta(meta), []
    updated_meta = dict(meta)
    updated_meta[MetaKeys.RAG_DB_QUESTION_OVERRIDE] = source_question.strip()
    if isinstance(resolved_hash, str):
        updated_meta[_RESOLVED_USER_INTENT_META_CONSUMED_KEY] = resolved_hash
    return (
        "rag_db_qa",
        updated_meta,
        [f"intent_id={intent_id} relationship={relationship} routed_node=rag_db_qa"],
    )
```

- [ ] **Step 5: Run helper tests**

Run:

```bash
pytest tests/test_orchestrator_user_intent_resolution.py -q
```

Expected: helper tests pass.

- [ ] **Step 6: Integrate helpers into orchestrator flow**

In `graph/nodes/orchestrator/node.py`:

Import user-intent classifier helpers:

```python
from ...memory import (
    classify_user_intent_reference,
    compact_user_intent_cards,
    ensure_memory_state,
    resolve_reference_for_turn,
)
```

When fresh user turn clears resolved task meta, also clear resolved user-intent meta:

```python
meta = _clear_resolved_task_meta(meta)
meta = _clear_resolved_user_intent_meta(meta)
```

When not a fresh unanswered turn, clear consumed user-intent meta:

```python
meta = _clear_consumed_resolved_task_meta(meta)
meta = _clear_consumed_resolved_user_intent_meta(meta)
```

After `_route_from_resolved_task_meta(...)` and before deterministic readiness, add `_route_from_resolved_user_intent_meta(...)`.

Add classifier invocation only when:

```python
not next_action
and fresh_unanswered_user_turn
and dict(memory.get("user_intents") or {})
```

The implementation should call `classify_user_intent_reference(...)`, then:

- if result `target == "existing_user_intent"` and `needs_clarification is False`, apply resolved user-intent meta
- if result `target == "ambiguous"` or `needs_clarification is True`, set clarification meta and output a short clarification question
- if result `target == "new_user_intent"`, do nothing and let normal routing proceed
- if result `target == "completed_task"`, let existing completed-task resolver/planner path handle it; do not set user-intent meta

- [ ] **Step 7: Thread question override into DB-RAG call site**

Find the place in graph construction or node invocation where `rag_db_qa_node` receives `question_override`. If it already reads `question_override` from caller, update orchestrator dispatch to pass `meta[MetaKeys.RAG_DB_QUESTION_OVERRIDE]` when routing to `rag_db_qa`. If dispatch is registry-owned, add this to the node wrapper that calls `rag_db_qa_node`.

The expected behavior is equivalent to:

```python
question_override = (state.get("meta") or {}).get(MetaKeys.RAG_DB_QUESTION_OVERRIDE)
updated = rag_db_qa_node(state, llm, provider=provider, service=service, question_override=question_override)
```

After DB-RAG consumes the override, clear `MetaKeys.RAG_DB_QUESTION_OVERRIDE` from meta.

- [ ] **Step 8: Add integration tests for orchestrator flow**

Extend `tests/test_orchestrator_user_intent_resolution.py` with an integration test following existing patterns from `tests/test_orchestrator_memory_resolution.py`:

```python
def test_orchestrator_routes_validated_user_intent_to_rag_db(monkeypatch) -> None:
    # Build state with an unanswered human message "continue previous query"
    # and a cancelled db_rag_query user intent.
    # Monkeypatch classify_user_intent_reference to return existing_user_intent.
    # Assert next_action == "rag_db_qa" and meta RAG_DB_QUESTION_OVERRIDE is original source_question.
```

Use the same state construction helpers and monkeypatch style already used in `tests/test_orchestrator_memory_resolution.py`.

- [ ] **Step 9: Run orchestrator tests**

Run:

```bash
pytest tests/test_orchestrator_user_intent_resolution.py tests/test_orchestrator_memory_resolution.py -q
```

Expected: all selected tests pass.

- [ ] **Step 10: Commit**

```bash
git add graph/state.py graph/nodes/orchestrator/node.py tests/test_orchestrator_user_intent_resolution.py
git commit -m "Route validated user intent references"
```

---

### Task 5: Add Planner Context For User Intents

**Files:**
- Modify: `graph/nodes/orchestrator/context_builder.py`
- Test: `tests/test_orchestrator_planner_environment.py`

- [ ] **Step 1: Write failing planner environment test**

Add to `tests/test_orchestrator_planner_environment.py`:

```python
from graph.memory import upsert_user_intent_from_db_rag_intent
from graph.nodes.orchestrator.context_builder import build_planner_environment


def test_planner_environment_includes_compact_user_intents() -> None:
    state = _state()
    state = upsert_user_intent_from_db_rag_intent(
        state,
        active_intent={
            "intent_id": "rag-1",
            "source_question": "Query my database for age",
            "goal_text": "Query age among index cases",
        },
        source_message_hash="hash-1",
        status="cancelled",
    )

    env = build_planner_environment(state, ["rag_db_qa", "qa"])

    assert env["candidate_user_intents"] == [
        {
            "intent_id": state["memory"]["last_user_intent_id"],
            "display_ordinal": 1,
            "kind": "db_rag_query",
            "source_question": "Query my database for age",
            "goal_text": "Query age among index cases",
            "status": "cancelled",
            "created_at": state["memory"]["user_intents"][state["memory"]["last_user_intent_id"]]["created_at"],
            "completed_task_id": None,
        }
    ]
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
pytest tests/test_orchestrator_planner_environment.py::test_planner_environment_includes_compact_user_intents -q
```

Expected: fail because `candidate_user_intents` is missing.

- [ ] **Step 3: Add compact planner cards**

In `graph/nodes/orchestrator/context_builder.py`, import:

```python
from ...memory import compact_user_intent_cards, latest_task_cards
```

Add:

```python
MAX_PLANNER_ENV_USER_INTENTS = 8


def _compact_user_intent_cards(state: dict) -> list[dict[str, object]]:
    cards = compact_user_intent_cards(
        state,
        kind="db_rag_query",
        limit=MAX_PLANNER_ENV_USER_INTENTS,
    )
    compacted: list[dict[str, object]] = []
    for card in cards:
        compacted.append(
            {
                "intent_id": card.get("intent_id"),
                "display_ordinal": card.get("display_ordinal"),
                "kind": card.get("kind"),
                "source_question": card.get("source_question"),
                "goal_text": card.get("goal_text"),
                "status": card.get("status"),
                "created_at": card.get("created_at"),
                "completed_task_id": card.get("completed_task_id"),
            }
        )
    return compacted
```

In `build_planner_environment`, add:

```python
        "candidate_user_intents": _compact_user_intent_cards(state),
        "validated_reference": _validated_reference_from_meta(state),
```

Add `_validated_reference_from_meta`:

```python
def _validated_reference_from_meta(state: dict) -> dict[str, object] | None:
    meta = dict(state.get("meta") or {})
    target_id = meta.get(MetaKeys.RESOLVED_USER_INTENT_ID)
    relationship = meta.get(MetaKeys.RESOLVED_USER_INTENT_RELATIONSHIP)
    if isinstance(target_id, str) and isinstance(relationship, str):
        return {
            "target": "existing_user_intent",
            "target_id": target_id,
            "relationship": relationship,
        }
    task_id = meta.get(MetaKeys.RESOLVED_TASK_ID)
    task_relationship = meta.get(MetaKeys.RESOLVED_TASK_RELATIONSHIP)
    if isinstance(task_id, str) and isinstance(task_relationship, str):
        return {
            "target": "completed_task",
            "target_id": task_id,
            "relationship": task_relationship,
        }
    return None
```

- [ ] **Step 4: Run planner environment tests**

Run:

```bash
pytest tests/test_orchestrator_planner_environment.py -q
```

Expected: all planner environment tests pass.

- [ ] **Step 5: Commit**

```bash
git add graph/nodes/orchestrator/context_builder.py tests/test_orchestrator_planner_environment.py
git commit -m "Expose user intents to planner context"
```

---

### Task 6: Wire DB-RAG User Intent Write Points

**Files:**
- Modify: `graph/nodes/db_rag_qa/node.py`
- Test: `tests/test_rag_db_qa_node.py`

- [ ] **Step 1: Add failing DB-RAG intent write tests**

Add tests to `tests/test_rag_db_qa_node.py` near existing active-intent tests:

```python
def test_fresh_db_rag_question_writes_user_intent_memory(monkeypatch) -> None:
    state = _state("Which forms contain age?")
    service = _Service()

    updated = rag_db_qa_node(state, object(), provider="openai", service=service)

    memory = updated["memory"]
    intent_id = memory["last_user_intent_id_by_kind"]["db_rag_query"]
    card = memory["user_intents"][intent_id]
    assert card["source_question"] == "Which forms contain age?"
    assert card["kind"] == "db_rag_query"
    assert card["status"] in {"awaiting_extraction_opt_in", "awaiting_column_review"}
    assert card["active_intent_id"] == updated["agents"]["rag_db_qa"]["active_intent"]["intent_id"]


def test_question_override_creates_continued_user_intent(monkeypatch) -> None:
    state = _state("continue previous query")
    service = _Service()

    updated = rag_db_qa_node(
        state,
        object(),
        provider="openai",
        service=service,
        question_override="Query my database for age",
    )

    card = updated["memory"]["user_intents"][updated["memory"]["last_user_intent_id"]]
    assert card["source_question"] == "Query my database for age"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
pytest tests/test_rag_db_qa_node.py -q
```

Expected: new assertions fail because DB-RAG does not write user-intent memory.

- [ ] **Step 3: Import helper and upsert after active intent creation**

In `graph/nodes/db_rag_qa/node.py`, import:

```python
from ...memory import upsert_user_intent_from_db_rag_intent
```

In `_handle_fresh_db_rag_question`, after `_reset_active_workflow_for_new_question(...)` and before starting column review or metadata answer, add:

```python
    source_hash = str((state.get("meta") or {}).get(MetaKeys.LAST_USER_MESSAGE_HASH) or "").strip() or None
    intent_status = "awaiting_column_review" if explicit_extraction else "awaiting_extraction_opt_in"
    state = upsert_user_intent_from_db_rag_intent(
        state,
        active_intent=intent,
        source_message_hash=source_hash,
        status=intent_status,
        force_new=bool(question_override),
    )
```

Because `_handle_fresh_db_rag_question` does not currently receive `question_override`, pass a `force_new_intent: bool = False` parameter from `_rag_db_qa_node_impl`:

```python
return _handle_fresh_db_rag_question(
    state,
    rag_state,
    service=service,
    reranker_model=reranker_model,
    question=question,
    force_new_intent=bool(question_override),
)
```

Thread `force_new_intent` through the function signature and use it for `force_new`.

- [ ] **Step 4: Upsert on pending extraction reply/refinement**

In `_handle_pending_extraction_reply`, after `active_intent["mode"] = "extraction"` in the `"yes"` branch:

```python
        state = upsert_user_intent_from_db_rag_intent(
            state,
            active_intent=active_intent,
            source_message_hash=str((state.get("meta") or {}).get(MetaKeys.LAST_USER_MESSAGE_HASH) or "").strip() or None,
            status="awaiting_column_review",
        )
```

In the `"no"` branch, after `active_intent["status"] = "extraction_declined"`:

```python
            state = upsert_user_intent_from_db_rag_intent(
                state,
                active_intent=active_intent,
                source_message_hash=str((state.get("meta") or {}).get(MetaKeys.LAST_USER_MESSAGE_HASH) or "").strip() or None,
                status="declined",
            )
```

In the `"substantive_followup"` branch, upsert after resolving `intent` and before `_start_column_review(...)` with status `"awaiting_column_review"`.

- [ ] **Step 5: Clear question override after DB-RAG consumes it**

At the start of `_rag_db_qa_node_impl`, after computing `question`, clear the override key from state meta if present:

```python
    if question_override is not None:
        meta = dict(state.get("meta") or {})
        meta.pop(MetaKeys.RAG_DB_QUESTION_OVERRIDE, None)
        state = {**state, "meta": meta}
```

- [ ] **Step 6: Run DB-RAG tests**

Run:

```bash
pytest tests/test_rag_db_qa_node.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

```bash
git add graph/nodes/db_rag_qa/node.py tests/test_rag_db_qa_node.py
git commit -m "Record DB-RAG user intents"
```

---

### Task 7: Update DB-RAG Review Cancel And SQL Completion Linking

**Files:**
- Modify: `graph/nodes/human_review_rag_db_column_selection.py`
- Modify: `graph/nodes/human_review_rag_db_sql_execution.py`
- Modify: `graph/nodes/db_rag_qa/helpers.py`
- Test: `tests/test_review_control_events.py`
- Test: `tests/test_rag_db_qa_node.py`

- [ ] **Step 1: Add failing cancel status tests**

Extend DB-RAG cancel tests in `tests/test_review_control_events.py`:

```python
def test_rag_db_column_review_cancel_marks_user_intent_cancelled(monkeypatch) -> None:
    _install_stubs(decision="cancel", suggestion="")
    for mod in (
        "graph.nodes.human_review_rag_db_column_selection",
        "graph.nodes.human_review_cancel",
        "graph.state",
        "graph.nodes.state_helpers",
    ):
        sys.modules.pop(mod, None)
    module = importlib.import_module("graph.nodes.human_review_rag_db_column_selection")
    state = {
        "messages": [],
        "output": {},
        "meta": {"last_user_message_hash": "u-rag-col-cancel-intent"},
        "artifacts": {
            "files": {
                "sel-1": {
                    "kind": "db_rag_column_selection",
                    "artifact_id": "sel-1",
                    "content": {
                        "status": "awaiting_review",
                        "goal_text": "extract diabetes rows",
                        "source_question": "extract diabetes rows",
                    },
                }
            }
        },
        "agents": {
            "rag_db_qa": {
                "active_intent": {
                    "intent_id": "rag-intent-1",
                    "source_question": "extract diabetes rows",
                    "goal_text": "extract diabetes rows",
                },
                "pending_column_review_artifact_id": "sel-1",
                "pending_column_review": {"status": "awaiting_review"},
                "thread_status": "awaiting_column_review",
                "active_thread": True,
            }
        },
    }
    state = upsert_user_intent_from_db_rag_intent(
        state,
        active_intent=state["agents"]["rag_db_qa"]["active_intent"],
        source_message_hash="hash-1",
        status="awaiting_column_review",
    )
    intent_id = state["memory"]["last_user_intent_id"]

    updated = module.human_review_rag_db_column_selection_node(state)

    assert updated["memory"]["user_intents"][intent_id]["status"] == "cancelled"
    assert updated["memory"]["completed_tasks"] == {}
```

Add equivalent SQL review cancel test for `human_review_rag_db_sql_execution_node`.

- [ ] **Step 2: Add failing SQL completion link test**

Extend `tests/test_rag_db_qa_node.py` near SQL execution completion tests:

```python
def test_sql_execution_completion_links_originating_user_intent(monkeypatch) -> None:
    state, rag_state, candidate = _successful_sql_execution_state()
    state = upsert_user_intent_from_db_rag_intent(
        state,
        active_intent={
            "intent_id": "intent:sql",
            "source_question": "Generate the SQL to subset index cases with diabetes.",
            "goal_text": "Generate the SQL to subset index cases with diabetes.",
        },
        source_message_hash="hash-1",
        status="awaiting_sql_review",
    )
    intent_id = state["memory"]["last_user_intent_id"]

    updated = _execute_successful_sql(monkeypatch, state, rag_state, candidate)

    task_id = updated["memory"]["last_task_id_by_kind"]["db_rag_sql_extraction"]
    assert updated["memory"]["user_intents"][intent_id]["completed_task_id"] == task_id
    assert updated["memory"]["user_intents"][intent_id]["status"] == "completed"
    assert (
        updated["memory"]["completed_tasks"][task_id]["provenance"]["originating_user_intent_id"]
        == intent_id
    )
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
pytest tests/test_review_control_events.py tests/test_rag_db_qa_node.py -q
```

Expected: new assertions fail.

- [ ] **Step 4: Mark cancel status in review nodes**

In both DB-RAG review nodes, import:

```python
from ..memory import update_user_intent_status
```

Use correct relative import from node package. If direct relative path is `from ..memory` invalid from `graph/nodes`, use `from graph.memory import update_user_intent_status`.

In the cancel branch, before `return update_agent_state(...)`, add:

```python
        active_intent = dict(rag_state.get("active_intent") or {})
        updated_state = update_user_intent_status(
            updated_state,
            active_intent_id=str(active_intent.get("intent_id") or ""),
            status="cancelled",
        )
```

Do not create a new intent if no match exists.

- [ ] **Step 5: Link SQL completion to originating user intent**

In `graph/nodes/db_rag_qa/helpers.py`, import:

```python
from ...memory import latest_user_intent, link_user_intent_completed_task
```

Before `complete_task(...)`, compute:

```python
    originating_intent = latest_user_intent(updated, kind="db_rag_query")
    originating_user_intent_id = (
        originating_intent.get("intent_id")
        if isinstance(originating_intent, dict)
        and originating_intent.get("active_intent_id") == str(_read_value(candidate, "intent_id", "") or "")
        else None
    )
```

If candidate does not expose `intent_id`, use `sql_candidate_artifact.get("intent_snapshot", {}).get("intent_id")` or `approved_selection.get("intent_snapshot", {}).get("intent_id")` for matching.

Extend provenance:

```python
        provenance={
            "producer_node": "rag_db_qa",
            "selection_id": str(getattr(candidate, "selection_id", "") or ""),
            **(
                {"originating_user_intent_id": originating_user_intent_id}
                if originating_user_intent_id
                else {}
            ),
        },
```

After `complete_task(...)`, get the new task ID and link:

```python
    if originating_user_intent_id:
        task_id = updated["memory"]["last_task_id_by_kind"].get("db_rag_sql_extraction")
        if isinstance(task_id, str):
            updated = link_user_intent_completed_task(
                updated,
                intent_id=originating_user_intent_id,
                task_id=task_id,
            )
```

- [ ] **Step 6: Run tests**

Run:

```bash
pytest tests/test_review_control_events.py tests/test_rag_db_qa_node.py -q
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

```bash
git add graph/nodes/human_review_rag_db_column_selection.py graph/nodes/human_review_rag_db_sql_execution.py graph/nodes/db_rag_qa/helpers.py tests/test_review_control_events.py tests/test_rag_db_qa_node.py
git commit -m "Link DB-RAG intents through cancel and completion"
```

---

### Task 8: Full Verification And Spec Alignment

**Files:**
- Modify only if verification exposes a mismatch.

- [ ] **Step 1: Run targeted test suite**

Run:

```bash
pytest tests/test_user_intent_memory.py tests/test_user_intent_reference_classifier.py tests/test_orchestrator_user_intent_resolution.py tests/test_orchestrator_planner_environment.py tests/test_review_control_events.py tests/test_rag_db_qa_node.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run full test suite**

Run:

```bash
pytest -q
```

Expected: all tests pass. Existing protobuf deprecation warnings are acceptable if unchanged.

- [ ] **Step 3: Review runtime-canonical gating policy**

Run:

```bash
sed -n '1,140p' docs/superpowers/specs/orchestrator-gating-policy.md
```

Expected: no policy change is required. The new behavior is reference resolution before planner fallback, not a new planner fallback rule.

- [ ] **Step 4: Check worktree**

Run:

```bash
git status --short
```

Expected: no uncommitted files except intentionally ignored local artifacts.

- [ ] **Step 5: Commit any final fixes**

If Step 1 or Step 2 required fixes:

```bash
git add graph/memory/schema.py graph/memory/task_store.py graph/memory/user_intent_store.py graph/memory/__init__.py graph/state.py graph/nodes/orchestrator/node.py graph/nodes/orchestrator/context_builder.py graph/nodes/db_rag_qa/node.py graph/nodes/db_rag_qa/helpers.py graph/nodes/human_review_rag_db_column_selection.py graph/nodes/human_review_rag_db_sql_execution.py tests/test_user_intent_memory.py tests/test_task_memory.py tests/test_user_intent_reference_classifier.py tests/test_orchestrator_user_intent_resolution.py tests/test_orchestrator_planner_environment.py tests/test_review_control_events.py tests/test_rag_db_qa_node.py
git commit -m "Stabilize user intent history integration"
```

If no fixes were needed, do not create an empty commit.
