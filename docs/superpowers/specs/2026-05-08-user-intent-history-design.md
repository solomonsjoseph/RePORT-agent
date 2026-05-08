# User Intent History Design

Date: 2026-05-08
Status: Proposed

## Goal

Add a small current-thread intent history so the system can resolve references such as "continue the previous query", "retry that database query", and future UI actions such as edit or copy on prior user messages.

The first implementation slice should focus on DB-RAG query intents, especially cancelled or incomplete extraction workflows. Cancel should stop the pending workflow, but it should not make the original user query disappear as a referenceable intent.

## Problem

The app currently has three related but different state surfaces:

- `planner["memory"]`
  - concise orchestrator routing memory
  - includes `active_user_goal`, `latest_user_update`, `conversation_intent_summary`, and `unresolved_user_constraints`
  - rolling and summary-like, not a stable timestamped registry

- `agents["rag_db_qa"]["active_intent"]`
  - DB-RAG-specific live executable intent
  - includes `intent_id`, `source_question`, `goal_text`, status, requested fields, filters, and review feedback
  - suitable for the current workflow, but not sufficient as a durable address book after cancel or workflow cleanup

- `memory.completed_tasks`
  - stable registry for completed reusable results
  - correct for completed SQL extraction, QA answers, and code analysis results
  - not appropriate for cancelled or incomplete user query attempts

This separation is mostly correct. The missing layer is a stable list of user-initiated intents that may or may not have completed.

The failure mode is:

1. User asks: "Query my database, help me to subset age, gender, diabetes status, and TB outcome among index case."
2. DB-RAG enters column review.
3. User clicks Cancel.
4. User says: "continue previous query."

The cancelled review should not resume automatically. However, "previous query" should resolve to the latest timestamped user DB-RAG intent and start a fresh DB-RAG workflow from that query.

## Design Principles

1. Keep planner memory concise and routing-facing.
2. Keep DB-RAG `active_intent` as the canonical live DB-RAG semantic object.
3. Keep `memory.completed_tasks` limited to completed reusable outputs.
4. Add only a compact current-thread user intent registry.
5. Resolve "previous query" by structured timestamps, not by planner summary text.
6. Treat cancel as terminal for the pending workflow but reference-preserving for the originating intent.
7. Do not let cancelled intent history suppress future routing.
8. Avoid making an LLM responsible for task lifecycle truth.

## State Model

Add `user_intents` and ordering metadata under top-level `memory`.

Suggested shape:

```python
memory = {
    "user_intents": {
        "intent_3f8a9c21": {
            "intent_id": "intent_3f8a9c21",
            "display_ordinal": 4,
            "kind": "db_rag_query",
            "agent": "rag_db_qa",
            "source_question": "Query my database, help me to subset age, gender, diabetes status, and TB outcome among index case.",
            "goal_text": "Subset age, gender, diabetes status, and TB outcome among index cases.",
            "status": "cancelled",
            "source_event_id": "evt_...",
            "source_message_hash": "...",
            "active_intent_id": "intent:...",
            "artifact_refs": {
                "selection_artifact_id": "sel-art-..."
            },
            "review_refs": {
                "cancel_event_id": "evt_..."
            },
            "completed_task_id": None,
            "continued_from_intent_id": None,
            "created_at": "2026-05-08T00:00:00+00:00",
            "updated_at": "2026-05-08T00:00:00+00:00"
        }
    },
    "intent_order": ["intent_3f8a9c21"],
    "last_user_intent_id": "intent_3f8a9c21",
    "last_user_intent_id_by_kind": {
        "db_rag_query": "intent_3f8a9c21"
    }
}
```

`intent_id` should be an opaque memory-layer ID. It may link to DB-RAG's `active_intent.intent_id` through `active_intent_id`, but the two IDs should not be required to match.

## Status Semantics

Allowed initial statuses:

- `active`
  - user intent has a live workflow in progress

- `awaiting_review`
  - workflow is paused at a human review node

- `cancelled`
  - user cancelled the pending workflow
  - this status is referenceable but not executable by itself

- `completed`
  - workflow produced a reusable result and has a linked `completed_task_id`

- `superseded`
  - a newer fresh user intent replaced this one as the active DB-RAG query

- `declined`
  - user declined optional extraction for a metadata question

Cancel does not mean failed. A cancelled intent is a user query whose current workflow was intentionally stopped.

## Layer Boundaries

### `planner["memory"]`

Planner memory remains the current concise routing summary.

It may mention the current user goal, but it should not become a historical intent list.

### `agents["rag_db_qa"]["active_intent"]`

DB-RAG `active_intent` remains the live executable semantic object.

DB-RAG continues to own extraction gates, column review, SQL review, regeneration feedback, and SQL execution state.

### `memory.user_intents`

User intent history owns stable current-thread references to user-initiated goals, including incomplete and cancelled goals.

It stores compact metadata only. It must not embed SQL, long schema context, dataframe contents, or full review payloads.

### `memory.completed_tasks`

Completed task memory remains the registry for completed reusable outputs.

When a DB-RAG SQL extraction completes, the completed task should link back to the originating user intent, and the originating user intent should link forward to `completed_task_id`.

## Write Points

### Fresh DB-RAG Question

When DB-RAG creates a fresh `active_intent`, it should upsert a `db_rag_query` record in `memory.user_intents`.

The record should include:

- `source_question`
- `goal_text`
- `kind="db_rag_query"`
- `agent="rag_db_qa"`
- `status="active"`
- `source_event_id` when available
- `source_message_hash`
- `active_intent_id`
- `created_at`
- `updated_at`

This should happen for both metadata-style DB-RAG questions and extraction-oriented DB-RAG questions.

### DB-RAG Intent Refinement

When a user refinement updates the current DB-RAG `active_intent`, the system should update the same user-intent record rather than create a new one, unless the boundary classifier has determined that the message is a fresh new question.

The record should update:

- `goal_text`
- `status`
- `updated_at`
- compact artifact or review references when created

The original `source_question` should remain the initiating user message. Later refinement text can be represented by review feedback, event refs, or a future compact `revision_history` if needed.

### Human Review Cancel

When a DB-RAG column-selection or SQL-execution review is cancelled, the system should:

- keep the existing cancel behavior that clears live workflow pointers
- mark the related `memory.user_intents[...]` record `status="cancelled"`
- set `updated_at`
- record the cancel event reference when available
- avoid creating or mutating `memory.completed_tasks`

Cancel should not add planner suppression rules. Future DB-RAG requests should route normally.

### SQL Execution Completion

When reviewed DB-RAG SQL execution succeeds and writes a completed task:

- set the originating user intent `status="completed"`
- set `completed_task_id`
- set `updated_at`
- keep the completed task card as the reusable result record

The completed task remains the source for result-focused references such as "summarize the output" or "what SQL did you use?"

## Reference Resolution

The current task-memory resolver handles completed tasks. User-intent history adds an earlier reference surface for incomplete or cancelled user goals.

Recommended resolution order after higher-priority live workflows:

1. Pending human review or interrupt
2. Pending clarification or DB-RAG opt-in
3. Pending deterministic workflow continuation
4. User-intent reference resolver for incomplete/cancelled user goals
5. Completed-task reference resolver
6. Normal deterministic routing predicates
7. Planner fallback

The user-intent resolver should be deterministic for common references:

- "previous query"
- "last query"
- "continue previous query"
- "continue that database query"
- "retry that DB query"

For these phrases, choose the latest `memory.user_intents` record by `created_at`, scoped to `kind="db_rag_query"` when the phrase says query or database.

If the user says "previous result", "last output", or asks to inspect SQL/dataset/output, use completed task memory instead.

## Continue Behavior

When the user says "continue previous query" and the latest DB-RAG user intent is `cancelled`, the system should not resume the cancelled review node.

Instead it should:

1. resolve the latest DB-RAG `memory.user_intents` record
2. create a new live DB-RAG `active_intent` from that record's `source_question` and `goal_text`
3. create or update a new user-intent record with `continued_from_intent_id` pointing to the prior record
4. route into DB-RAG as a fresh workflow

This preserves the user's intended query while respecting that Cancel ended the old pending review.

## Future UI Fit

This design supports future modern chat actions without requiring a separate model later:

- Copy user query: read `source_question`
- Edit and resubmit: create a new user intent with `continued_from_intent_id` or `edited_from_intent_id`
- Retry cancelled query: create a new active DB-RAG intent linked to the cancelled intent
- Show intent history: render compact records ordered by `intent_order`

No UI work is required in the first implementation slice.

## Testing

Add focused tests for:

- memory initialization preserves existing `completed_tasks` behavior
- fresh DB-RAG question creates a `memory.user_intents` record
- DB-RAG active-intent update upserts the same user-intent record
- DB-RAG column-review cancel marks the originating user intent `cancelled`
- DB-RAG SQL-review cancel marks the originating user intent `cancelled`
- cancel still does not create or mutate `memory.completed_tasks`
- "continue previous query" resolves the latest timestamped DB-RAG user intent
- continuing a cancelled query starts a fresh DB-RAG workflow instead of reopening the cancelled review
- completed SQL extraction links completed task memory and originating user-intent memory

Existing tests for approve, regenerate, cancel audit events, and completed-task memory should continue to pass.

## Non-Goals

- No cross-session memory.
- No long-term preference learning from cancelled reviews.
- No automatic routing suppression because a user cancelled once.
- No full transcript storage inside `memory.user_intents`.
- No migration of planner memory into user-intent history.
- No replacement of `agents["rag_db_qa"]["active_intent"]`.
- No UI edit/copy implementation in the first slice.

## Implementation Notes

The helper code should live under `graph/memory/`, next to task memory helpers, because this is a generic current-thread memory layer even though the first producer is DB-RAG.

DB-RAG should call the helper at its natural intent write points rather than duplicating memory mutation logic in review nodes. Review nodes may call a small status-update helper on cancel if they have the originating active intent ID.

The reference resolver should keep exact phrase handling deterministic first. An LLM resolver can be added later only if deterministic intent-reference handling proves insufficient.
