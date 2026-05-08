# User Intent History Design

Date: 2026-05-08
Status: Proposed

## Goal

Add a small current-thread intent history so the system can resolve references such as "continue the previous query", "retry that database query", and future UI actions such as edit or copy on prior user messages.

The first implementation slice should focus on DB-RAG query intents, especially cancelled or incomplete extraction workflows. Cancel should stop the pending workflow, but it should not make the original user query disappear as a referenceable intent.

The implementation must stay lean: this is not a second task-memory system and not a second LLM resolver. It is a compact address book for user-originated DB-RAG intents.

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

The cancelled review should not resume automatically. However, "previous query" should resolve to the latest user-originated DB-RAG intent and start a fresh DB-RAG workflow from that query.

## Design Principles

1. Keep planner memory concise and routing-facing.
2. Keep DB-RAG `active_intent` as the canonical live DB-RAG semantic object.
3. Keep `memory.completed_tasks` limited to completed reusable outputs.
4. Add only a compact current-thread user intent registry.
5. Resolve "previous query" by structured intent pointers and creation order, not by planner summary text.
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
            "source_message_hash": "...",
            "active_intent_id": "intent:...",
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

`intent_order` is creation order for user-initiated intent records. Status-only updates such as review cancel must update `updated_at` but must not move an older record to the end of `intent_order`. This keeps "latest query" tied to the latest user-originated query, not the latest internal status mutation.

## Status Semantics

Allowed initial statuses:

- `active`
  - user intent has a live workflow in progress

- `awaiting_extraction_opt_in`
  - DB-RAG answered a metadata question and is waiting for the extraction opt-in reply

- `awaiting_column_review`
  - DB-RAG is paused at the column-selection review node

- `awaiting_sql_review`
  - DB-RAG is paused at the SQL-execution review node

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

## Schema And Helpers

Extend the existing memory helpers rather than creating a second memory normalization path.

`graph/memory/schema.py` should add a compact `UserIntentCard` shape and allowed user-intent constants:

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
```

`graph/memory/task_store.py` should extend `EMPTY_MEMORY_STATE`, `MEMORY_KEY_TYPES`, and `ensure_memory_state` with:

- `user_intents: dict`
- `intent_order: list`
- `last_user_intent_id: str | None`
- `last_user_intent_id_by_kind: dict[str, str]`

`ensure_memory_state` should prune `intent_order`, `last_user_intent_id`, and `last_user_intent_id_by_kind` against existing `user_intents`, mirroring the completed-task cleanup pattern.

Add helper functions in `graph/memory/user_intent_store.py` or the existing memory package:

- `upsert_user_intent_from_db_rag_intent(...)`
- `update_user_intent_status(...)`
- `link_user_intent_completed_task(...)`
- `latest_user_intent(...)`

All helper inputs must be JSON-safe and should deep-copy caller-provided dict/list values before storing them.

Do not add a separate cached user-intent resolver in the first slice. If a deterministic helper is useful for "continue previous query", keep it as a small phrase matcher near the existing reference-resolution path and do not introduce LLM prompting, candidate ranking, ambiguity handling, or a second resolution cache.

## Write Points

### Fresh DB-RAG Question

When DB-RAG creates a fresh `active_intent`, it should upsert a `db_rag_query` record in `memory.user_intents`.

The record should include:

- `source_question`
- `goal_text`
- `kind="db_rag_query"`
- `agent="rag_db_qa"`
- `status="active"`
- `source_message_hash`
- `active_intent_id`
- `created_at`
- `updated_at`

This should happen for both metadata-style DB-RAG questions and extraction-oriented DB-RAG questions.

If there is an older DB-RAG user intent in `active`, `awaiting_extraction_opt_in`, `awaiting_column_review`, or `awaiting_sql_review` status and the DB-RAG boundary classifier has determined the new message is a fresh question, mark the older intent `superseded` before creating the new record.

### DB-RAG Intent Refinement

When a user refinement updates the current DB-RAG `active_intent`, the system should update the same user-intent record rather than create a new one, unless the boundary classifier has determined that the message is a fresh new question.

The upsert lookup order should be:

1. existing `memory.user_intents` record whose `active_intent_id` matches DB-RAG `active_intent.intent_id`
2. `last_user_intent_id_by_kind["db_rag_query"]` when that record is still live and belongs to `rag_db_qa`
3. create a new record

The record should update:

- `goal_text`
- `status`
- `updated_at`

The original `source_question` should remain the initiating user message. Later refinement text can be represented by review feedback, event refs, or a future compact `revision_history` if needed.

### Human Review Cancel

When a DB-RAG column-selection or SQL-execution review is cancelled, the system should:

- keep the existing cancel behavior that clears live workflow pointers
- mark the related `memory.user_intents[...]` record `status="cancelled"`
- set `updated_at`
- avoid creating or mutating `memory.completed_tasks`

Cancel should not add planner suppression rules. Future DB-RAG requests should route normally.

The cancel status update should find the user-intent record from DB-RAG `active_intent.intent_id` first. If that is unavailable, it may use the selection or SQL artifact `intent_snapshot.intent_id`. If no matching user-intent record exists, cancel should remain successful and audit-only; it should not create a brand-new cancelled intent from partial review state.

### SQL Execution Completion

When reviewed DB-RAG SQL execution succeeds and writes a completed task:

- set the originating user intent `status="completed"`
- set `completed_task_id`
- set `updated_at`
- add a reciprocal `originating_user_intent_id` field to the completed task's `provenance`
- keep the completed task card as the reusable result record

The completed task remains the source for result-focused references such as "summarize the output" or "what SQL did you use?"

## Reference Resolution

The current task-memory resolver handles completed tasks. User-intent history adds a reference surface for incomplete or cancelled user goals.

Recommended resolution order after higher-priority live workflows:

1. Pending human review or interrupt
2. Pending clarification or DB-RAG opt-in
3. Pending deterministic workflow continuation
4. Completed-task reference resolver for result, output, SQL, dataset, figure, or artifact references
5. User-intent reference resolver for incomplete/cancelled user query references
6. Normal deterministic routing predicates
7. Planner fallback

The completed-task resolver should run before the user-intent resolver when the text points to a result or artifact because "summarize the output above" and "what SQL did you use?" should bind to completed work, not the originating query intent.

The user-intent resolver should be deterministic for common references:

- "previous query"
- "last query"
- "continue previous query"
- "continue that database query"

For these phrases, choose `last_user_intent_id_by_kind["db_rag_query"]` when the phrase says query or database. If that pointer is missing or invalid, fall back to the newest valid DB-RAG intent by `intent_order`, then `created_at`. Do not choose by `updated_at`, because cancel and completion are status updates rather than new user queries.

If the user says "previous result", "last output", or asks to inspect SQL/dataset/output, use completed task memory instead.

The first implementation should not add a general user-intent resolver result model. It only needs one deterministic branch:

```python
intent = latest_user_intent(state, kind="db_rag_query")
if intent and user_text_means_continue_previous_query(user_text):
    route_to_db_rag_with_question_override(intent["source_question"])
```

This keeps completed-task reference resolution as the only full reference resolver. `retry`, `edit`, `copy`, ambiguity handling, and LLM-backed intent resolution are reserved for later work.

## Continue Behavior

When the user says "continue previous query" and the latest DB-RAG user intent is `cancelled`, the system should not resume the cancelled review node.

Instead it should:

1. resolve the latest DB-RAG `memory.user_intents` record
2. create a new live DB-RAG `active_intent` from that record's `source_question` and `goal_text`
3. create or update a new user-intent record with `continued_from_intent_id` pointing to the prior record
4. route into DB-RAG as a fresh workflow

This preserves the user's intended query while respecting that Cancel ended the old pending review.

Continuation should not reuse stale review artifacts, `pending_column_review`, `pending_sql_candidate`, approved selection IDs, or SQL approval IDs from the cancelled workflow. It should enter DB-RAG through the same fresh-question path used by a new extraction-oriented request.

The new continued intent should get a new memory-layer `intent_id` and a new DB-RAG `active_intent.intent_id`. The prior cancelled intent should remain `cancelled`; it should not be mutated back to `active`.

## Future UI Fit

This design supports future modern chat actions without requiring a separate model later:

- Copy user query: read `source_question`
- Edit and resubmit: create a new user intent with `continued_from_intent_id` or `edited_from_intent_id`
- Retry cancelled query: create a new active DB-RAG intent linked to the cancelled intent
- Show intent history: render compact records ordered by `intent_order`

No UI work is required in the first implementation slice. The first slice should not add `edited_from_intent_id`, copy actions, edit actions, or retry actions until the UI needs them.

## Testing

Add focused tests for:

- memory initialization preserves existing `completed_tasks` behavior
- fresh DB-RAG question creates a `memory.user_intents` record
- DB-RAG active-intent update upserts the same user-intent record
- DB-RAG column-review cancel marks the originating user intent `cancelled`
- DB-RAG SQL-review cancel marks the originating user intent `cancelled`
- cancel still does not create or mutate `memory.completed_tasks`
- "continue previous query" resolves the latest user-originated DB-RAG intent
- continuing a cancelled query starts a fresh DB-RAG workflow instead of reopening the cancelled review
- status-only cancel updates do not reorder `intent_order`
- a fresh DB-RAG question supersedes the previous live DB-RAG user intent
- output/result references continue to use completed-task memory before user-intent memory
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

The only first-slice reference behavior should be exact deterministic phrase handling for "continue previous query" style requests. An LLM resolver can be added later only if deterministic intent-reference handling proves insufficient.
