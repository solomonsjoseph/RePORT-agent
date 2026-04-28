# DB-RAG Extraction Opt-In And Thread-Intent Design

> Superseded by `docs/superpowers/specs/2026-04-28-db-rag-intent-review-workflow-design.md`.
> This file is retained for historical context only and is no longer the canonical implementation contract.

## Goal

Revise the DB-RAG workflow so extraction-oriented requests do not immediately enter column review, do not rely on the latest user message as the retrieval query, and do not require a separate UI interrupt for extraction opt-in.

The revised workflow must:

1. answer metadata/schema questions directly from retrieved context
2. detect when a request is moving toward row-level extraction or subsetting
3. ask an explicit follow-up question before starting column-selection review
4. preserve a structured DB-RAG thread intent across turns
5. allow the opt-in follow-up to accept substantive user refinements
6. generate column selections from the stored extraction intent, not from a vague follow-up utterance
7. keep SQL generation gated behind approved column review
8. keep SQL execution gated behind explicit SQL review approval

## Problem

The current `rag_db_qa` node has two design weaknesses:

1. It derives the effective question from the latest user message for most DB-RAG turns.
   - A follow-up like "perform the sql extraction for me" becomes the retrieval query.
   - Retrieval then loses the original semantics of the extraction request.

2. It moves directly from SQL-needed detection into column-selection review.
   - The system never asks the user whether they want to proceed to selection.
   - This removes an expected conversational checkpoint.

These are workflow-boundary problems, not retrieval-ranking problems. Expanding retrieval to more raw chat turns would only blur the query further.

## Design Principles

- Do not solve follow-up interpretation with keyword heuristics.
- Do not concatenate arbitrary conversation history into the retrieval query.
- Use typed workflow state for DB-RAG progression.
- Treat extraction opt-in as a clarification state, not as a free-form fresh query.
- Preserve thread memory after a decline, but prevent automatic progression.
- Freeze an intent snapshot before column selection so later turns cannot silently mutate the reviewed target.

## Recommended Approach

Use a state-machine DB-RAG workflow with a dedicated clarification subtype for extraction opt-in.

Do not add a new UI interrupt node for extraction opt-in.

Instead, reuse the existing clarification flow:

- `rag_db_qa` answers the metadata/schema portion of an extraction-oriented request.
- It then asks: "Would you like me to identify the tables and columns suitable for this extraction?"
- It stores structured extraction intent in DB-RAG thread state.
- It sets clarification metadata so the next user reply is consumed as the answer to that question, not as a fresh retrieval query.

This preserves the conversational feel of a coding agent while keeping the workflow typed and deterministic.

## Request Classes

`rag_db_qa` will handle four request classes:

1. Metadata QA
   - Examples: schema, variable meanings, forms, joinable tables, table descriptions.
   - Behavior: retrieve context, answer directly, stop.

2. Extraction-oriented QA
   - Examples: subset, extract, count, filter, aggregate, row-level cohort requests.
   - Behavior: retrieve context, answer briefly from metadata, store structured extraction intent, ask whether to proceed to table/column identification.

3. Extraction opt-in clarification reply
   - Examples: "yes", "no", "yes but only confirmed index cases".
   - Behavior:
     - positive: proceed using stored intent
     - negative: clear pending opt-in, keep intent in thread memory as inactive
     - substantive refinement: update stored intent and proceed directly into column-selection generation

4. Review-driven SQL workflow
   - Column selection is prepared only after opt-in.
   - SQL is prepared only after approved column review.
   - SQL is executed only after approved SQL review.

## Structured DB-RAG State

Revise `agents["rag_db_qa"]` to include:

```python
{
  "thread_status": "idle" | "answered_metadata" | "awaiting_extraction_opt_in" |
                   "awaiting_column_review" | "awaiting_sql_review" |
                   "completed" | "error",
  "active_thread": bool,
  "active_intent": {
    "intent_id": str,
    "source_user_question": str,
    "goal_text": str,
    "mode": "metadata" | "extraction",
    "population": str,
    "requested_fields": list[str],
    "filters": list[str],
    "status": "active" | "declined" | "superseded",
  },
  "intent_snapshot_for_selection": {
    "intent_id": str,
    "goal_text": str,
    "population": str,
    "requested_fields": list[str],
    "filters": list[str],
  } | None,
  "pending_extraction_opt_in": {
    "question": str,
    "intent_id": str,
    "goal_text": str,
    "status": "awaiting_reply",
  } | None,
  "pending_column_review": {
    "question": str,
    "selection_id": str,
    "tables": list[str],
    "columns": list[dict],
    "rationale": str,
    "feedback_history": list[dict],
    "status": "awaiting_review" | "approved" | "needs_revision",
  } | None,
  "pending_sql_candidate": {
    "question": str,
    "sql": str,
    "tables": list[str],
    "columns": list[dict],
    "selection_id": str,
    "status": "prepared",
  } | None,
  "last_retrieval_context": {
    "tables": list[str],
    "columns": list[str],
  } | None,
  "last_database_question": str,
}
```

## Intent Resolution

Add an explicit intent-resolution stage in the DB-RAG service boundary.

The service should produce a structured intent object for extraction-oriented requests rather than forcing `rag_db_qa` to reason from raw text alone.

Minimal service responsibility:

```python
def resolve_intent(question: str, context: DbRagContext, prior_intent: dict | None = None) -> DbRagIntent:
    ...
```

This method should:

- determine whether the request is metadata-only or extraction-oriented
- normalize the extraction goal into `goal_text`
- extract population, requested fields, and filters where possible
- merge substantive clarification feedback into the prior intent when resuming an extraction opt-in clarification

The node may still call `answer_from_context()`, but the answer path must receive enough structured intent information to persist thread state.

## Retrieval Contract

Retrieval must not default to the latest user utterance.

Instead:

1. For a new DB-RAG turn, retrieval may use the new question.
2. For an active extraction thread, retrieval should use the resolved intent representation:
   - `goal_text`
   - population
   - requested fields
   - filters
3. For column-selection generation, retrieval should use `intent_snapshot_for_selection`, not whatever the latest user message happens to be.

This means a reply like "perform the sql extraction for me" or "yes" never becomes a semantic retrieval query by itself.

Multi-turn grounding should be selective and structured, not full-history concatenation.

## Clarification Integration

Extraction opt-in should be implemented as a dedicated clarification subtype rather than a new review UI node.

When `rag_db_qa` identifies an extraction-oriented request:

- append the metadata-aware answer
- ask whether to identify tables/columns suitable for the extraction
- set:
  - `awaiting_user_clarification = True`
  - `clarification_kind = "rag_db_extraction_opt_in"`
  - `clarification_return_node = "rag_db_qa"`
  - `pending_question` to the prompt being asked
- persist `active_intent`
- persist `pending_extraction_opt_in`

On clarification resume:

- `yes` or equivalent:
  - freeze `intent_snapshot_for_selection`
  - generate column selection
  - clear clarification metadata

- `no` or equivalent:
  - clear `pending_extraction_opt_in`
  - keep `active_intent` with `status="declined"`
  - clear clarification metadata
  - stop

- substantive refinement:
  - update `active_intent`
  - freeze revised `intent_snapshot_for_selection`
  - generate column selection directly using the revised intent
  - clear clarification metadata

The system should not restate the revised intent before proceeding. The concrete validation point is the column-selection review itself.

## DB-RAG Node Flow

Each `rag_db_qa` invocation should follow this order:

1. Check provider/readiness.
2. If there is a prepared SQL candidate awaiting SQL review, replay it.
3. If there is a pending approved column review with no SQL candidate, prepare SQL.
4. If there is a pending revision on column review, regenerate selection using the frozen selection intent and feedback history.
5. If resuming `clarification_kind="rag_db_extraction_opt_in"`, consume the user reply as an opt-in clarification outcome.
6. Otherwise treat the turn as a new DB-RAG question:
   - retrieve context
   - answer metadata
   - resolve intent
   - if metadata-only: stop
   - if extraction-oriented: ask opt-in question and wait

## Human Review Stages

Column review remains explicit and unchanged in principle:

- generate selection only after opt-in
- show interpreted extraction goal in the review payload
- allow approval or revision feedback

SQL review remains explicit and unchanged in principle:

- generate SQL only after approved column review
- execute only after approved SQL review

## Output Contract

For metadata-only requests:

- answer directly
- do not create selection or SQL state

For extraction-oriented requests before opt-in:

- answer briefly from metadata
- ask whether to identify relevant tables and columns
- do not create `pending_column_review`
- do not create `pending_sql_candidate`
- do not populate `output["prepared_sql_candidate"]`

For column review:

- show the interpreted extraction goal
- show selected tables
- show exact selected columns
- show rationale
- state that SQL will be prepared only after approval

For SQL review:

- show approved tables
- show approved columns
- show prepared SQL
- ask for execution confirmation

## Required Revisions

### `graph/nodes/rag_db_qa.py`

- Add `pending_extraction_opt_in` handling.
- Stop treating the latest user message as the default effective retrieval query for active extraction follow-ups.
- Split extraction opt-in from column-review generation.
- Freeze `intent_snapshot_for_selection` before generating column selections.
- Stop writing column-review payloads into `output["prepared_sql_candidate"]`.
- Store `thread_status` explicitly.

### `graph/nodes/clarification.py`

- Add resume support for `clarification_kind="rag_db_extraction_opt_in"`.
- Route back into `rag_db_qa` with the stored DB-RAG intent context.

### `graph/nodes/orchestrator/policy.py`

- Ensure active DB-RAG clarification states continue to prefer DB-RAG over generic QA.
- Preserve deterministic resumption through clarification before planner fallback.

### `graph/nodes/orchestrator/workflow_status.py`

- Add a blocked milestone for extraction opt-in clarification if needed for status reporting.

### `graph/nodes/node_registry.py`

- No new extraction-opt-in review node is needed.
- Existing clarification readiness must continue to surface the follow-up question as a blocked waiting state.

### `db_rag/service.py`

- Add structured intent resolution.
- Ensure column-selection generation can accept a frozen selection intent rather than loosely phrased follow-up text.

### `db_rag/retrieval.py`

- Keep retrieval query inputs structured and selective.
- Avoid raw multi-turn concatenation as the default retrieval strategy.

### Review UI Components

- Existing column review UI should include the interpreted extraction goal.
- Existing SQL review UI can remain conceptually unchanged.

## Testing

Add coverage for:

1. extraction-oriented request yields a follow-up opt-in question instead of immediate column review
2. positive opt-in proceeds into column selection using stored intent
3. negative opt-in preserves inactive intent and stops progression
4. substantive clarification updates the intent and proceeds directly into column selection
5. vague follow-up utterances do not become retrieval queries
6. column-review payload does not populate `prepared_sql_candidate`
7. approved column review still prepares SQL
8. approved SQL review still executes SQL and persists subset artifacts

## Non-Goals

- No new extraction opt-in UI node
- No heuristic keyword-only patch for "yes", "extract", or "do it"
- No full conversation-history retrieval concatenation
- No change to the existing explicit human review requirement for SQL execution

## Recommendation

Implement the extraction opt-in as a clarification subtype plus structured DB-RAG intent state.

That is the smallest correct design that fixes:

- missing next-step prompting
- loss of extraction context across follow-ups
- accidental retrieval on vague user replies
- ambiguous progression from QA to selection to SQL
