# DB-RAG Intent, Opt-In, And Review-Constraint Workflow Design

## Supersedes

This document supersedes the following earlier design notes and is the canonical design reference for this DB-RAG workflow change:

- `docs/superpowers/specs/2026-04-27-db-rag-extraction-opt-in-design.md`
- `docs/superpowers/specs/2026-04-28-db-rag-review-feedback-constraints-design.md`

Those earlier documents remain on disk for historical context but should not be used as the implementation contract going forward.

## Goal

Revise the DB-RAG workflow so extraction-oriented requests follow an explicit opt-in flow, preserve structured thread intent across turns, and treat human review feedback as binding constraints that affect retrieval, selection, and validation.

The revised workflow must:

1. answer metadata/schema questions directly from retrieved context
2. detect when a request is moving toward extraction, subsetting, counting, or filtering
3. ask an explicit follow-up before starting table/column review
4. preserve a structured DB-RAG `active_intent` across turns
5. allow the opt-in reply to include substantive user refinements
6. generate table/column candidates from the stored intent, not from a vague latest utterance
7. propagate human review feedback into retrieval before regeneration
8. treat explicit reviewer table/column directives as hard constraints
9. validate regenerated selections against those hard constraints
10. keep SQL generation gated behind approved column review
11. keep SQL execution gated behind approved SQL review

## Problem

The current DB-RAG workflow has two coupled design failures.

### Failure 1: latest-message-driven follow-up handling

`rag_db_qa` usually derives the effective question from the latest user message.

That makes vague follow-ups such as:

- "perform the sql extraction for me"
- "yes"
- "do it"

become retrieval queries or DB-RAG inputs even though they are not semantically complete requests.

This causes retrieval drift and breaks thread continuity.

### Failure 2: weak review-feedback propagation

When the user gives explicit feedback during human review:

- retrieval is regenerated from the original review question rather than revised constraints
- feedback only has strong effect when it happens to mention exact schema tokens
- natural-language reviewer corrections are mostly reduced to prompt text

As a result, human feedback appears visible in the interface but often does not materially change the next regenerated candidate set.

These are workflow-boundary problems, not just prompt-quality or retrieval-ranking problems.

## Design Principles

- Do not solve follow-up interpretation with keyword heuristics.
- Do not concatenate arbitrary conversation history into retrieval queries.
- Use typed workflow state for DB-RAG progression.
- Use one canonical structured intent object: `active_intent`.
- Freeze explicit snapshots of intent for review stages instead of letting later turns mutate the review target.
- Convert human feedback into structured constraints before the next retrieval pass.
- Treat explicit human review directives as binding unless they are impossible to satisfy.
- Fail clearly on unsatisfied hard constraints instead of silently drifting to approximate matches.

## Workflow Overview

The revised workflow is a small state machine.

### Request classes

`rag_db_qa` handles four request classes:

1. Metadata QA
   - Examples: schema, forms, join keys, variable meanings, table descriptions.
   - Behavior: retrieve context, answer directly, stop.

2. Extraction-oriented QA
   - Examples: subset, extract, count, filter, aggregate, or row-level cohort requests.
   - Behavior: retrieve context, answer briefly from metadata, resolve and store `active_intent`, ask whether to identify suitable tables and columns, then wait.

3. Extraction opt-in clarification reply
   - Examples: "yes", "no", "yes but only confirmed index cases".
   - Behavior:
     - positive: proceed using stored intent
     - negative: clear pending opt-in, keep intent in thread memory as inactive
     - substantive refinement: update `active_intent` and proceed directly into table/column generation

4. Review-driven SQL workflow
   - Column review happens only after opt-in.
   - SQL generation happens only after approved column review.
   - SQL execution happens only after approved SQL review.

### Thread stages

`agents["rag_db_qa"]["thread_status"]` should use:

- `idle`
- `answered_metadata`
- `awaiting_extraction_opt_in`
- `awaiting_column_review`
- `awaiting_sql_review`
- `completed`
- `error`

## Structured State

Revise `agents["rag_db_qa"]` to include:

```python
{
  "thread_status": "idle" | "answered_metadata" | "awaiting_extraction_opt_in" |
                   "awaiting_column_review" | "awaiting_sql_review" |
                   "completed" | "error",
  "active_thread": bool,
  "active_intent": {
    "intent_id": str,
    "source_question": str,
    "goal_text": str,
    "mode": "metadata" | "extraction",
    "population": str | None,
    "requested_fields": list[str],
    "filters": list[str],
    "required_tables": list[str],
    "required_columns": list[str],
    "excluded_tables": list[str],
    "excluded_columns": list[str],
    "feedback_history": list[dict[str, str]],
    "status": "active" | "declined" | "superseded",
  } | None,
  "intent_snapshot_for_selection": {
    "intent_id": str,
    "goal_text": str,
    "population": str | None,
    "requested_fields": list[str],
    "filters": list[str],
    "required_tables": list[str],
    "required_columns": list[str],
    "excluded_tables": list[str],
    "excluded_columns": list[str],
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

## Active Intent Contract

`active_intent` is the only canonical mutable intent object.

Its fields serve these purposes:

- `goal_text`
  - normalized extraction goal

- `population`
  - cohort/entity target such as index cases or household contacts

- `requested_fields`
  - user-facing variables requested for the subset or extraction

- `filters`
  - row-level restrictions that affect eligibility

- `required_tables`
  - reviewer-required tables

- `required_columns`
  - reviewer-required exact columns

- `excluded_tables`
  - reviewer-disallowed tables

- `excluded_columns`
  - reviewer-disallowed exact columns

- `feedback_history`
  - raw audit trail of review feedback and review transitions

### Snapshot semantics

`intent_snapshot_for_selection` is a frozen copy of `active_intent` used for a specific review or regeneration cycle.

This prevents later turns from mutating the review target silently while a human is evaluating table/column candidates.

## Clarification Integration

Extraction opt-in should reuse the existing clarification mechanism rather than introducing a new review UI node.

When `rag_db_qa` identifies an extraction-oriented request, it should:

- append the metadata-aware answer
- ask: "Would you like me to identify the tables and columns suitable for this extraction?"
- set:
  - `awaiting_user_clarification = True`
  - `clarification_kind = "rag_db_extraction_opt_in"`
  - `clarification_return_node = "rag_db_qa"`
  - `pending_question` to the opt-in prompt
- persist `active_intent`
- persist `pending_extraction_opt_in`
- set `thread_status = "awaiting_extraction_opt_in"`

### Clarification reply behavior

On clarification resume:

- `yes` or equivalent:
  - freeze `intent_snapshot_for_selection`
  - generate table/column candidates
  - clear clarification metadata

- `no` or equivalent:
  - clear `pending_extraction_opt_in`
  - keep `active_intent` with `status="declined"`
  - clear clarification metadata
  - stop

- substantive refinement:
  - update `active_intent`
  - freeze revised `intent_snapshot_for_selection`
  - generate table/column candidates directly from the revised intent
  - clear clarification metadata

The system should not insert another separate confirmation after a substantive refinement. The next concrete validation point is column review.

## Retrieval Contract

Retrieval must not default to the latest user utterance for active extraction follow-ups.

### Retrieval input construction

For new DB-RAG turns:

- retrieval may use the new question text

For active extraction threads and review regeneration:

- retrieval should use structured intent fields:
  - `goal_text`
  - `population`
  - `requested_fields`
  - `filters`

Then apply hard constraints:

- inject `required_tables` and `required_columns` into the candidate set before ranking/finalization
- remove `excluded_tables` and `excluded_columns` from the candidate set before selection prompting

### Why raw multi-turn concatenation is not the answer

Full-history retrieval concatenation is explicitly out of scope because it:

- adds noise
- blurs the semantic target
- cannot express exclusion cleanly
- still depends on similarity ranking to do control-flow work

Retrieval should be structured and selective, not history-heavy.

## Review Feedback Propagation

Human review feedback must be converted into structured constraint updates before the next retrieval pass.

### Regeneration pipeline

1. human review captures feedback
2. system appends raw feedback to `active_intent.feedback_history`
3. system parses feedback into updates to:
   - `population`
   - `filters`
   - `required_tables`
   - `required_columns`
   - `excluded_tables`
   - `excluded_columns`
4. system updates `active_intent`
5. system freezes a new `intent_snapshot_for_selection`
6. retrieval runs from that updated snapshot
7. selection LLM sees both the revised retrieval context and the explicit constraint state
8. post-selection validation enforces those constraints

This means review feedback must affect both:

- retrieval inputs
- the LLM selection prompt

It must not remain prompt-only.

## Hard-Constraint Semantics

Reviewer directives should act as hard constraints, not ranking preferences.

### Required tables and columns

If the reviewer explicitly requires a table or column:

- it must appear in the next candidate set
- selection output must include it when logically relevant to the request
- if the required item cannot be found or reconciled with the request, regeneration must stop with a clear explanation

### Excluded tables and columns

If the reviewer explicitly excludes a table or column:

- retrieval must not surface it as a usable candidate
- selection output must not include it
- SQL generation must not reference it

### Failure behavior

If hard constraints produce an unsatisfiable selection space, the system must:

1. stop regeneration
2. explain the failure clearly
3. preserve review state for correction

It must not silently back off to unrelated retrieved tables.

## Selection Prompt Contract

`prepare_column_selection()` should continue receiving:

- retrieved context
- feedback history
- previous selection candidate

It should additionally receive the constrained intent snapshot explicitly in the prompt.

The prompt should state that:

- required tables/columns are binding
- excluded tables/columns are disallowed
- previous selection is context, not authority
- the output must use only schema-valid exact pairs

## Post-Selection Validation

After the LLM returns a candidate selection, validate it against the constrained snapshot.

Validation must check:

1. every required table is present
2. every required column is present
3. no excluded table appears
4. no excluded column appears

If validation fails:

- reject the candidate
- return a clear regeneration failure or fallback review state
- do not silently accept the candidate

## Human Review Semantics

### Column review

Column review remains explicit.

It should display:

- interpreted extraction goal
- selected tables
- selected columns
- rationale
- prior feedback history

On regenerate:

- feedback updates `active_intent`
- retrieval regenerates from the updated constrained snapshot
- selection regenerates from the new retrieval context

On approve:

- the candidate is accepted as-is
- typed text should not silently mutate the accepted candidate

The current approve-with-typed-feedback confirmation behavior is therefore conceptually correct: `approve` means accept the current candidate, not revise it.

### SQL review

SQL review remains explicit and conceptually unchanged:

- SQL is prepared only after approved column review
- SQL execution happens only after explicit SQL approval

If SQL review requests regeneration, it must route back through the same `active_intent` update path rather than only appending a free-form human message.

## DB-RAG Node Flow

Each `rag_db_qa` invocation should follow this order:

1. check provider/readiness
2. if a prepared SQL candidate is awaiting SQL review, replay it
3. if an approved column review exists with no SQL candidate, prepare SQL
4. if a column review is marked `needs_revision`, update `active_intent` from review feedback, regenerate retrieval from the constrained snapshot, then regenerate selection
5. if resuming `clarification_kind="rag_db_extraction_opt_in"`, consume the user reply as an opt-in clarification outcome
6. otherwise treat the turn as a new DB-RAG question:
   - retrieve context
   - answer metadata
   - resolve or update `active_intent`
   - if metadata-only: stop
   - if extraction-oriented: ask opt-in question and wait

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

- show interpreted extraction goal
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

- add `pending_extraction_opt_in` handling
- split extraction opt-in from column-review generation
- stop treating the latest user message as the effective retrieval query for active extraction follow-ups
- stop regenerating retrieval directly from `review_question` on `needs_revision`
- update `active_intent` from feedback before regeneration
- freeze `intent_snapshot_for_selection` before each generation/review cycle
- stop writing column-review payloads into `output["prepared_sql_candidate"]`
- store `thread_status` explicitly

### `graph/nodes/clarification.py`

- add resume support for `clarification_kind="rag_db_extraction_opt_in"`
- route back into `rag_db_qa` with the stored DB-RAG intent context

### `graph/nodes/orchestrator/policy.py`

- ensure active DB-RAG clarification states continue to prefer DB-RAG over generic QA
- preserve deterministic resumption through clarification before planner fallback

### `graph/nodes/orchestrator/workflow_status.py`

- add a blocked milestone for extraction opt-in clarification if needed for status reporting

### `graph/nodes/node_registry.py`

- no new extraction opt-in review node is needed
- existing clarification readiness must continue to surface the follow-up question as a blocked waiting state

### `db_rag/service.py`

- add structured intent resolution
- add intent-update logic that merges human feedback into `active_intent`
- add retrieval helpers that accept required/excluded tables and columns
- ensure column-selection generation can accept a frozen constrained intent snapshot
- add post-selection constraint validation

### `db_rag/retrieval.py`

- support candidate injection and exclusion before final candidate assembly
- keep ranking logic but do not let ranking override hard constraints

### Review UI Components

- existing column review UI should include the interpreted extraction goal
- existing SQL review UI can remain conceptually unchanged
- approve-with-feedback should continue to preserve the semantics that `approve` accepts the current candidate unchanged

## Testing

Add coverage for:

1. extraction-oriented request yields an opt-in follow-up instead of immediate column review
2. positive opt-in proceeds into column selection using stored intent
3. negative opt-in preserves inactive intent and stops progression
4. substantive clarification updates intent and proceeds directly into column selection
5. vague follow-up utterances do not become retrieval queries
6. regenerate feedback changes retrieval inputs, not only the prompt
7. required table feedback forces that table into the next candidate set
8. excluded table feedback removes that table from the candidate set
9. invalid selection outputs that violate hard constraints are rejected
10. SQL-review regeneration follows the same intent-update path
11. approved column review still prepares SQL
12. approved SQL review still executes SQL and persists subset artifacts
13. column-review payload no longer populates `prepared_sql_candidate`

## Non-Goals

- no new extraction opt-in UI node
- no heuristic keyword-only patch for "yes", "extract", or "do it"
- no full conversation-history retrieval concatenation
- no silent fallback from unsatisfied hard constraints to approximate matches
- no mixing of approval semantics with regeneration semantics

## Recommendation

Implement a single unified DB-RAG workflow based on:

- `active_intent` as the canonical mutable intent object
- `intent_snapshot_for_selection` as the frozen review-stage snapshot
- clarification-driven extraction opt-in
- retrieval driven by structured intent rather than vague follow-up text
- hard review constraints enforced across retrieval, selection, and validation

This is the smallest correct design that fixes:

- missing next-step prompting
- loss of extraction context across follow-ups
- weak propagation of human review feedback
- accidental retrieval on vague replies
- ambiguous progression from QA to selection to SQL
