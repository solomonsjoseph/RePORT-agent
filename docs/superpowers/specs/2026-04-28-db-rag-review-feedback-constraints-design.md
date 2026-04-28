# DB-RAG Review Feedback Propagation And Hard-Constraint Design

> Superseded by `docs/superpowers/specs/2026-04-28-db-rag-intent-review-workflow-design.md`.
> This file is retained for historical context only and is no longer the canonical implementation contract.

## Goal

Revise DB-RAG review regeneration so explicit human feedback materially changes the next retrieval and selection pass.

The revised workflow must:

1. preserve reviewer feedback as structured state, not only raw text
2. propagate reviewer feedback into retrieval inputs before selection regeneration
3. keep reviewer feedback in the LLM selection prompt as explicit instruction context
4. treat explicit reviewer table/column directives as hard constraints
5. fail clearly when hard constraints cannot be satisfied instead of silently drifting to unrelated schema

## Problem

The current review-feedback path is too weak.

### Weakness 1: feedback does not affect retrieval

When column review is regenerated, `rag_db_qa` reuses the original `review_question` for retrieval and does not incorporate reviewer feedback into the retrieval query or retrieval constraints.

That means instructions such as:

- "use the cohort A final outcome table instead"
- "exclude Form 13"
- "only include confirmed index cases"

do not reliably change the retrieved context set.

### Weakness 2: feedback only has hard force for exact schema mentions

The current service logic only converts review feedback into concrete constraints when it can detect exact schema tokens such as:

- `Table.Column`
- a uniquely named column code

Natural-language instructions remain soft prompt text.

### Weakness 3: approve-with-feedback discards the feedback

If a reviewer types an instruction and still clicks approve, the current UI intentionally drops that text from the resume payload.

This is acceptable only if approval is meant to mean "proceed exactly as-is." It is not acceptable if the system is expected to preserve instruction text for downstream behavior.

## Design Principles

- Reviewer feedback must change system behavior before the next retrieval pass, not only during selection prompting.
- Do not concatenate raw feedback text blindly into retrieval.
- Convert feedback into structured intent constraints.
- Treat explicit reviewer directives as binding unless they are impossible to satisfy.
- Preserve a human-readable feedback history for auditability, but do not rely on it as the primary machine-readable control surface.

## Recommended Approach

Add a structured `revised_intent` layer between review feedback and retrieval regeneration.

The regeneration pipeline becomes:

1. human review captures feedback
2. system parses feedback into structured constraint updates
3. system updates the DB-RAG intent state
4. retrieval runs from the revised intent
5. selection LLM sees both the revised retrieval context and the explicit feedback/constraint state
6. post-selection validation enforces the hard constraints

This is better than choosing only retrieval or only prompt injection.

## Revised Intent Contract

Store a normalized selection-oriented intent object:

```python
{
  "intent_id": str,
  "source_question": str,
  "goal_text": str,
  "population": str | None,
  "requested_fields": list[str],
  "filters": list[str],
  "required_tables": list[str],
  "required_columns": list[str],
  "excluded_tables": list[str],
  "excluded_columns": list[str],
  "feedback_history": list[dict[str, str]],
}
```

### Field Semantics

- `goal_text`
  - normalized extraction goal

- `population`
  - cohort/entity target such as index cases or household contacts

- `requested_fields`
  - user-facing variables requested for the subset or extraction

- `filters`
  - row-level restrictions that affect cohort eligibility

- `required_tables`
  - explicit reviewer-required tables

- `required_columns`
  - explicit reviewer-required exact columns

- `excluded_tables`
  - explicit reviewer-disallowed tables

- `excluded_columns`
  - explicit reviewer-disallowed exact columns

- `feedback_history`
  - raw audit trail of human review instructions and review transitions

## Hard-Constraint Semantics

Reviewer directives should act as hard constraints, not ranking preferences.

### Required tables and columns

If the reviewer explicitly requires a table or column:

- it must be present in the next candidate set
- selection output must include it when logically relevant to the request
- if it cannot be found in schema or cannot be reconciled with the request, the system must stop and explain why

### Excluded tables and columns

If the reviewer explicitly excludes a table or column:

- retrieval must not surface it as a usable candidate
- selection output must not include it
- SQL generation must not reference it

### Failure behavior

If hard constraints produce an unsatisfiable selection space, the system must:

1. stop regeneration
2. surface the reason clearly
3. preserve the review state for correction

It must not silently back off to unrelated retrieved tables.

## Retrieval Contract

Retrieval should be driven by structured intent, not by raw feedback text alone.

### Retrieval input construction

Build retrieval from:

- `goal_text`
- `population`
- `requested_fields`
- `filters`

Then apply constraint semantics:

- inject `required_tables` and `required_columns` into the candidate set before ranking/finalization
- remove `excluded_tables` and `excluded_columns` from the candidate set before selection prompting

The retrieval layer should expose a path for explicit candidate injection rather than relying purely on embedding similarity.

### Why not raw text concatenation

Appending feedback directly to the retrieval query is insufficient because:

- it is noisy
- it is ambiguous
- it cannot express exclusion cleanly
- it still depends on vector similarity to do constraint work

Structured retrieval control is required.

## Selection Prompt Contract

`prepare_column_selection()` should continue receiving:

- retrieved context
- reviewer feedback history
- previous selection

It should additionally receive the revised intent constraints explicitly in the prompt.

The selection prompt should be told:

- required tables/columns are binding
- excluded tables/columns are disallowed
- previous reviewed selection is context, not authority
- it must return only schema-valid exact pairs

## Post-Selection Validation

After the LLM returns a candidate selection, validate it against the revised intent.

Validation must check:

1. every required table exists in the candidate table set
2. every required column exists in the candidate column set
3. no excluded table appears
4. no excluded column appears

If validation fails:

- reject the candidate
- return a clear regeneration error or fallback review state
- do not silently pass the candidate through

## Review Node Semantics

### Column review regeneration

When the reviewer clicks regenerate:

- preserve raw feedback in `feedback_history`
- parse the feedback into structured constraint updates
- update the revised intent
- regenerate retrieval from that revised intent
- regenerate selection from the revised retrieval context

### Approve with typed feedback

Approval should remain a no-op with respect to changing selection or retrieval behavior.

If the reviewer types text and still approves, that text should not mutate the approved selection silently.

This preserves a clean semantics:

- `approve` means accept the current candidate
- `regenerate` means change the candidate

The existing confirm step before approval is therefore conceptually correct.

## Required Revisions

### `graph/nodes/rag_db_qa.py`

- On `needs_revision`, derive a revised intent from review feedback before calling retrieval.
- Stop calling `retrieve_context(review_question, ...)` directly for regeneration.
- Store the revised intent and retrieval summary in agent state.

### `db_rag/service.py`

- Add intent-update logic that merges human feedback into a normalized revised intent object.
- Add retrieval helpers that accept required/excluded tables and columns.
- Add post-selection constraint validation.

### `db_rag/retrieval.py`

- Support candidate injection and exclusion prior to final candidate assembly.
- Keep ranking logic, but do not let ranking override hard constraints.

### `graph/nodes/human_review_rag_db_column_selection.py`

- Continue recording raw feedback history.
- Preserve the distinction between `approve` and `regenerate`.

### `graph/nodes/human_review_rag_db_sql_execution.py`

- When SQL-review feedback causes selection regeneration, route the same revised-intent update path instead of only appending a human message.

## Testing

Add coverage for:

1. regenerate feedback changes retrieval inputs, not just the prompt
2. required table feedback forces that table into the next candidate set
3. excluded table feedback removes that table from the candidate set
4. natural-language feedback is converted into revised intent state
5. invalid selection outputs that violate hard constraints are rejected
6. SQL-review regeneration follows the same revised-intent path
7. approve-with-feedback does not mutate the approved candidate

## Non-Goals

- No free-text multi-turn retrieval concatenation
- No silent fallback from unsatisfied hard constraints to approximate matches
- No mixing of approval semantics with regeneration semantics

## Recommendation

Add a structured revised-intent layer and make reviewer directives binding at retrieval, selection, and validation time.

This is the smallest correct design that fixes the current gap where human feedback appears in prompts but does not reliably affect the actual regenerated candidate set.
