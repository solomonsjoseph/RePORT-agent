# Human Review Cancel Design

Date: 2026-05-07
Status: Proposed

## Goal

Add a Cancel action to every human review interrupt so the user can stop the currently pending review workflow without approving, regenerating, executing, finalizing, or continuing from that review.

Cancel is primarily for cases where the workflow entered a review unexpectedly. It must make the conversation usable again for a new request and must not leave live state that causes the same review to reappear automatically.

## Scope

This design covers these review nodes:

- `human_review_before_run`
- `human_review_after_error`
- `human_review_before_output`
- `human_review_rag_db_column_selection`
- `human_review_rag_db_sql_execution`

The design also covers their Streamlit review panels and focused regression tests.

## Non-Goals

- No task-memory entry for cancelled reviews.
- No failed-task card for cancelled reviews.
- No planner memory or routing suppression based on cancel.
- No long-term preference that prevents future DB-RAG, code generation, or review workflows.
- No deletion of immutable artifacts or semantic conversation events.
- No fallback branch that treats cancel as regeneration.

## Global Cancel Contract

Each review UI sends:

```python
{"action": "cancel"}
```

Each review node handles `cancel` before approve, regenerate, feedback, or finalization logic.

Cancel must:

- append a short visible assistant message, such as "Cancelled the pending review. You can start a new request when ready."
- append a `review_decision` semantic event with `decision="cancel"` and the node's review kind
- clear live continuation state that would make deterministic routing resume the cancelled workflow
- leave artifacts and semantic conversation history intact
- avoid appending optional feedback text as a `HumanMessage`
- return the thread to an idle usable state

Cancel must not:

- approve code, SQL, columns, or final output
- regenerate code, column selections, or SQL
- execute code or SQL
- finalize successful output
- create or mutate task memory
- add routing hints for future runs

## Memory Semantics

Cancel is audit-only in this version.

It writes:

- a visible assistant message
- a `review_decision` semantic event with `decision="cancel"`

It does not write:

- `memory.completed_tasks`
- `memory.failed_tasks`
- a cancelled-task registry
- planner memory
- DB-RAG routing hints
- any future suppression rule

Reasoning: cancel means the current review should end. It does not mean the user completed, failed, or permanently rejected a task. Existing completed task cards remain unchanged. If the cancelled workflow was a revision of a prior task, the parent task remains available for normal future reference resolution.

## Review Node Behavior

### `human_review_before_run`

Cancel means the generated code must not run and must not regenerate.

The node should:

- append the global cancellation assistant message
- emit `review_decision` with `review_kind="before_run_review"` and `decision="cancel"`
- set `human_review.before_run_decision = "cancel"`
- set `human_review.approved_code_hash = None`
- clear `MetaKeys.EXECUTION_TICKET_HASH`
- clear `MetaKeys.ERROR_RECOVERY_ACTIVE`
- clear `MetaKeys.FINAL_APPROVED_CODE_HASH`
- set executor state to idle
- remove `output.generated_code` as the live continuation field so `human_review_before_run` is no longer ready
- ignore optional suggestion text

The generated code may remain in conversation events or artifacts if already stored there, but `output.generated_code` must not remain a live continuation signal.

### `human_review_after_error`

Cancel means the failed code-analysis workflow stops after repeated execution errors.

The node should:

- append the global cancellation assistant message
- emit `review_decision` with `review_kind="after_error_review"` and `decision="cancel"`
- set `human_review.after_error_decision = "cancel"`
- set `human_review.before_run_decision = None`
- set `human_review.approved_code_hash = None`
- reset `MetaKeys.ERROR_ITERATIONS = 0`
- clear `MetaKeys.EXECUTION_TICKET_HASH`
- clear `MetaKeys.ERROR_RECOVERY_ACTIVE`
- clear `MetaKeys.FINAL_APPROVED_CODE_HASH`
- set executor `run_status = "idle"`
- remove `output.generated_code` as the live continuation field and clear retry-continuation state so terminal error review does not immediately reappear
- ignore optional suggestion text

Prior error details remain visible through conversation and audit history.

### `human_review_before_output`

Cancel means the successful execution result is not accepted as final output.

The node should:

- append the global cancellation assistant message
- emit `review_decision` with `review_kind="final_review"` and `decision="cancel"`
- set `human_review.final_decision = "cancel"`
- clear `MetaKeys.FINAL_APPROVED_CODE_HASH`
- clear `MetaKeys.EXECUTION_TICKET_HASH`
- clear `MetaKeys.ERROR_RECOVERY_ACTIVE`
- set executor `run_status = "idle"` to neutralize final-review readiness
- avoid appending the approved final assistant result message
- avoid creating code-analysis task memory
- ignore optional suggestion text

Cancel does not mean the execution failed. It means the user declined to accept or finalize the result.

### `human_review_rag_db_column_selection`

Cancel means the active DB-RAG extraction workflow stops before SQL generation.

The node should:

- append the global cancellation assistant message
- emit `review_decision` with `review_kind="rag_db_column_selection"` and `decision="cancel"`
- mark the selection artifact content status as `"cancelled"` for audit clarity
- clear `pending_column_review_artifact_id`
- clear `approved_column_selection_artifact_id`
- clear `pending_sql_candidate_artifact_id`
- clear `pending_column_review`
- clear `pending_sql_candidate`
- clear `sql_review_approved_artifact_id`
- set DB-RAG `thread_status = "cancelled"`
- set DB-RAG `active_thread = False`
- avoid task memory writes
- ignore optional feedback text

The selection artifact and review request event remain available for audit, but no live pointer should cause the column review to resume.

### `human_review_rag_db_sql_execution`

Cancel means the active DB-RAG extraction workflow stops before SQL execution.

The node should:

- append the global cancellation assistant message
- emit `review_decision` with `review_kind="rag_db_sql_execution"` and `decision="cancel"`
- mark the SQL candidate artifact content status as `"cancelled"` for audit clarity
- clear `pending_sql_candidate_artifact_id`
- clear `pending_sql_candidate`
- clear `approved_column_selection_artifact_id`
- clear `pending_column_review_artifact_id`
- clear `pending_column_review`
- clear `sql_review_approved_artifact_id`
- set DB-RAG `thread_status = "cancelled"`
- set DB-RAG `active_thread = False`
- avoid SQL execution
- avoid dataset creation
- avoid `db_rag_sql_extraction` task memory writes
- ignore optional feedback text

The SQL and selection artifacts remain available for audit, but all live pointers must be removed so SQL cannot replay, regenerate, or execute from the old approval.

## UI Behavior

All five human review panels should include a Cancel button.

Cancel UI rules:

- no feedback text is required
- typed feedback is ignored when Cancel is clicked
- no approve-with-feedback confirmation is shown for Cancel
- no separate confirmation modal is required in the first version
- Cancel should be visually secondary or neutral
- preferred button ordering is approve or submit first, regenerate or feedback second where present, and Cancel last

For `human_review_after_error`, the panel currently has only a feedback submission path. It should add Cancel as a second action that does not require feedback.

## Routing And Readiness Requirements

The implementation must prove that cancel disables the exact readiness condition that would otherwise reopen the review:

- `human_review_before_run` readiness depends on live `output.generated_code`, idle or pending executor state, and no fresh run approval. Cancel must remove the live generated-code continuation signal or otherwise make the predicate false.
- `human_review_after_error` readiness depends on exhausted retryable execution error state and `after_error_decision is None`. Cancel must set a terminal review decision and clear retry-continuation state.
- `human_review_before_output` readiness depends on executor `run_status == "ok"` and missing final approval for the current code hash. Cancel must move executor status away from `ok`.
- `human_review_rag_db_column_selection` readiness depends on `pending_column_review.status == "awaiting_review"`. Cancel must clear or neutralize that pending review.
- `human_review_rag_db_sql_execution` readiness depends on `pending_sql_candidate.status == "prepared"`. Cancel must clear or neutralize that pending SQL candidate.

These requirements are deterministic control behavior. They are not planner fallback rules and should not alter `docs/superpowers/specs/orchestrator-gating-policy.md`.

## Testing

Add focused tests for:

- each review UI emits `{"action": "cancel"}` without requiring feedback
- each review node emits a `review_decision` event with `decision="cancel"`
- each review node appends the cancellation assistant message
- `human_review_before_run` cancel does not leave the before-run review ready and cannot route to `execute_code`
- `human_review_after_error` cancel does not trigger regeneration or another terminal error review
- `human_review_before_output` cancel does not append final approved output and does not leave final review ready
- DB-RAG column cancel clears pending column review live pointers
- DB-RAG SQL cancel clears pending SQL and approved selection live pointers
- cancelled reviews do not create or mutate `memory.completed_tasks` or `memory.failed_tasks`

Existing approve and regenerate tests should continue to pass unchanged except where UI layout assertions need to account for a third button.
