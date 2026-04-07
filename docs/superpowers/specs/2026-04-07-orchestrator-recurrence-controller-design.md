# Orchestrator Recurrence Controller Design

**Goal**

Replace the current split between snapshot-based stagnation hints and structural loop guards with a deterministic recurrence controller that:

- knows when the workflow is complete and should `END`
- distinguishes hard progress, weak progress, and no progress
- prevents graph recursion / max-iteration failures
- preserves planner-owned node selection after resume handling

## Problem

The current orchestrator has two incomplete recurrence mechanisms:

1. `progress.py` computes snapshot deltas and exposes `stagnation_count`, but it treats many artifact changes as progress even when the workflow is spinning.
2. `loop_guards.py` detects repeated action structures like two-node cycles, but it is only a structural fuse and is not milestone-aware.

This causes two classes of failure:

- false progress: repeated `generate_code` steps mutate code and reset progress even though the workflow has not advanced
- missing completion: after a direct `qa` answer like "who are you", the orchestrator may keep selecting `qa` instead of ending

## Design Principles

- Progress classification must be deterministic, not LLM-driven.
- Completion detection must be deterministic, not delegated to the planner.
- The planner remains the semantic next-node selector after resume handling.
- Hard safety gates and impossible-action masking remain deterministic.
- Recurrence control should be based on workflow milestones and blockers, not enumerated path graphs.

## Workflow Status Model

Each cycle derives three values from state:

```python
WorkflowMilestone = Literal[
    "awaiting_clarification",
    "waiting_for_tool_results",
    "needs_code",
    "awaiting_run_review",
    "ready_to_execute",
    "retrying_after_error",
    "awaiting_after_error_review",
    "awaiting_final_review",
    "answered",
    "terminal_error",
]

CompletionStatus = Literal["complete", "incomplete", "blocked_waiting"]

BlockerSignature = str | None
```

These values are derived from current state, not stored as ground truth.

## Milestone Derivation

Milestones are evaluated in priority order:

1. `awaiting_clarification`
   - `meta.awaiting_user_clarification == True`
   - completion: `blocked_waiting`
   - blocker: `waiting_for_user_clarification:{clarification_kind}`

2. `waiting_for_tool_results`
   - tool queue active or requester is waiting on tool results
   - completion: `blocked_waiting`
   - blocker: `waiting_for_tool_results`

3. `awaiting_after_error_review`
   - retryable execution error exists
   - retry budget exhausted
   - no consumed after-error decision yet
   - completion: `blocked_waiting`
   - blocker: `waiting_for_after_error_review:{error_category}:{error_type}`

4. `retrying_after_error`
   - retryable execution error exists and retry budget remains
   - or `error_recovery_active == True`
   - completion: `incomplete`
   - blocker: `retryable_error:{error_type}:{error_message_hash}`

5. `awaiting_run_review`
   - generated code exists
   - no valid execution ticket for current code
   - executor is not already successful
   - completion: `blocked_waiting`
   - blocker: `waiting_for_before_run_review`

6. `ready_to_execute`
   - generated code exists
   - execution ticket matches current code
   - executor is `idle` or `pending`
   - completion: `incomplete`
   - blocker: `ready_for_execution`

7. `awaiting_final_review`
   - executor `run_status == "ok"`
   - final review has not been consumed
   - completion: `blocked_waiting`
   - blocker: `waiting_for_final_review`

8. `terminal_error`
   - terminal execution error has been surfaced
   - completion: `complete`
   - blocker: `terminal_error:{category}`

9. `answered`
   - the current turn produced a direct user-facing answer
   - and no code/review/tool/clarification workflow remains active
   - completion: `complete`
   - blocker: `None`

10. `needs_code`
   - fallback when none of the above applies
   - completion: `incomplete`
   - blocker: `missing_next_step`

## Completion Controller

The orchestrator should deterministically decide when to end a run segment:

- if `completion_status == "complete"` -> `end`
- if `completion_status == "blocked_waiting"` -> `end`
- otherwise the workflow remains active and the planner selects the next node

This fixes the endless direct-QA loop:

- `qa` answers "who are you"
- no clarification/tool/code/review workflow remains
- milestone becomes `answered`
- completion becomes `complete`
- orchestrator ends

This keeps deterministic completion without reintroducing deterministic node auto-selection.

## Progress Classification

Each cycle also derives:

```python
ProgressClass = Literal["hard", "weak", "none"]
```

based on:

- previous milestone
- current milestone
- previous blocker
- current blocker
- selected action
- artifact deltas

### Hard Progress

Counts as real forward movement and resets recurrence counters.

Examples:

- `needs_code -> awaiting_run_review`
- `awaiting_run_review -> ready_to_execute`
- `ready_to_execute -> awaiting_final_review`
- `awaiting_clarification -> answered`
- retryable error signature changes materially
- tool results arrive and change the active blocker

### Weak Progress

Some local work product changed, but the workflow milestone and blocker did not.

Examples:

- generated code hash changed while still `awaiting_run_review`
- generated code hash changed while still `retrying_after_error`
- planner picked a different action but the workflow stayed in the same blocked state

Important rule:

- code mutation alone is never `hard` progress

### No Progress

The workflow milestone and blocker did not change, and no meaningful work product changed.

Examples:

- repeated `qa` after already answering the same turn
- repeated `execute_code` with the same error signature
- repeated waiting state without new input
- repeated planning churn with the same effective state

## Recurrence Controller

Track:

- `current_milestone`
- `current_blocker`
- `progress_class`
- `stagnation_count`
- `weak_progress_count`
- `last_meaningful_action`

Update policy:

- `hard` -> reset `stagnation_count` and `weak_progress_count`
- `weak` -> increment `weak_progress_count`
- `none` -> increment `stagnation_count`

Escalate when:

- repeated `none` in the same milestone/blocker exceeds threshold
- repeated `weak` churn in the same milestone/blocker exceeds threshold
- same action repeats too often in the same milestone

The current loop-guard ideas become one part of this controller, not a separate system.

## Escalation Policy

Escalation must be deterministic.

Recommended order:

1. if the workflow plausibly lacks user direction, surface a concise clarification / cannot-progress summary
2. if the workflow is stuck in code or error churn, surface a concise failure summary
3. only use hard `end` as the final safety fuse

This replaces raw graph-recursion failure with explicit framework-controlled termination behavior.

## Orchestrator Decision Ladder

The target control order is:

1. consume-on-use human review decisions
2. clarification resume handling
3. tool-handler requester return
4. derive workflow milestone, blocker, and completion status
5. if completion is `complete` or `blocked_waiting`, return `end`
6. planner selects next node semantically
7. action mask blocks impossible actions
8. graph routing enforces execute safety
9. recurrence controller may override with escalation or final `end`

This preserves planner-centric node selection while restoring deterministic stopping and recurrence control.

## Implementation Shape

Recommended module changes:

- add `graph/nodes/orchestrator/workflow_status.py`
  - derive milestone, blocker, and completion
- add `graph/nodes/orchestrator/progress_controller.py`
  - derive progress class and recurrence escalation
- simplify or retire:
  - `graph/nodes/orchestrator/progress.py`
  - `graph/nodes/orchestrator/loop_guards.py`
- update `graph/nodes/orchestrator/node.py`
  - use completion controller before planner choice
  - use recurrence controller after planner choice

## Testing

Required test coverage:

1. direct QA answer ends cleanly
2. repeated code generation in same review milestone is weak/no progress, not hard progress
3. retry loop with changed error signature counts as hard progress
4. retry loop with same error signature eventually escalates
5. clarification pending ends the current run segment cleanly
6. final reviewed result ends cleanly
7. stale code workflow does not force continuation after a complete direct answer
