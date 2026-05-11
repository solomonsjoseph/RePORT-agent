# Workflow Reference Turn Boundary Design

Date: 2026-05-11
Status: Proposed

## Goal

Prevent unresolved workflow-reference messages such as "resume the previous query" from being answered as fresh DB-RAG metadata questions.

The system should classify the latest user turn at the routing boundary, resolve workflow references against structured current-thread memory, and only enter DB-RAG retrieval when the message is a concrete database question or a validated continuation of one.

This design extends `2026-05-08-user-intent-history-design.md`. It does not replace `memory.user_intents`; it defines when and how that memory must be consulted before DB-RAG retrieval.

## Problem

The observed failure mode is:

1. A previous DB-RAG extraction request exists or is implied by the conversation.
2. The user later says "lets resume the previous query".
3. The message reaches the fresh DB-RAG path as literal question text.
4. DB-RAG retrieves schema context and calls `answer_from_context(...)` on the unresolved workflow pointer.
5. The response contains two parts:
   - a metadata-style answer to the literal message, such as "I don't have the actual previous query";
   - an extraction opt-in prompt built from the active or resolved intent.

The bad behavior is not the wording of the answer. The bad behavior is that an unresolved control-plane message entered the DB-RAG data-plane retrieval path.

## Design Principle

Workflow references are not database questions.

They are routing instructions about prior work. A workflow reference must be resolved or rejected before DB-RAG retrieval runs. If resolution succeeds, DB-RAG receives the stored original database question through a controlled override. If resolution fails, the system asks for clarification or asks the user to restate the query.

The invariant is:

```text
rag_db_qa fresh-question retrieval must never receive an unresolved workflow pointer as question text.
```

Examples of workflow pointers include "continue the previous query", "resume that extraction", "retry the database query", and "go back to the prior request". The implementation must not rely on hard-coded phrase lists for these examples. They illustrate the semantic class only.

## Grill Review Decisions

The following design questions were resolved during review:

1. **Should this introduce a new independent classifier?**
   - Recommended answer: no. Extend the existing current-thread user-intent reference classifier into a turn-boundary classifier. The current helper already builds compact `candidate_user_intents`, validates selected IDs, and returns advisory JSON. The improvement should add `turn_type` and clarification-reply awareness to that path instead of creating a parallel resolver.

2. **Should the classifier run when there are no candidate user intents?**
   - Recommended answer: yes for turn typing, no for target selection. The system still needs to know whether "resume the previous query" is a workflow reference. If no valid target exists, the result should be `workflow_reference` with `target="none"` and a terminal restatement prompt. It must not short-circuit to `new_request` before classifying the turn type.

3. **Should pending clarification always consume the next user turn?**
   - Recommended answer: no. Pending clarification is a priority only when the latest message satisfies that clarification kind's reply contract. Otherwise the clarification is abandoned for that turn and normal routing continues.

4. **Should DB-RAG have a defensive fallback for unresolved workflow references?**
   - Recommended answer: yes, but only as an assertion-style guard. The primary boundary belongs in the orchestrator. DB-RAG may reject a structured `turn_type="workflow_reference"` entry if one leaks through, but DB-RAG should not run semantic workflow-reference detection over fresh question text.

5. **Should `retry` be a first-slice relationship?**
   - Recommended answer: no. The existing validator supports `continue`, `refine`, `inspect_result`, and `new_request`. Treat user wording such as "retry the previous query" as relationship `continue` unless a later design adds true retry semantics. This avoids adding an unsupported relationship label without a distinct lifecycle behavior.

## Non-Goals

- Do not add phrase heuristics for words like "resume", "previous", or "continue".
- Do not make DB-RAG retrieval responsible for detecting unresolved workflow pointers.
- Do not ask the planner to reconstruct DB-RAG history from the transcript.
- Do not add cross-session memory.
- Do not resume cancelled review artifacts directly.
- Do not change SQL approval, column review, or execution semantics.

## Current State Surfaces

The design uses the existing state layers:

- `memory.user_intents`
  - compact current-thread registry for user-originated DB-RAG query intents.
  - stores source question, goal text, status, active DB-RAG intent ID, and ordering.

- `agents["rag_db_qa"]["active_intent"]`
  - live DB-RAG semantic intent for the current workflow.
  - not sufficient by itself after cancel or workflow cleanup.

- `memory.completed_tasks`
  - completed reusable outputs, especially executed DB-RAG SQL extraction tasks.
  - should win for result/artifact references such as "show the SQL" or "summarize the output".

- `meta`
  - one-turn routing handoff state.
  - should carry validated reference resolution with keys such as `resolved_user_intent_id` and `rag_db_question_override`.

## Turn Boundary Classifier

Extend the existing user-intent reference classifier into a semantic turn-boundary classifier that runs before fresh DB-RAG retrieval and planner fallback.

The current `classify_user_intent_reference(...)` path already owns compact intent payload construction and code-side validation. The new design should evolve that path rather than add a second classifier. The classifier should return both:

- a **turn type**, which determines whether the latest message is a clarification reply, workflow reference, fresh database question, or non-database request;
- an optional **validated target**, which identifies a specific user intent or completed task when the turn refers to prior work.

The classifier receives compact routing context only:

```json
{
  "latest_user_message": "lets resume the previous query",
  "active_workflow": {
    "awaiting_clarification": true,
    "clarification_kind": "rag_db_extraction_opt_in",
    "rag_db_thread_status": "awaiting_extraction_opt_in",
    "rag_db_active_thread": true,
    "pending_review": false
  },
  "candidate_user_intents": [
    {
      "intent_id": "intent_...",
      "display_ordinal": 3,
      "kind": "db_rag_query",
      "source_question": "Query my database for marital status, alcohol use, and diabetes status among index case participants.",
      "goal_text": "Extract index case participant factors associated with loss to follow-up.",
      "status": "cancelled",
      "created_at": "2026-05-11T00:00:00+00:00",
      "completed_task_id": null
    }
  ],
  "candidate_completed_tasks": [
    {
      "task_id": "task_...",
      "kind": "db_rag_sql_extraction",
      "label": "DB-RAG SQL extraction: ...",
      "source_question": "...",
      "summary": "...",
      "status": "completed"
    }
  ],
  "recent_turns": [
    {"role": "user", "text": "..."},
    {"role": "assistant", "text": "..."}
  ]
}
```

The payload must not include full SQL, full schema context, dataframe contents, artifact payloads, or long transcript excerpts.

### Output Schema

The classifier returns a small typed decision:

```json
{
  "turn_type": "workflow_reference",
  "target": "existing_user_intent",
  "target_id": "intent_...",
  "relationship": "continue",
  "confidence": "high",
  "needs_clarification": false,
  "reason": "The user is asking to resume a prior DB-RAG query rather than asking a new metadata question."
}
```

Allowed `turn_type` values:

- `clarification_reply`
- `workflow_reference`
- `fresh_database_question`
- `fresh_non_database_question`
- `ambiguous`

Allowed `target` values:

- `existing_user_intent`
- `completed_task`
- `new_user_intent`
- `none`

Allowed workflow-reference relationships:

- `continue`
- `refine`
- `inspect_result`

Only `continue` and `refine` against `existing_user_intent` route to DB-RAG fresh-question continuation. `inspect_result` should prefer completed task memory.

User wording such as "retry the previous query" should map to `relationship="continue"` in the first slice. True retry can be added later only if it needs distinct behavior from continuing the stored source question.

### No-Candidate Behavior

The turn-boundary classifier must still run when `memory.user_intents` is empty.

The previous user-intent resolver can short-circuit target resolution when no candidates exist, but this design needs turn typing even without candidates. For example, if the latest message is "resume the previous query" and there are no candidate intents, the correct result is:

```json
{
  "turn_type": "workflow_reference",
  "target": "none",
  "target_id": null,
  "relationship": "continue",
  "confidence": "high",
  "needs_clarification": false,
  "reason": "The user asked to continue prior work, but no prior DB-RAG user intent is recorded."
}
```

The orchestrator then returns the no-stored-prior-intent message. It does not route to fresh DB-RAG and does not call `answer_from_context(...)`.

## Validation Rules

The classifier is advisory. Code validates every actionable output.

For `workflow_reference` targeting `existing_user_intent`:

- `target_id` must exist in `memory.user_intents`.
- the card kind must be `db_rag_query`.
- `source_question` must be non-empty.
- `relationship` must be one of `continue` or `refine`.
- the source question used for continuation must be copied from memory, not from classifier free text.
- stale review pointers must not be reused.

For `workflow_reference` targeting `completed_task`:

- `target_id` must exist in `memory.completed_tasks`.
- task kind must support the requested relationship.
- result/artifact inspection must route through the completed-task reference path, not user-intent continuation.

For `ambiguous` or low-confidence output:

- the system asks a short clarification.
- it must not call DB-RAG retrieval on the unresolved latest message.

For invalid output:

- clear any partial resolved-reference metadata.
- ask the user to restate the prior database query if no valid target can be determined.

For `workflow_reference` with `target="none"`:

- do not route to `rag_db_qa`;
- do not ask the planner to reinterpret the message;
- return the no-stored-prior-intent response and mark the user turn answered.

For `fresh_database_question`:

- do not require an existing user-intent target;
- proceed through normal DB-RAG routing;
- create or update `memory.user_intents` at the DB-RAG intent write point.

## Routing Order

The orchestrator should apply this order for a fresh unanswered user turn:

1. Consume hard human-review decisions.
2. Consume deterministic review lifecycle transitions, such as approved SQL or cancelled review.
3. Build routing context, including pending clarification kind, active DB-RAG thread status, compact user intents, and compact completed tasks.
4. Classify the latest turn boundary.
5. If the turn is a valid reply for the active clarification kind, route to the clarification return node or pending DB-RAG workflow.
6. If the turn is a workflow reference to a completed result or artifact, run completed-task reference resolution.
7. If the turn is a validated workflow reference to a DB-RAG user intent, set `MetaKeys.RAG_DB_QUESTION_OVERRIDE` to the stored `source_question` and route to `rag_db_qa`.
8. If the turn is an unresolved workflow reference, ask clarification or ask the user to restate the query.
9. If the turn is a fresh database question, route normally to `rag_db_qa`.
10. Otherwise use normal planner fallback.

The important change is that a pending clarification should not automatically consume every subsequent user message. It should consume only messages that match its expected reply contract.

This order intentionally moves semantic turn-boundary classification before the current eager clarification branch. In the current code, `was_awaiting_clarification` can set `next_action` before user-intent classification runs. That ordering is the core reason an unrelated or referential message can bypass intent resolution.

## Clarification Reply Contracts

Each clarification kind should define what counts as a valid reply:

- `rag_db_extraction_opt_in`
  - valid replies: yes, no, or a substantive refinement of the active extraction intent.
  - invalid replies: unrelated new questions, workflow references to earlier intents, result inspection requests.

- `db_rag_recoverable_error`
  - valid replies: information needed to recover from the specific DB-RAG error.
  - invalid replies: fresh unrelated questions or references to a different prior query.

- `memory_reference_resolution`
  - valid replies: selected task ID, display ordinal, or unambiguous task label.
  - invalid replies: fresh DB-RAG request or new analysis request.

- `qa_followup`
  - soft clarification.
  - can be superseded by a workflow reference or a fresh database question.

If a message is not a valid reply to the active clarification, the clarification is abandoned for that turn and normal routing continues. This is a lifecycle rule, not a phrase heuristic.

### Clarification State Clearing

When a pending clarification is abandoned for the current turn:

- clear `AWAITING_USER_CLARIFICATION`, `CLARIFICATION_KIND`, `CLARIFICATION_RETURN_NODE`, and `PENDING_QUESTION` before routing the message as a new turn;
- preserve durable memory such as `memory.user_intents` and `memory.completed_tasks`;
- do not mutate DB-RAG review artifacts unless the abandoned clarification was itself the owner of those artifacts;
- append an observation explaining that the clarification was superseded by the latest turn boundary.

This prevents stale pending questions from being prepended later by `_question_from_stale_qa_followup(...)`.

## DB-RAG Entry Contract

`rag_db_qa` should receive one of three DB-RAG entry classes:

1. **Resolved continuation**
   - `question_override` is present.
   - DB-RAG treats the override as the canonical source question.
   - pending review artifacts and stale clarification state are ignored.

2. **Pending workflow reply**
   - active DB-RAG state indicates a pending opt-in, column review, SQL review, or recoverable error.
   - the latest message has already been accepted as a valid reply for that pending workflow.

3. **Fresh database question**
   - latest user message is classified or routed as a concrete database question.
   - DB-RAG may retrieve context and answer or start extraction.

DB-RAG should not need to infer workflow-reference semantics in the fresh-question path. That boundary belongs to the orchestrator.

### Defensive Guard

DB-RAG may include a defensive guard that rejects an explicit structured workflow-reference turn if the orchestrator passes one through by mistake. The guard should be driven by structured metadata such as `meta["turn_type"] == "workflow_reference"`, not by scanning question text for words. The response should be terminal and should not call retrieval.

## Failure Handling

### No Stored Prior Intent

If the turn is a workflow reference but there is no valid DB-RAG user intent:

```text
I do not have a previous DB-RAG query recorded in this thread. Please restate the database query you want to continue.
```

The exact wording can vary, but the behavior must not fall through to DB-RAG metadata retrieval.

### Multiple Candidate Intents

If multiple DB-RAG intents are plausible:

```text
Which previous database query did you want to continue?
Task 1: ...
Task 2: ...
```

The user selection then resolves through the existing memory-reference clarification pattern or a user-intent-specific equivalent.

### Prior Intent Was Cancelled

Cancellation is terminal for the old pending review, but the originating user intent remains referenceable.

Continuing a cancelled intent should:

1. leave the prior intent status as `cancelled`;
2. start a new DB-RAG workflow from the prior card's `source_question`;
3. create a new active DB-RAG intent;
4. create or update a new `memory.user_intents` card with `continued_from_intent_id`;
5. avoid reusing old `pending_column_review_artifact_id`, `pending_sql_candidate_artifact_id`, or approved selection state.

## State Transitions

### Resolved Workflow Reference

When validation succeeds:

```python
meta["resolved_user_intent_id"] = intent_id
meta["resolved_user_intent_kind"] = "db_rag_query"
meta["resolved_user_intent_relationship"] = "continue"
meta["resolved_user_intent_source_question"] = memory["user_intents"][intent_id]["source_question"]
meta["resolved_user_intent_user_message_hash"] = current_hash
meta["rag_db_question_override"] = memory["user_intents"][intent_id]["source_question"]
```

After `rag_db_qa` consumes the override, `rag_db_question_override` must be cleared. Resolved-reference metadata must be consumed once and cleared when the user message hash changes.

### Unresolved Workflow Reference

When validation fails:

```python
meta["awaiting_user_clarification"] = True
meta["clarification_kind"] = "user_intent_reference_resolution"
meta["clarification_return_node"] = "orchestrator"
meta["pending_question"] = "Which previous database query did you want to continue?"
```

If there are no candidates, the system may answer terminally and clear clarification state instead of asking the user to choose from an empty list.

The recommended first-slice behavior for no candidates is terminal answer plus cleared clarification state. Asking "which one?" when the candidate list is empty creates another dead-end clarification.

## Planner Role

The planner should not discover workflow references from raw transcript text.

The planner may receive compact context:

```json
{
  "candidate_user_intents": [...],
  "validated_reference": {
    "target": "existing_user_intent",
    "target_id": "intent_...",
    "relationship": "continue"
  }
}
```

If `validated_reference` exists, the planner treats it as a routing fact. It should not override or reinterpret the reference unless the target action is unavailable.

The planner should receive `turn_type` as an input fact after validation. It should not be responsible for deciding that a raw message is a workflow reference, because planner fallback happens too late to protect DB-RAG retrieval from literal workflow-pointer text.

## Implementation Boundaries

Recommended first-slice implementation boundaries:

- Extend `graph/memory/user_intent_store.py` classifier schema and validation to include `turn_type` and `target="none"`.
- Remove the no-candidate `new_user_intent` short-circuit for turn-boundary classification. It may remain only inside a lower-level target resolver that runs after turn type is known.
- Add an orchestrator helper that classifies the turn before eager clarification routing chooses `next_action`.
- Add a helper that decides whether a `turn_type="clarification_reply"` result is valid for the active `CLARIFICATION_KIND`.
- Keep `rag_db_qa` continuation entry through `MetaKeys.RAG_DB_QUESTION_OVERRIDE`.
- Keep DB-RAG fresh-question retrieval unchanged except for optional structured defensive guard.

Do not implement this by adding text checks inside `_handle_fresh_db_rag_question(...)`, `answer_from_context(...)`, or retrieval prompts.

## Testing

Add focused tests for:

- `continue previous query` with a valid cancelled DB-RAG user intent routes to `rag_db_qa` with `rag_db_question_override`.
- the same message with no `memory.user_intents` is classified as `workflow_reference`, does not route to fresh DB-RAG retrieval, and asks the user to restate the query.
- pending `rag_db_extraction_opt_in` consumes `yes` and `no` replies normally.
- pending `rag_db_extraction_opt_in` does not consume a workflow reference to a different prior intent.
- soft `qa_followup` clarification can be superseded by a workflow reference.
- completed-task result references still beat user-intent continuation.
- low-confidence or ambiguous workflow-reference classification asks clarification.
- invalid classifier target IDs are rejected and do not set `rag_db_question_override`.
- resolved continuation consumes `rag_db_question_override` exactly once.
- DB-RAG fresh-question tests assert that unresolved workflow-reference text is not passed to `retrieve_context()` or `answer_from_context()`.
- the classifier payload remains compact and does not include SQL text, schema context, dataframe contents, or artifact payloads.
- abandoned clarification state is cleared so stale pending questions are not prepended into the next DB-RAG question.
- wording like "retry the prior query" maps to `relationship="continue"` until true retry semantics are designed.

## Acceptance Criteria

- A workflow-reference turn is resolved or clarified before DB-RAG retrieval.
- The literal text "resume previous query" is never answered through `answer_from_context(...)`.
- Valid continuations use the stored `memory.user_intents[*].source_question`.
- Workflow-reference turns with no stored prior intent produce a terminal restatement prompt, not a new DB-RAG metadata answer.
- Cancelled DB-RAG review state is not resumed directly.
- Existing DB-RAG opt-in replies still work.
- No phrase-list heuristic is added as the primary decision mechanism.
