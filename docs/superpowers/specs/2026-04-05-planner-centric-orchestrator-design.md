# Planner-Centric Orchestrator Design

Date: 2026-04-05
Status: Draft for review

## Context

The current orchestration design mixes three concerns:

- shared workflow facts
- node selection policy
- deterministic safety routing

Today, node selection is spread across:

- `graph/nodes/node_registry.py`
- `graph/nodes/orchestrator/policy.py`
- `graph/routing.py`
- `graph/nodes/orchestrator/node.py`

This creates two problems:

1. The workflow state is carrying more than minimal facts because parts of it are effectively used to decide which node is "ready".
2. The planner/orchestrator is not acting as the true master agent because routing logic is fragmented across readiness predicates, fallback rules, and graph edges.

The target design keeps the planner/orchestrator as the master decision-maker, gives it richer environment and action history, and removes node-selection policy from minimal shared state.

## Goals

- Keep the planner/orchestrator as the primary agent that selects the next node.
- Give the planner richer decision context: environment, recent history, node observations, and semantic routing guidance.
- Keep shared workflow state minimal and factual.
- Preserve the current clarification flow for tool usage.
- Add deterministic progress and stagnation detection so the workflow stops looping before hitting `MAX_ITERATION`.
- Make it straightforward to add new nodes later without recreating a registry-policy tangle.

## Non-Goals

- This design does not replace the planner with a deterministic state machine.
- This design does not redesign the current tool-clarification behavior.
- This design does not introduce long-conversation context compaction in the first implementation pass.
  The first version may continue using the existing raw transcript plus recent-window approach.
- This design does not fully specify future hard guardrails such as mandatory human review before execution or after successful execution.
  The framework will reserve a place for them, but concrete guardrails are deferred.

## Design Summary

The planner remains the true orchestrator. After each step, it chooses the next node based on a rich semantic context, not from `is_ready(state)` predicates.

The framework is split into five responsibilities:

1. Shared workflow state stores durable facts only.
2. Nodes produce factual updates plus planner-visible observations.
3. A planner context builder assembles the current environment and history.
4. A narrow action mask removes only impossible actions.
5. The planner chooses the next node semantically from the full environment.

## Core Architecture

### 1. Shared Workflow State

The workflow state should store only durable facts and user-visible artifacts. It should not encode node-selection policy.

Target shape:

```python
state = {
    "messages": [...],
    "artifacts": {
        "generated_code": ...,
        "execution_output": ...,
        "error": ...,
        "tool_results": ...,
    },
    "node_data": {
        "qa": {...},
        "generate_code": {...},
        "executor": {...},
        "human_review": {...},
    },
    "planner": {
        "last_decision": ...,
        "decision_trace": [...],
    },
    "meta": {
        "workflow_trace": [...],
        "clarification": {...},
    },
}
```

Principles:

- `artifacts` contains shared outputs used by UI, replay, and downstream reasoning.
- `node_data` contains node-owned persistent facts, not routing rules.
- `planner` contains planner-facing decision history.
- `meta` contains framework metadata such as workflow trace and clarification bookkeeping.

The planner may read all of this as environment context. None of it should act as readiness policy by itself.

### 2. Node Contract

Nodes should use a uniform result shape:

```python
class NodeResult(TypedDict):
    state_patch: dict
    observations: list[str]
    result_summary: str | dict | None
```

Each node does three things:

- updates factual state
- emits structured observations for planner consumption
- optionally emits a concise summary of what changed

Nodes do not decide the next node. They do not own routing policy.

Examples of node observations:

- `code_generated`
- `execution_failed_retryable`
- `execution_failed_terminal`
- `tool_requests_created`
- `tool_results_received`
- `needs_user_clarification`
- `final_answer_ready`

These observations are planner inputs, not deterministic transitions.

### 3. Planner Context Builder

The planner should not be choosing from a thin ready list with a short state summary. It should see a richer environment model each cycle.

The planner context builder should assemble:

- latest user message
- recent workflow trace
- recent planner decisions
- recent node observations and summaries
- relevant artifacts
- relevant `node_data` slices
- static node capability descriptions
- semantic routing guidance
- progress and stagnation signals

The planner context builder is the core abstraction of this redesign. It centralizes environment assembly so node selection is based on coherent semantic context rather than scattered readiness logic.

In the first implementation pass, the planner context builder may continue to use the current conversation model:

- full transcript retained in workflow state
- recent raw message window sent to the model
- no rolling summary or pinned-facts layer yet

This is intentionally simple. If long conversations later require compaction, the planner context builder is the extension point for adding:

- rolling summaries
- pinned facts / durable constraints
- other memory-compaction logic

That future work should not require redesigning the rest of the framework.

### 4. Narrow Action Mask

A deterministic action mask should still exist, but its scope must stay narrow.

Its job is only to remove actions that are impossible or explicitly disallowed at the current moment.

Examples:

- do not allow `execute_code` when no generated code exists
- do not allow `tool_handler` when no tool requests are pending
- preserve the current tool clarification resume behavior when clarification is actively in progress

This mask is not the semantic router. It is only a hard constraint layer.

### 5. Planner / Orchestrator

The planner/orchestrator remains the master agent.

Each cycle:

1. A node runs and writes facts plus observations.
2. The framework records the state delta and recent outcome.
3. The planner context builder assembles the environment.
4. The action mask removes impossible actions.
5. The planner chooses the best next node semantically.

This preserves the planner's role as the decision-maker while removing policy from shared workflow state.

## Node Extensibility

Adding a node later should be simple and local.

Each node module should export:

- `name`
- `capability`
- `run(state, deps) -> NodeResult`
- optional `planner_context(state) -> str | dict`

To add a new node:

1. create the node module
2. register it once in the graph builder / node map
3. expose its capability to the planner context
4. add an impossibility-mask rule only if the action is literally invalid in some states
5. add a `node_data[new_node_name]` slice if the node needs persistent local facts

No `is_ready` predicate should be required for node selection.

## Node-Specific State

Node-specific state is allowed and expected. It should live under `node_data`.

Examples:

- `node_data["qa"]`
- `node_data["generate_code"]`
- `node_data["executor"]`
- `node_data["human_review"]`

Rules:

- node-specific state stores facts, results, and local history
- node-specific state may be read by the planner as part of the environment
- node-specific state must not be used as embedded node-selection policy

This preserves extensibility without turning node state into a hidden routing engine.

## Preserved Clarification Flow

The current clarification flow for tool usage should remain structurally unchanged for now.

Preserved path:

- `qa`
- `clarification`
- back to `qa` or tool routing

The planner can see that clarification is active, but the detailed clarification behavior is not being redesigned in this spec.

This reduces risk and preserves an existing workflow that already behaves acceptably.

## Failure Handling

Failure handling should distinguish between framework failures and domain outcomes.

### Framework failures

If a node crashes or returns an invalid result shape, the framework should handle that deterministically through an internal error path. The planner should not be asked to reason over malformed framework behavior.

### Domain outcomes

If a node completes normally and produces a meaningful outcome, that outcome should be recorded as facts and observations and fed back into planner context.

Examples:

- retryable execution error
- terminal execution error
- tool failure
- clarification needed

### Hard interruption states

Human review interrupts and the current tool-clarification flow remain explicit pause/resume states in the workflow.

## Progress and Stagnation Control

The graph-loop failure indicates that the framework lacks a strong notion of progress. This should be solved by deterministic progress tracking, not only by a raw iteration ceiling.

### Progress snapshot

After each step, compute a normalized snapshot such as:

```python
ProgressSnapshot = {
    "last_action": ...,
    "generated_code_hash": ...,
    "execution_status": ...,
    "error_signature": ...,
    "tool_result_count": ...,
    "clarification_pending": ...,
    "review_status": ...,
    "user_visible_output_hash": ...,
}
```

### Progress rule

`progress_made_last_step = True` when the step creates new usable information or materially changes the workflow situation.

Examples that count as progress:

- new generated code hash
- new tool result
- execution status changed
- error signature changed
- clarification resolved
- new clarification question asked
- new user-visible answer or failure summary produced

Examples that do not count as progress:

- only the trace changed
- same node repeated with same error signature
- same action repeated without artifact delta
- no new user-visible output

### Stagnation controls

The orchestrator layer should maintain:

- `progress_made_last_step`
- `stagnation_count`
- `repeated_failure_signature`

Then apply three controls:

1. action repetition guard
2. state-delta guard
3. escalation policy

When stagnation is detected, the system should steer toward a recovery action such as:

- user clarification
- human review
- concise terminal summary
- explicit "cannot make progress" response

`MAX_ITERATION` remains a final safety fuse, not the primary convergence mechanism.

## Testing Strategy

The redesign should be tested at the architecture boundaries.

### Planner context tests

Verify that planner context contains the correct semantic signals:

- recent actions
- recent observations
- progress flags
- error signature
- clarification state
- relevant node-specific facts

### Action mask tests

Verify that only impossible actions are removed.

Examples:

- no `execute_code` without code
- no `tool_handler` without pending tool requests
- preserved clarification behavior remains intact

### Planner-orchestrator behavior tests

Given mocked planner outputs and controlled state snapshots, verify:

- planner choices are honored
- stagnation metadata is included when relevant
- repeated no-progress loops trigger escalation before `MAX_ITERATION`

### Preserved subworkflow tests

Retain and adapt tests for:

- tool clarification
- human review
- execution error handling

The goal is to ensure the redesign improves planner context without regressing stable subworkflows.

## Migration Plan

This refactor should be incremental.

### Phase 1: extract planner context builder

Move environment assembly, workflow trace, observations, and progress signals into a dedicated planner context layer.

### Phase 2: replace readiness-driven planner inputs

Stop using `is_ready`-driven summaries as the planner's main routing input. Replace them with:

- static node capability metadata
- node observations
- relevant shared facts
- narrow impossibility mask

### Phase 3: reshape state gradually

Evolve toward `artifacts`, `node_data`, `planner`, and `meta` without breaking the existing graph all at once.

### Phase 4: add deterministic progress tracking

Introduce progress and stagnation tracking before removing more of the old policy code. This addresses loop failures early.

### Phase 5: simplify legacy policy

Only after the new planner context and progress controls are in place, simplify:

- `graph/nodes/orchestrator/policy.py`
- readiness-based selection in `graph/nodes/node_registry.py`
- duplicated fallback logic in current orchestrator routing

## Deferred Topics

The following topics are intentionally deferred:

- long-conversation context compaction such as rolling summaries or pinned facts
- exact hard guardrails for mandatory human review before execution
- exact hard guardrails for mandatory human review after successful execution
- whether the thin action mask should later grow into a stronger invariant layer

The framework should leave room for these, but they are not part of the first implementation plan.

## Decision

Adopt a planner-centric orchestration framework where:

- the planner remains the master agent
- shared workflow state stays factual and minimal
- node-specific state lives under `node_data`
- tool clarification behavior remains as-is
- deterministic progress tracking prevents graph stagnation loops
- readiness predicates stop being the main mechanism for node selection
