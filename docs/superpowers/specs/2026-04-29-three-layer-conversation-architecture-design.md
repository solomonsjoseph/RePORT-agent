# Three-Layer Conversation Architecture Design

Date: 2026-04-29
Status: Proposed

## Goal

Redesign the conversation/state architecture so the framework remains maintainable as it grows across:

- ordinary QA with agent tool loops
- code generation and execution
- DB-RAG retrieval and SQL review
- human review checkpoints
- richer artifacts such as code, figures, tables, SQL candidates, and exports

The redesign should separate raw runtime trace, semantic conversation history, artifact storage, and workflow snapshot state so UI, orchestration, and exports do not all depend on the same overloaded structures.

## Current Problems

The current framework mixes multiple concerns into the same state surfaces.

- `messages` currently acts as both raw LangGraph/LangChain runtime trace and chat UI source.
- `output` acts partly as latest-result snapshot and partly as a convenience transcript.
- heavy payloads such as code, figures, and SQL candidates do not follow one shared artifact contract.
- UI rendering depends directly on raw `messages`, which leaks provider-native or tool-native trace details into the product surface.
- nodes and UI each infer their own view of conversation history instead of reading from one normalized layer.

This has already started to create maintenance problems around:

- tool traces in QA
- clarification resumption semantics
- duplicate display content
- code/figure attachment representation
- export inconsistency

## Design Principles

1. Keep runtime trace and product-facing history separate.
2. Preserve current orchestrator and worker memory behavior unless explicitly changed.
3. Use append-only, audit-friendly history for semantic events.
4. Keep payload-heavy artifacts out of the event log.
5. Centralize event, artifact, output, and display projections behind shared helpers.
6. Migrate in stages toward one canonical target rather than keep permanent parallel abstractions.

## Target State Model

The framework should use four distinct state roles:

### 1. Raw Runtime Trace

`state["messages"]`

- authoritative LangGraph/LangChain execution transcript
- provider-native and tool-native message objects
- used for prompt history and runtime continuity
- not the default UI transcript source
- may be compacted in the future for very long threads

### 2. Semantic History

`artifacts["conversation_events"]`

- mandatory shared thread-level event log
- append-only
- human-readable and normalized
- used for UI, export, audit, and future analysis
- not the default prompt-history source

### 3. Artifact Store

`artifacts["files"]`

- shared artifact namespace for the thread
- stores manifests and payload references
- holds code, figures, tables, SQL candidates, text artifacts, and similar outputs
- payloads are immutable snapshots

### 4. Workflow Snapshot

`output`

- current resolved workflow result only
- latest answer, latest code, latest execution result, latest error, latest SQL candidate
- not full history
- treated as a current-view cache / projection

### 5. Control Plane

`meta`, `planner`, `orchestrator`

- clarification state
- recurrence and progress state
- workflow milestone / blocker state
- semantic last-action override
- planner memory
- turn identity bookkeeping

## Semantic Event Layer

### Scope

The event layer is shared across all nodes, not node-specific.

All nodes append into one shared event log with explicit producer identity.

### Storage Location

Store the event log in:

- `artifacts["conversation_events"]`

Seed it at thread creation as an empty list.

### Versioning

Version the semantic event and artifact manifest schemas from day one.

Suggested top-level fields:

- `artifacts["conversation_events_version"] = 1`
- `artifacts["artifact_manifest_version"] = 1`

### Event Ordering and Identity

Every event should carry:

- `event_id`
- `seq`
- `created_at`
- `user_turn_hash`
- `actor`
- `actor_role`
- `type`

Ordering rules:

- `seq` is authoritative for order
- `created_at` is for diagnostics and telemetry

Sequence allocation is centralized per thread. Nodes do not invent local ordering.

### Event Semantics

The event log is append-only.

Events are not mutated or deleted after creation. If a later step changes the meaning of an earlier step, emit a new event that resolves or supersedes it.

Useful linkage fields:

- `parent_event_id`
- `resolved_by_event_id`
- `superseded_by_event_id`
- `status`

### Event Types

Use a discriminated union, not one loose dict shape.

Common user-facing event types:

- `user`
- `assistant`
- `clarification`
- `review_request`
- `review_decision`
- `code`
- `figure`
- `table`
- `sql_candidate`

Operational event types:

- `tool_call`
- `tool_result`
- `retrieval`
- `execution_started`
- `execution_finished`
- `retry`
- `error`
- `routing_decision`
- `workflow_ended`
- `guard_triggered`

DB-RAG should use the same shared base schema with DB-RAG-specific event types such as:

- `column_selection_request`
- `column_selection_revision`
- `sql_review_request`

### User Events

User messages must also be emitted into `conversation_events`.

The graph, not the UI, owns user-event emission.

The orchestrator is the correct ingress point because every thread turn passes through it and it already owns fresh user-turn detection.

## Artifact Model

Artifacts should live in one shared thread-level namespace, not per-node subtrees.

Suggested shape:

```python
artifacts["files"][artifact_id] = {
    "kind": "figure" | "code" | "sql" | "table" | "text",
    "producer": "qa" | "generate_code" | "executor" | "rag_db_qa",
    "mime": "...",
    "summary": "...",
    "created_at": "...",
    "content": ...,
}
```

Guidelines:

- artifact IDs are immutable references
- payloads are immutable snapshots
- new versions create new artifact IDs
- events reference artifacts by `artifact_id`

### Large Payload Handling

Checkpoint:

- manifests
- references
- summaries
- hashes
- small payloads

Move out of state when large:

- PNG bytes
- large code files
- large tables
- bulky retrieval payloads

Use a storage abstraction from the start, backed initially by local thread-scoped runtime storage.

## Output Snapshot Contract

`output` remains intentionally narrow.

Suggested sanctioned fields include:

- `qa_response`
- `generated_code`
- `text`
- `error`
- `generated_sql`
- `prepared_sql_candidate`
- possibly a small set of latest display artifact IDs

Do not mirror the full event log into `output`.

`output` is a latest-result snapshot for orchestration and lightweight UI status, not the full conversation history.

## Memory and History Behavior

The redesign must not silently change orchestrator or worker prompt behavior.

### Prompt History

Prompt history remains on raw `messages`.

- worker nodes continue to read bounded windows from raw `messages`
- planner prompt construction continues to read recent raw turns plus workflow state
- semantic events do not implicitly change prompts

### Control Memory

Planner and orchestrator memory remain where they are today:

- `planner.memory`
- `orchestrator`
- `meta`

This includes:

- active user goal
- conversation intent summary
- unresolved constraints
- clarification state
- recurrence / stagnation state
- workflow milestone and blocker state

### Semantic History

`conversation_events` is side-band by default:

- human-readable
- export-friendly
- audit-friendly
- optionally prompt-capable in the future
- not the default prompt substrate now

### Future Compaction

The long-term model should allow compaction of old raw `messages` while keeping:

- recent prompt-accurate raw messages
- structured planner memory
- durable semantic history in `conversation_events`

## UI Design

### Default Chat Surface

Default chat should render only user-facing content:

- `user`
- final `assistant`
- active `clarification`
- active `review_request`
- final code/figure/table outputs

### Trace / Debug Surface

Operational events should be available separately:

- tool calls
- tool results
- retrieval details
- retries
- routing decisions
- resolved review steps
- superseded or stale intermediate events

### Tool Trace Presentation

Default UI should not dump raw tool payloads inline.

Instead:

- show a compact summary such as "Used weather tool"
- expose detailed tool events only in trace/debug view

### Rendering Contract

The UI should render through one projection helper, not by iterating directly over raw `messages`.

Suggested helper:

- `build_display_history(state, mode="default" | "trace")`

## Shared Helper Layer

Centralize semantic event and artifact handling in shared helpers.

Suggested API surface:

- `append_event(state, event)`
- `store_artifact(state, artifact)`
- `derive_output_snapshot(state)`
- `build_display_history(state, mode=...)`

Nodes should describe what happened, not handcraft view-specific structures.

## Node Responsibilities

### Orchestrator

Emits only sparse system-level events:

- `routing_decision`
- `workflow_ended`
- `guard_triggered`

It should not generate assistant-style prose events.

### QA

Should emit:

- `assistant`
- `clarification`
- `tool_call`
- `tool_result`

while keeping raw `messages` for actual prompt/runtime continuity.

### Generate Code / Execute Code

Must validate the event model across:

- assistant summaries
- code artifacts
- execution lifecycle
- figure / table / text artifacts
- errors and retries

### DB-RAG

Uses the same shared event infrastructure with DB-RAG-specific event types.

DB-RAG remains special in content, not in event plumbing.

## Migration Strategy

Use staged migration with one canonical target.

### Phase 1

Introduce:

- `artifacts["conversation_events"]`
- `artifacts["files"]`
- schema version fields
- shared event/artifact helpers

### Phase 2

First real migration slice:

- `qa`
- `generate_code`
- `execute_code`

This slice is large enough to validate:

- user/assistant events
- clarification
- tool calls/results
- code artifacts
- execution artifacts
- errors

### Phase 3

Switch UI rendering to a resolver:

- prefer semantic events when present
- fall back to legacy `messages` during migration

### Phase 4

Migrate:

- `rag_db_qa`
- human review nodes
- export pipeline

### Phase 5

Remove direct UI dependence on raw `messages`.

Retain raw `messages` as runtime trace and prompt substrate.

## Non-Goals

This redesign does not, by itself:

- change the current orchestrator routing policy
- replace raw `messages` as the prompt-history source
- fully redesign planner memory
- introduce distributed artifact storage on day one
- force prompt behavior to read semantic events

## Main Benefits

- cleaner separation of concerns
- stable UI model
- stable export model
- better auditability
- less node-specific rendering logic
- less coupling between provider-native messages and product behavior
- future support for trace panels, compaction, richer attachments, and multiple client frontends

## Recommended Next Step

Write an implementation plan for:

1. schema and helper introduction
2. thread bootstrap changes
3. first migration slice (`qa + generate_code + execute_code`)
4. UI projection switch
5. DB-RAG and review-node migration
