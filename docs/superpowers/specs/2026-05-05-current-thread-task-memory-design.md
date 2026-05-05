# Current-Thread Task Memory Design

Date: 2026-05-05
Status: Proposed

## Goal

Add a current-thread memory layer that lets the system resolve follow-up references such as "that query", "add gender too", "rerun it", "show me the SQL", and "analyze that subset" without treating every post-completion turn as a fresh request.

This design optimizes first for follow-up correctness. It does not introduce cross-session memory, user profiles, external vector stores, or long-term persistence beyond the existing checkpointed graph state.

## Problem

The app already has several state surfaces:

- `messages`: raw LangGraph and LangChain runtime trace
- `artifacts["conversation_events"]`: semantic transcript and audit events
- `artifacts["files"]`: immutable payload artifacts such as SQL, code, reviewed selections, and figures
- `artifacts["datasets"]`: uploaded and generated datasets
- `agents[...]`: live node workflow state
- `planner["memory"]`: shallow planner summary
- `meta`: control and routing flags
- `output`: latest visible result snapshot

These layers do not yet provide a reusable completed-work registry. After a workflow completes, the system often clears live pointers correctly, but then loses a structured handle that later turns can reference.

The DB-RAG SQL workflow shows the problem clearly. After reviewed SQL execution completes, `pending_sql_candidate_artifact_id`, `approved_column_selection_artifact_id`, and related active pointers are cleared. That is correct because those fields mean "resume pending SQL/review workflow", but the next user turn can no longer reliably refer to the completed SQL query or generated subset.

The same issue will appear in QA and code-generation workflows:

- "explain that more" should reference a prior QA answer
- "make that plot log scale" should reference a prior code-analysis task
- "analyze that subset" should reference a DB-RAG-generated dataset
- "what SQL did you use?" should reference a completed SQL extraction

## Design Principles

1. Keep live workflow state separate from completed reusable memory.
2. Define task lifecycle deterministically from node/workflow facts.
3. Use LLMs for semantic labeling and reference resolution, not for task boundary truth.
4. Store compact memory cards, not raw payloads.
5. Keep immutable payloads in `artifacts`.
6. Make reference resolution current-thread-only for the first slice.
7. Validate all resolver outputs deterministically before routing.
8. Avoid phrase-heavy heuristic routing. Use deterministic checks only for exact IDs, cache hits, missing memory, and safety validation.
9. Use opaque internal task IDs with human-friendly display ordinals.
10. Keep memory out of node prompts unless a reference has been resolved for the current turn.

## State Model

Add a top-level `memory` key to `AgentState`.

Suggested shape:

```python
{
    "completed_tasks": {
        "task_3f8a9c21": {
            "task_id": "task_3f8a9c21",
            "display_ordinal": 3,
            "kind": "db_rag_sql_extraction",
            "label": "diabetes index-case subset",
            "source_question": "Subset index cases with diabetes.",
            "goal_text": "Subset index cases with diabetes.",
            "summary": "Reviewed SQL was executed and saved as a subset dataset.",
            "artifact_refs": {
                "selection_artifact_id": "sel-art-...",
                "sql_candidate_artifact_id": "sql-art-...",
                "dataset_artifact_id": "subset-..."
            },
            "event_refs": {
                "user_event_id": "evt-...",
                "completion_event_id": "evt-..."
            },
            "parent_task_id": None,
            "relationship_to_parent": None,
            "provenance": {
                "producer_node": "rag_db_qa",
                "provider": "openai",
                "model": "gpt-...",
                "execution_mode": None
            },
            "status": "completed",
            "created_at": "2026-05-05T00:00:00+00:00",
            "completed_at": "2026-05-05T00:00:00+00:00",
            "last_enriched_at": None
        }
    },
    "failed_tasks": {},
    "task_order": ["task_3f8a9c21"],
    "last_task_id": "task_3f8a9c21",
    "last_task_id_by_kind": {
        "db_rag_sql_extraction": "task_3f8a9c21"
    },
    "last_failed_task_id_by_kind": {},
    "last_reference_resolution": {
        "user_message_hash": "...",
        "result": {
            "label": "resolved",
            "task_id": "task_3f8a9c21",
            "relationship": "revision",
            "intended_action": "add_fields",
            "confidence": "high",
            "needs_reference": False,
            "reason": "Short diagnostic reason."
        }
    },
    "pending_reference_clarification": None
}
```

`task_id` values are opaque and stable. User-facing displays may show `display_ordinal` and `label`, such as "Task 3: diabetes index-case subset".

`failed_tasks` is reserved for failed-but-reusable error contexts such as "fix that error". The first implementation slice may define the schema and helpers while leaving population of failed task cards to a later error-recovery slice.

## Layer Boundaries

The memory layer is not a replacement for existing state surfaces.

- `agents[...]` owns live workflow state: pending reviews, pending clarification gates, active execution tickets, and current node-specific status.
- `memory.completed_tasks` owns completed reusable work that future turns can reference.
- `artifacts["files"]` owns immutable payloads.
- `artifacts["datasets"]` owns dataset registry entries.
- `artifacts["conversation_events"]` owns transcript and audit events.
- `output` owns only latest visible result fields.
- `meta` owns routing and control flags.

Completed tasks must refer to artifacts and events by ID. They must not embed full SQL, code, dataframe contents, figures, or long transcripts.

Active task shells belong in `agents[...]`, not in `memory.completed_tasks`. For example, a DB-RAG revision may keep this live provenance while the review workflow is still in progress:

```python
agents["rag_db_qa"]["active_task"] = {
    "task_id": "task_9d41f02a",
    "parent_task_id": "task_3f8a9c21",
    "relationship_to_parent": "revision"
}
```

The completed card is written to `memory.completed_tasks` only after the revised workflow reaches a stable result.

## Task Definition

A task is one user-intended unit of work that produces a reusable result.

A task is not:

- one LangGraph node execution
- one assistant message
- one artifact
- one Streamlit rerun

A task starts from a user goal, may pass through multiple nodes, tools, reviews, and artifacts, and ends when the system produces a stable result that later turns may refer to.

Examples:

- `qa_answer`: a direct QA answer to a user question
- `db_rag_metadata_answer`: a DB-RAG metadata/schema answer that may also offer extraction
- `db_rag_sql_extraction`: a reviewed DB-RAG SQL extraction that saves a subset dataset
- `code_analysis`: generated code plus execution/final result for an analysis request

Follow-up turns that change a previous result should create a new linked task instead of mutating the completed task.

Example:

```python
{
    "task_id": "task_9d41f02a",
    "display_ordinal": 8,
    "kind": "db_rag_sql_extraction",
    "source_question": "add gender too",
    "parent_task_id": "task_3f8a9c21",
    "relationship_to_parent": "revision",
    "status": "completed"
}
```

Limited task-card metadata may be enriched after creation. Editable enrichment fields are:

- `label`
- `summary`
- `tags`
- `last_enriched_at`

Provenance fields are immutable after task creation:

- `task_id`
- `display_ordinal`
- `kind`
- `source_question`
- `goal_text`
- `artifact_refs`
- `event_refs`
- `parent_task_id`
- `relationship_to_parent`
- `created_at`
- `completed_at`
- `status`

## Task Lifecycle Ownership

Task lifecycle is deterministic and node-owned.

The system should not ask an LLM whether a task completed when the workflow already knows.

Initial lifecycle rules:

- `qa_answer`
  - completed when the QA node emits a final assistant answer for a user turn
- `db_rag_metadata_answer`
  - completed when DB-RAG emits a metadata answer and extraction prompt
  - this completed answer task may coexist with an active `pending_extraction_opt_in` gate in `agents["rag_db_qa"]`
- `db_rag_sql_extraction`
  - started when DB-RAG enters extraction flow
  - completed when reviewed SQL execution succeeds and saves a subset dataset
- `code_analysis`
  - started when code generation begins from an analysis request
  - completed when final review approves the generated result

The first implementation slice should support the schema and helpers generically, then populate memory for `db_rag_sql_extraction` first. QA and generated-code task completion can be added as separate slices.

Artifact-inspection answers should create linked `qa_answer` tasks when the answer is substantive and user-visible. For example, "what SQL did you use?" should answer from the SQL artifact and write a `qa_answer` task with `parent_task_id` pointing to the original `db_rag_sql_extraction` task and `relationship_to_parent="inspect_artifact"`.

## LLM Responsibilities

LLMs may enrich and resolve memory. They must not own the authoritative task lifecycle.

Allowed LLM responsibilities:

- generate a short task `label`
- generate a compact task `summary`
- infer semantic tags if needed
- resolve a user follow-up against completed task cards

Disallowed LLM responsibilities:

- deciding whether a state-machine workflow completed
- deciding whether a pending review pointer is valid
- bypassing artifact existence checks
- routing to an action whose required artifacts are missing

Task completion must write a usable deterministic fallback label and summary without waiting for a separate LLM call. LLM enrichment is optional and lazy:

```python
label = "DB-RAG SQL extraction: subset index cases with diabetes"
summary = "Completed DB-RAG SQL extraction and saved dataset subset-a13f."
```

The resolver may batch-enrich basic cards before resolution if needed, but task completion must not fail or slow down because enrichment is unavailable.

## Reference Resolution

Reference resolution runs before normal dataset/code/DB-RAG routing and before planner fallback, after the orchestrator has handled higher-priority live pending workflows.

Priority order:

1. pending human review or interrupt
2. pending clarification or DB-RAG opt-in
3. pending deterministic workflow continuation
4. memory reference resolver
5. normal deterministic routing predicates
6. planner fallback

The planner may see the validated reference result as context, but the planner is not the primary resolver.

Input to the resolver should be compact:

```python
{
    "user_message": "add gender too",
    "latest_task_id": "task_3f8a9c21",
    "completed_tasks": [
        {
            "task_id": "task_a10c0b91",
            "kind": "qa_answer",
            "label": "TB outcome variable explanation",
            "source_question": "What does OUTCOME mean?",
            "summary": "Explained OUTCOME coding."
        },
        {
            "task_id": "task_b22e13dd",
            "kind": "code_analysis",
            "label": "Kaplan-Meier plot by treatment group",
            "source_question": "Plot survival by treatment group.",
            "summary": "Generated Python code and a survival plot."
        },
        {
            "task_id": "task_3f8a9c21",
            "kind": "db_rag_sql_extraction",
            "label": "diabetes index-case subset",
            "source_question": "Subset index cases with diabetes.",
            "summary": "Generated reviewed SQL and saved a subset dataset."
        }
    ]
}
```

Initial candidate selection should pass the latest 12 task cards plus any task explicitly mentioned by task ID, display ordinal, or artifact ID. The resolver should inspect task cards only during initial resolution. It should not inspect full SQL, code, figures, schemas, or dataset contents until after a task has been selected and validated.

The resolver output must be structured:

```python
{
    "label": "resolved",
    "task_id": "task_3f8a9c21",
    "relationship": "revision",
    "intended_action": "add_fields",
    "confidence": "high",
    "needs_reference": false,
    "reason": "User asks to add a field to the latest extraction task."
}
```

Allowed `label` values:

- `resolved`
- `new_task`
- `ambiguous`
- `unknown`

Allowed `relationship` values:

- `revision`
- `rerun`
- `explain`
- `inspect_artifact`
- `use_as_input`
- `compare`

`unknown` results should include `needs_reference`. If `needs_reference=True`, route to clarification. If `needs_reference=False`, route as a fresh request.

## Resolver Guardrails

The resolver should not run for every turn.

Skip resolver when:

- there are no completed tasks
- a live pending workflow must handle the turn first
- the latest user message hash matches a cached `last_reference_resolution`
- the user references an exact known task ID or artifact ID that can be resolved deterministically

Use an LLM resolver when:

- there are completed task cards
- no live pending workflow has priority
- the turn may depend on prior context

Do not use large raw payloads in the resolver prompt. Pass compact task cards first. Only load full artifacts after a task has been resolved and validated.

The resolver should live in a generic memory package, not under DB-RAG service code:

```text
graph/memory/
  schema.py
  task_store.py
  reference_resolver.py
  validation.py
```

The resolver should use a separate lightweight model configuration when available:

```text
MEMORY_RESOLVER_PROVIDER
MEMORY_RESOLVER_MODEL
```

If no resolver-specific config is available, it may fall back to the active app provider/model.

## Deterministic Validation

Every resolver output must be validated before routing.

Validation rules:

- `resolved.task_id` must exist in `memory.completed_tasks`
- `relationship` must be supported by the task kind
- required artifact IDs for the intended action must exist
- artifact IDs must point to artifacts of the expected kind
- `ambiguous` must ask a clarification question with candidate task labels
- `unknown` must not silently bind to an arbitrary task
- `unknown + needs_reference=True` must ask clarification
- `unknown + needs_reference=False` may route as a fresh request
- `new_task` clears the active reference and routes normally
- if the user explicitly selected a task ID or artifact ID, the resolver cannot override that direct address with `new_task`

Reference resolution should write:

```python
memory["last_reference_resolution"] = {
    "user_message_hash": "...",
    "result": validated_result,
}
```

The full resolution object belongs in `memory`. For routing convenience, the orchestrator may also write compact derived keys into `meta`, such as:

```python
meta["resolved_task_id"] = "task_3f8a9c21"
meta["resolved_task_relationship"] = "use_as_input"
```

These `meta` values are derived caches. If they conflict with `memory["last_reference_resolution"]`, memory wins.

## Reference Clarification

Ambiguous reference resolution and `unknown + needs_reference=True` should use the existing clarification mechanism with a new kind:

```python
MetaKeys.CLARIFICATION_KIND = "memory_reference_resolution"
MetaKeys.CLARIFICATION_RETURN_NODE = "orchestrator"
```

Candidate state should live under `memory`, while `meta` carries only generic clarification routing keys:

```python
memory["pending_reference_clarification"] = {
    "status": "awaiting_reply",
    "user_message_hash": "...",
    "original_user_message": "rerun it",
    "candidates": [
        {
            "task_id": "task_3f8a9c21",
            "display_ordinal": 3,
            "kind": "db_rag_sql_extraction",
            "label": "diabetes index-case subset",
            "hint": "Saved dataset subset-a13f"
        }
    ]
}
```

The clarification prompt should include compact task-kind and artifact/result hints, not full payloads. The user's answer may reference a display ordinal, task ID, or label text.

## Routing Behavior

The orchestrator should route using both the latest user message and the validated reference resolution.

Examples:

- `resolved db_rag_sql_extraction + revision`
  - route to `rag_db_qa`
  - DB-RAG creates a new extraction task linked to the parent task
  - opens column review again

- `resolved db_rag_sql_extraction + inspect_artifact`
  - route to `rag_db_qa`
  - DB-RAG answers directly from the SQL candidate or dataset artifact

- `resolved db_rag_sql_extraction + use_as_input`
  - route to `generate_code`
  - set `meta["analysis_dataset_id"]` to the referenced dataset artifact for that turn
  - do not overwrite `artifacts["active_dataset_id"]` globally unless the user explicitly selects it

- `resolved code_analysis + revision`
  - route to `generate_code`
  - code generation uses prior code/artifacts as context

- `resolved qa_answer + explain`
  - route to `qa`
  - QA uses the prior answer event/task summary as context

- `ambiguous`
  - route to clarification
  - ask the user which prior task they mean

- `new_task`
  - ignore memory reference for routing
  - run normal orchestrator policy

Memory should enter node prompts only after reference resolution has selected a task for the current turn. Fresh requests should not receive broad completed-task history by default.

## DB-RAG First Slice

The first implementation slice should focus on DB-RAG SQL extraction because it is the current concrete failure mode.

On successful reviewed SQL execution:

1. Preserve existing behavior that clears live pending review pointers.
2. Create a completed `db_rag_sql_extraction` task card.
3. Link the task card to:
   - approved column-selection artifact
   - SQL candidate artifact
   - generated subset dataset artifact
4. Update `last_task_id` and `last_task_id_by_kind`.
5. Keep `thread_status="completed"` and `active_thread=False`.

This avoids reusing live workflow pointers as memory.

For a follow-up like "add gender too":

1. Resolver identifies the last DB-RAG SQL extraction as a `revision`.
2. Orchestrator routes to `rag_db_qa`.
3. DB-RAG creates an `agents["rag_db_qa"]["active_task"]` shell linked to the parent task.
4. DB-RAG loads the parent task card and artifacts.
5. DB-RAG creates a new intent linked to the parent task.
6. DB-RAG opens a new column review.
7. On completion, DB-RAG writes a new completed task with `parent_task_id`.

For "what SQL did you use?":

1. Resolver identifies the SQL extraction task as `inspect_artifact`.
2. DB-RAG loads the SQL candidate artifact.
3. DB-RAG answers directly without reopening review.
4. DB-RAG writes a linked `qa_answer` completed task for the artifact-inspection answer.

For "analyze that subset":

1. Resolver identifies the SQL extraction task as `use_as_input`.
2. Orchestrator routes to `generate_code`.
3. The referenced dataset artifact becomes `meta["analysis_dataset_id"]` for the turn.

## Export Behavior

Thread export should include task memory as structured metadata.

Add `memory.json` to the export ZIP:

```text
memory.json = completed task cards, failed task cards, task order, and last reference resolution
conversation.json = transcript/event history
artifacts/files = payload artifacts
datasets = dataset artifact manifests and files
```

`memory.json` must not duplicate full SQL/code/dataset payloads. It should reference artifact IDs.

## Testing Requirements

Use test-first development for implementation.

Add unit tests for memory helpers:

- initializing memory on old states
- completing a task updates `completed_tasks`, `task_order`, `last_task_id`, and `last_task_id_by_kind`
- completed task cards reject non-JSON payloads
- task cards store artifact references without embedding artifact payloads
- cached reference resolution is reused for the same user message hash
- task IDs are opaque and display ordinals are assigned monotonically
- failed task containers initialize on old states
- editable enrichment fields can be updated while immutable provenance fields remain stable

Add resolver tests:

- exact task ID resolves without LLM
- valid LLM `resolved` output is accepted
- missing task ID is rejected
- missing required artifact is rejected
- `ambiguous` output routes to clarification
- `unknown + needs_reference=True` routes to clarification
- `unknown + needs_reference=False` routes as fresh
- resolver is skipped when a live pending workflow exists
- resolver sees the latest 12 task cards plus exact direct-address matches
- resolver prompt uses task cards only and does not include artifact payloads

Add DB-RAG tests:

- SQL execution completion writes a `db_rag_sql_extraction` completed task
- live DB-RAG pointers are still cleared after SQL completion
- "what SQL did you use?" resolves to the completed task and answers from SQL artifact
- "add gender too" resolves to the completed task and opens a new column review
- "analyze that subset" resolves to the dataset artifact and routes to code generation
- "what SQL did you use?" creates a linked `qa_answer` task

Add export tests:

- thread export includes `memory.json`
- `memory.json` contains task references but not full artifact payloads

## Non-Goals

- Cross-session memory
- External vector database memory
- User profile memory
- Full transcript embedding
- Letting LLMs decide workflow completion
- Keeping old pending DB-RAG pointers alive after completion
- Replacing `conversation_events` or `artifacts`
- Populating QA and code-analysis tasks in the first implementation slice
- Streamlit Recent Work UI in the first implementation slice

## Open Implementation Notes

- `AgentState` should add `memory: dict`, while helper code should also tolerate older checkpoints without this key.
- The resolver prompt should use compact task cards and a strict JSON output contract.
- The first slice should not require embeddings. Semantic resolution can be LLM-only over a small card list.
- If completed tasks grow large, pass only the latest 12 task cards plus exact direct-address matches.
- Backend memory cards should include display fields from day one. Streamlit can expose a read-only Recent Work expander in a later slice.
