# Latency Diagnostics Design

Date: 2026-05-08
Status: Proposed

## Goal

Add diagnostics-first latency tracing for the API-backed agent workflow so we can identify the true bottleneck before changing routing, DB-RAG behavior, model selection, or streaming behavior.

The first slice is measurement only. It must not change runtime routing decisions, fallback policy, DB-RAG handoff behavior, planner prompts, model choices, or user-visible output outside debug mode.

## Problem

The app uses API calls for model inference, so local GPU capacity is not the relevant bottleneck. The perceived latency can come from several layers:

- sequential orchestration and planner calls
- planner context construction and prompt size
- DB-RAG retrieval, reranking, intent resolution, SQL generation, or review preparation
- final QA or generation model latency
- missing or delayed streaming/progress feedback in the UI

The repo already has DB-RAG stage timing through `utils.performance.collect_timings()` and `utils.performance.timing_stage()`, and Streamlit can display `meta["db_rag_timing"]` in debug mode. The missing layer is broader workflow and planner timing.

Without that evidence, optimization would be guesswork and could accidentally violate the runtime-canonical orchestrator policy in `docs/superpowers/specs/orchestrator-gating-policy.md`.

## Design Principles

1. Measure before optimizing.
2. Reuse the existing `utils.performance` timing primitives.
3. Keep DB-RAG's existing timing surface stable.
4. Store diagnostics in `meta`, not planner memory or task memory.
5. Show latency diagnostics only in debug UI.
6. Do not change routing, fallback, recurrence, DB-RAG gating, or dataset handoff behavior in the first slice.
7. Make timing records compact, JSON-safe, and bounded.
8. Use the measurements to choose one later optimization branch.

## Existing Timing Surface

Existing primitives:

- `utils/performance.py`
  - `collect_timings()`
  - `timing_stage(stage, **metadata)`

Existing DB-RAG storage:

```python
meta["db_rag_timing"] = {
    "node": "rag_db_qa",
    "stages": [...],
}
```

Existing UI:

- `streamlit_app.py` displays DB-RAG timing inside debug mode.

This design extends that pattern rather than introducing a new metrics framework.

## Scope

### In Scope

- Add workflow node elapsed timings.
- Add planner internal timings.
- Store broad workflow timings separately from `db_rag_timing`.
- Display the new timing table in Streamlit debug mode.
- Add tests for timing storage and planner timing records.
- Define a small benchmark set for manual latency evaluation.

### Out Of Scope

- Deterministic fast-path routing.
- Planner prompt trimming.
- DB-RAG retrieval or reranker tuning.
- Model tier changes.
- Streaming or progress UX changes.
- Persistent telemetry, external metrics backends, or always-on logging.
- Any change to `orchestrator-gating-policy.md`.

## Timing Layers

### Layer A: Workflow Node Timing

Capture elapsed time for each graph node at the graph wrapper boundary.

Recommended integration point:

- `graph/builder.py`
  - `_run_and_mark(node_name, fn)`

Suggested stage names:

- `node.orchestrator`
- `node.qa`
- `node.rag_db_qa`
- `node.generate_code`
- `node.execute_code`
- `node.clarification`
- `node.tool_handler`
- `node.error_handler`
- human review node names as applicable

This layer answers: which node consumed the most wall-clock time?

### Layer B: Planner Internal Timing

Capture planner substage timing inside:

- `graph/nodes/orchestrator/planner.py`
  - `llm_plan_next_action(...)`

Suggested stage names:

- `planner.total`
- `planner.runtime_state`
- `planner.action_mask`
- `planner.context_build`
- `planner.prompt_format`
- `planner.llm_invoke`
- `planner.parse_response`

This layer answers: when the orchestrator planner is slow, is the cost context building, prompt formatting, API latency, or response parsing?

## Storage Shape

Keep existing DB-RAG timing unchanged.

Add a separate top-level diagnostic key under `meta`:

```python
meta["workflow_timing"] = {
    "stages": [
        {
            "stage": "node.orchestrator",
            "elapsed_ms": 123.45
        },
        {
            "stage": "planner.llm_invoke",
            "elapsed_ms": 842.17,
            "actions": 8
        }
    ]
}
```

Bound the stage list to the latest 100 records, matching the existing DB-RAG timing bound.

Do not store prompt text, model responses, SQL, dataframe contents, schema payloads, retrieved chunks, or full user messages in timing records.

Allowed timing metadata should be compact and non-sensitive, such as:

- action count
- node name
- candidate count
- provider/model name only when already exposed in debug state
- boolean flags such as `used_planner`

## UI Surface

In Streamlit debug mode:

- Keep the existing `DB-RAG timing` expander.
- Add a separate `Workflow timing` expander.
- Render `meta["workflow_timing"]["stages"]` as a dataframe.

Do not display workflow timing outside debug mode.

Do not merge DB-RAG and workflow timing in the first slice. A merged view can be added later if the separate tables are insufficient.

## Benchmark Set

After instrumentation, run a fixed set of representative prompts and record stage timings.

Recommended prompts:

1. Simple conceptual QA.
2. Explicit DB-RAG database query.
3. DB-RAG follow-up on an active database thread.
4. Code or dataset analysis request.
5. Cancelled DB-RAG query followed by "continue previous query".
6. Ambiguous previous-query reference that should ask for clarification.

For each benchmark, record:

- selected route
- total workflow elapsed time
- slowest node
- planner total time
- planner LLM invoke time
- DB-RAG total time when applicable
- slowest DB-RAG stage when applicable
- whether debug timing is present and bounded

## Optimization Decision Tree

The first optimization should be chosen only after benchmark evidence exists.

If planner LLM time dominates:

- consider deterministic fast-path routing for policy-obvious turns
- consider a smaller/faster planner model

If planner context or prompt formatting dominates:

- trim planner context
- reduce repeated capability text or large recent-turn payloads

If DB-RAG retrieval/reranking dominates:

- reduce rerank candidate count
- skip reranking for high-confidence deterministic/schema matches
- cache stable schema or retrieval inputs

If final QA/generation model dominates:

- review model tiering by task type
- consider streaming and progress feedback

If UI wait dominates:

- add streaming or earlier progress events

## Validation

Tests should verify:

- `timing_stage()` remains a no-op without a collector.
- workflow timing records are stored under `meta["workflow_timing"]`.
- workflow timing records are bounded.
- DB-RAG timing remains stored under `meta["db_rag_timing"]`.
- planner timing includes the expected stage names when the LLM planner runs.
- Streamlit debug mode reads `workflow_timing` without requiring DB-RAG timing.

Manual validation should confirm:

- normal routes still behave the same as before instrumentation.
- DB-RAG review, cancel, and continuation behavior still follows the existing specs.
- no timing table is shown when debug mode is off.
- benchmark runs identify at least one dominant latency source before any optimization work begins.

## Follow-Up Work

After diagnostics are implemented and benchmarked, write a separate design or implementation plan for the first optimization branch. That follow-up must read the relevant specs under `docs/superpowers/specs/` before changing routing, fallback, gating, DB-RAG flow, or dataset handoff behavior.
