# Latency Diagnostics Benchmark Checklist

Date: 2026-05-08

## Purpose

Run this checklist after workflow and planner timing are implemented. The goal is to identify the dominant latency source before making optimization changes.

## Preconditions

- Streamlit debug state is enabled.
- The app uses the same provider/model settings for every benchmark run.
- DB-RAG assets are ready for DB-RAG prompts.
- No routing, prompt, model, retrieval, reranker, or streaming optimization has been added yet.

## Results Table

| Case | Prompt | Selected route | Total workflow ms | Slowest workflow stage | Planner total ms | Planner LLM ms | DB-RAG total ms | Slowest DB-RAG stage | Notes |
|---|---|---:|---:|---|---:|---:|---:|---|---|
| 1 | Simple conceptual QA |  |  |  |  |  |  |  |  |
| 2 | Explicit DB-RAG database query |  |  |  |  |  |  |  |  |
| 3 | DB-RAG follow-up |  |  |  |  |  |  |  |  |
| 4 | Code or dataset analysis request |  |  |  |  |  |  |  |  |
| 5 | Cancelled DB-RAG query then "continue previous query" |  |  |  |  |  |  |  |  |
| 6 | Ambiguous previous-query reference |  |  |  |  |  |  |  |  |

## Prompts

1. Simple conceptual QA:
   - "What can this app help me do?"

2. Explicit DB-RAG database query:
   - "Query my database and identify columns for age, gender, diabetes status, and TB outcome among index cases."

3. DB-RAG follow-up:
   - After a DB-RAG query starts, ask: "Narrow that to index cases only."

4. Code or dataset analysis request:
   - "Analyze the selected dataset and summarize missingness by column."

5. Cancelled DB-RAG continuation:
   - Start the explicit DB-RAG query from case 2.
   - Cancel during review.
   - Ask: "continue previous query."

6. Ambiguous previous-query reference:
   - Create at least two DB-RAG query intents.
   - Ask: "continue that query."

## Decision Rule

Choose the next optimization branch from the measured bottleneck:

- Planner LLM dominates: design deterministic fast path or planner model tiering.
- Planner context/prompt dominates: design planner context trimming.
- DB-RAG retrieval/rerank dominates: design retrieval/reranker tuning.
- Final model dominates: design model tiering or streaming.
- UI wait dominates: design progress or streaming updates.
