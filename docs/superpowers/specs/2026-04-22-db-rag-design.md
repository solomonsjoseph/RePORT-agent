# DB-RAG Design

## Routing Boundaries
- `qa` handles general explanation, conceptual questions, and non-database conversational turns.
- `rag_db_qa` handles database-grounded questions about records, cohorts, forms, columns, and subset-oriented follow-ups.
- `generate_code` remains the downstream epidemiologic analysis path and consumes an explicit dataset artifact.

## Conversational SQL Offer
- `rag_db_qa` always starts with semantic retrieval over the local DB-RAG assets and answers from retrieved metadata/context.
- After that answer, `rag_db_qa` offers optional read-only SQL extraction.
- SQL generation and execution happen only after a positive user confirmation.

## Dataset Artifacts
- Uploaded datasets are persisted under `runtime/datasets/<thread_id>/` as artifact entries instead of living only in Streamlit memory.
- SQL-derived subsets are persisted as separate Parquet artifacts with provenance, including the originating question and SQL.
- `artifacts.datasets` stores the registry and `artifacts.active_dataset_id` points at the newest active subset.

## Downstream Analysis Handoff
- If only one dataset artifact exists, analysis uses it directly.
- If both an uploaded dataset and an extracted subset exist, the app must ask explicitly which dataset to analyze.
- Generated code and execution resolve the selected dataset at runtime rather than capturing a static `df` at graph-build time.
