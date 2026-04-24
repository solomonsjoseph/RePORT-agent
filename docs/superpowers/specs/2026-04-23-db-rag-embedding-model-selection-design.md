# DB-RAG Embedding Model Selection Design

## Goal

Add explicit embedding-model selection for `db_rag` so the project can:

- keep using OpenAI embeddings through the OpenAI API
- add `Qwen/Qwen3-Embedding-4B` and `Qwen/Qwen3-Embedding-8B` through OpenRouter
- build and reuse one Chroma index per embedding model
- keep one shared DuckDB database for unchanged source data
- guarantee that query-time embeddings always use the same model that built the selected index

## Problem

The current `db_rag` implementation hard-codes a single embedding model and a single Chroma storage path. That creates three limits:

1. switching embedding models requires overwriting the previous index
2. query-time model selection is not explicit
3. OpenAI and OpenRouter embedding providers cannot be routed cleanly from one DB-RAG configuration surface

The desired workflow is to build each embedding model once, keep the prebuilt indexes on disk, and switch the active runtime model explicitly without ambiguity.

## Design Principles

- `DB_RAG_EMBEDDING_MODEL` is the single source of truth for the active DB-RAG embedding model.
- DuckDB is shared because it depends on tabular source data, not embedding geometry.
- Chroma indexes are model-specific because stored vectors are only valid for the model that created them.
- OpenAI-backed and OpenRouter-backed embeddings use the same OpenAI-compatible client interface, but with different credentials and base URL routing.
- The runtime must never auto-pick an index from what happens to exist on disk.
- Readiness failures must explain which model is missing and how to build it.

## Supported Models

The first pass supports exactly these values for `DB_RAG_EMBEDDING_MODEL`:

- `OpenAI/text-embedding-3-small`
- `Qwen/Qwen3-Embedding-4B`
- `Qwen/Qwen3-Embedding-8B`

No reranker is added in this pass.

## Configuration

### Required Environment Variables

For all DB-RAG runs:

- `DB_RAG_EMBEDDING_MODEL`

For OpenAI embeddings:

- `OPENAI_API_KEY`

For Qwen embeddings through OpenRouter:

- `DB_RAG_OPENROUTER_API_KEY`

### Optional Environment Variables

- `DB_RAG_OPENROUTER_BASE_URL`
  - default: `https://openrouter.ai/api/v1`

### Provider Routing

- `OpenAI/...` model names use the normal OpenAI API key and default OpenAI base URL
- `Qwen/...` model names use the OpenRouter API key and OpenRouter base URL

The answer-generation LLM path remains unchanged. Only DB-RAG embedding calls use this routing.

## Storage Layout

Shared DuckDB:

- `runtime/db_rag/report.duckdb`

Per-model Chroma indexes:

- `runtime/db_rag/indexes/openai_text-embedding-3-small/`
- `runtime/db_rag/indexes/qwen_qwen3-embedding-4b/`
- `runtime/db_rag/indexes/qwen_qwen3-embedding-8b/`

Per-model manifests:

- `runtime/db_rag/manifests/openai_text-embedding-3-small.json`
- `runtime/db_rag/manifests/qwen_qwen3-embedding-4b.json`
- `runtime/db_rag/manifests/qwen_qwen3-embedding-8b.json`

Model directory names are derived from a deterministic sanitized model slug.

## Manifest Contract

Each manifest describes one model-specific index build. Example fields:

```json
{
  "embedding_model": "Qwen/Qwen3-Embedding-4B",
  "source_fingerprint": "abc123",
  "source_root": "local_data/db_rag_source",
  "duckdb_path": "runtime/db_rag/report.duckdb",
  "chroma_path": "runtime/db_rag/indexes/qwen_qwen3-embedding-4b",
  "manifest_path": "runtime/db_rag/manifests/qwen_qwen3-embedding-4b.json",
  "table_chunk_count": 18,
  "column_chunk_count": 742,
  "built_at": "2026-04-23T18:00:00Z"
}
```

The manifest exists to answer:

- which embedding model built this index
- whether the source data still matches
- where the matching Chroma index is stored

## Build Behavior

`python -m db_rag.bootstrap --rebuild` follows this order:

1. resolve and validate `DB_RAG_EMBEDDING_MODEL`
2. compute the current source fingerprint
3. ensure shared DuckDB exists for the current source fingerprint
4. resolve the manifest path and Chroma directory for the selected model
5. if the manifest exists and the source fingerprint matches and the Chroma directory exists, reuse the existing model-specific index
6. otherwise rebuild only the selected model-specific Chroma index and write the manifest

Rebuild rules:

- source data changed: rebuild DuckDB and the selected model index
- source data unchanged and selected model index exists: reuse
- source data unchanged and selected model index missing: build only that model index

The system does not delete other model-specific indexes during a rebuild of one selected model.

## Runtime Behavior

`DbRagService` resolves the active model from `DB_RAG_EMBEDDING_MODEL`, then:

1. validates that the model is supported
2. resolves the manifest path for that model
3. loads that manifest and matching Chroma directory
4. uses the same model to embed incoming user questions

This preserves the required invariant: query embeddings and stored vectors always come from the same embedding model.

If the selected model has no valid manifest or no matching Chroma directory, readiness returns a clear message instructing the operator to rebuild with that model.

## Code Structure

`db_rag/service.py` will own:

- supported model constants
- model slug helper
- provider-routing helper
- embedding client construction
- manifest and Chroma path resolution helpers
- readiness checks that reference the selected embedding model

`db_rag/bootstrap.py` will own:

- selected-model rebuild flow
- manifest write/update
- shared DuckDB rebuild logic
- model-specific Chroma creation/reuse logic

No change is required to the answer-generation model configuration in `db_rag/quick_test.py` or the Streamlit generation path.

## Error Handling

- unsupported `DB_RAG_EMBEDDING_MODEL`: fail fast with the supported values
- missing `OPENAI_API_KEY` for `OpenAI/...`: fail with a provider-specific message
- missing `DB_RAG_OPENROUTER_API_KEY` for `Qwen/...`: fail with a provider-specific message
- selected model index missing: readiness explains the exact rebuild command shape
- stale manifest fingerprint: rebuild the selected model index

## Non-Goals

- no interactive picker during build
- no reranker support
- no automatic fallback from one embedding model to another
- no duplicated DuckDB per embedding model
- no change to chunk text generation beyond what is already in `db_rag.bootstrap`

## Testing

Manual verification should cover:

1. build with `OpenAI/text-embedding-3-small`
2. build with `Qwen/Qwen3-Embedding-4B`
3. switch back to `OpenAI/text-embedding-3-small` without rebuilding and confirm readiness succeeds
4. switch to `Qwen/Qwen3-Embedding-8B` before it exists and confirm readiness reports the missing model-specific index
5. rebuild `Qwen/Qwen3-Embedding-8B` and confirm the new index is stored without deleting the 4B and OpenAI indexes
