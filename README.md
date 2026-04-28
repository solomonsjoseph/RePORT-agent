# RePORT-agent
A multi-agent framework for RePORT India studies.


## Orchestrator-driven workflow
The LangGraph workflow is orchestrator-driven: a lightweight planning node chooses
the next specialized agent based on the current state, recent observations, and
any human feedback. This makes the graph flexible and easy to extend with new
agents without hard-coding a strict pipeline.

### Planner mode
The orchestrator runs in hybrid mode by default: it lets the LLM propose the
next action and falls back to deterministic rules when the LLM response is not
usable.

### Available specialist nodes
- **Code Generator**: produces Python analysis code.
- **Executor**: runs generated code on the dataset.
- **Error Handler**: fixes code issues and retries execution.
- **Q&A**: answers general questions without code.
- **Human-in-the-loop**: review checkpoints before and after execution.
- **Tool Handler**: executes MCP tools requested by agents.

## Experimental tools
Tool integrations are under active development and may change frequently. The
documentation intentionally omits tool-specific setup details until the
interface stabilizes.

## Setup

This project is tested to work with Python `3.12.7`.

### Clone the repository
```bash
git clone -b dev-transit --single-branch https://github.com/xutao-wang/RePORT-agent.git
cd RePORT-agent
```

### Create and activate a Python virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
```

### Install required packages
```bash
pip install -r requirements.txt
```

### Build .env file (optional)
create `.env` file and add your API keys inside the `.env` so we do not have to enter API key every time. For example:
```
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY"
```

### Run the app
```bash
python -m streamlit run streamlit_app.py
```

### Activate Langraph
In the pop up webpage, enter the API key in the field and click submit. If `.env` was set up previously, click submit directly.

### RAG DB
If you want to use the DB-RAG feature, place the source files in this repo under:

- `local_data/db_rag_source/reviewed_annotated_json_files/`
- `local_data/db_rag_source/filtered_excel_files/`

Supported DB-RAG indexing models:

- `OpenAI/text-embedding-3-small`
- `Qwen/Qwen3-Embedding-4B`
- `Qwen/Qwen3-Embedding-8B`
- `voyage-4-large`

Build the DB-RAG assets from the repo root by selecting one of the supported indexing models:

```bash
python -m db_rag.build_index --indexing-model Qwen/Qwen3-Embedding-4B
```

To rebuild that model even if the index already exists, add `--rebuild`:

```bash
python -m db_rag.build_index --indexing-model Qwen/Qwen3-Embedding-4B --rebuild
```

Build behavior:

- `--indexing-model` is required.
- The selected indexing model is written to `.env` as `DB_RAG_EMBEDDING_MODEL=...`.
- Every run prints the active `.env` value so you can confirm which indexing model is primary for DB-RAG.
- If the selected model-specific Chroma index and manifest already exist, the command exits without rebuilding and prints the exact `--rebuild` command to rerun.
- A rebuild prints step-level progress while it loads source files, builds chunks, writes DuckDB, builds Chroma, and writes the manifest.

Relevant `.env` keys for DB-RAG:

```env
DB_RAG_EMBEDDING_MODEL=OpenAI/text-embedding-3-small
DB_RAG_RERANKER_MODEL=cohere/rerank-v3.5
DB_RAG_OPENROUTER_API_KEY=...
DB_RAG_OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
VOYAGE_API_KEY=...
```

Supported DB-RAG reranker models:

- `cohere/rerank-v3.5`
- `cohere/rerank-4-fast`
- `cohere/rerank-4-pro`
- `rerank-2.5`

Reranking is optional and does not require rebuilding the DB-RAG index.
When `DB_RAG_RERANKER_MODEL` is set, the LangGraph DB-RAG app path uses it for column reranking as well.
`voyageai` is included in `requirements.txt`, so no extra package install is required for Voyage support in this repo.

For OpenAI indexing:

```env
DB_RAG_EMBEDDING_MODEL=OpenAI/text-embedding-3-small
```

For Qwen indexing through OpenRouter, use one of:

```env
DB_RAG_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B
DB_RAG_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
```

For Voyage indexing and reranking:

```env
DB_RAG_EMBEDDING_MODEL=voyage-4-large
DB_RAG_RERANKER_MODEL=rerank-2.5
VOYAGE_API_KEY=...
```

This creates the shared DuckDB asset plus model-specific DB-RAG index assets in `runtime/db_rag/`.

To run a quick terminal demo against the DB-RAG assets:

```bash
python -m db_rag.quick_test "your question here"
```

If you omit the question, the script drops into interactive mode. If the runtime
assets are missing, it rebuilds them first.

To use DB-RAG in the Streamlit app:

1. Build the DB-RAG index with the embedding model you want to use.
2. Set `DB_RAG_EMBEDDING_MODEL` in `.env` to the same embedding model used for the built index.
3. Optionally set `DB_RAG_RERANKER_MODEL` to one of the supported rerankers.
4. Launch the app with `python -m streamlit run streamlit_app.py`.
5. In the sidebar under `DB-RAG Runtime`, confirm the app shows the expected embedding index and reranker.

Example `.env` for app usage:

```env
DB_RAG_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B
DB_RAG_RERANKER_MODEL=cohere/rerank-v3.5
DB_RAG_OPENROUTER_API_KEY=...
DB_RAG_OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Example `.env` for Voyage-backed app usage:

```env
DB_RAG_EMBEDDING_MODEL=voyage-4-large
DB_RAG_RERANKER_MODEL=rerank-2.5
VOYAGE_API_KEY=...
```

If `DB_RAG_RERANKER_MODEL` is unset, the app uses the default ChromaDB column ordering with reranking disabled.

To run the sibling-style retrieval benchmark against prebuilt DB-RAG assets:

```bash
python -m db_rag.benchmark.evaluate --benchmark db_rag/benchmark/data/retrieval_benchmark_100.csv --indexing-model Qwen/Qwen3-Embedding-4B
```

To run the benchmark with reranking enabled:

```bash
python -m db_rag.benchmark.evaluate --benchmark db_rag/benchmark/data/retrieval_benchmark_100.csv --indexing-model Qwen/Qwen3-Embedding-4B --reranker cohere/rerank-v3.5
```

Benchmark behavior:

- `--indexing-model` is required.
- The benchmark uses the prebuilt DuckDB, Chroma index, and manifest for the selected indexing model.
- The benchmark does not rebuild assets.
- If the selected assets are missing, it exits immediately and tells you to run `db_rag.build_index` first.
- Each benchmark run writes to `db_rag/benchmark/results/<tag>/` with:
  - `details.csv`
  - `summary.json`

To render the benchmark figure for a saved run:

```bash
python -m db_rag.benchmark.plot_results --run-dir db_rag/benchmark/results/<tag>
```

This writes `figure.png` into the same run directory.

### Notes:
- Synthetic demo data are included under `data/` folder.
