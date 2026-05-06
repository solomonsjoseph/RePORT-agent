# RePORT-agent
An AI Agent for RePORT India studies (demo).

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
- **Q&A**: subagent answers general questions.
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

### Build .env file
create `.env` file and add your API keys inside the `.env` so we do not have to enter API key every time. For example:
```env
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
ANTHROPIC_API_KEY="YOUR_ANTHROPIC_API_KEY"
```

### Run the app
```bash
python -m streamlit run streamlit_app.py
```
 ### Activate system
In the pop up webpage, enter the API key in the field and click submit. If `.env` was set up previously, choose API provide and click submit directly.

### Clean saved app datasets
Datasets saved by the app are stored in `runtime/datasets/`. These files are
inside the repository and are not cleaned by macOS temp cleanup. To remove all
saved app dataset artifacts:

```bash
rm -rf runtime/datasets/*
```


## RAG DB

DB-RAG uses retrieval-augmented generation (RAG) to let users query a prebuilt database in natural language. Currently, DB-RAG access depends on the OpenRouter and Voyage APIs, configured with the following environment variables in `.env`:

```env
DB_RAG_OPENROUTER_API_KEY=""
VOYAGE_API_KEY=""
```

If user needs to use local resource to acees models, additional modification is needed.

### Get started
To use the DB-RAG feature, place the source files in this repo under:

- `RePORT-agent/local_data/db_rag_source/reviewed_annotated_json_files/*.json`
- `RePORT-agent/local_data/db_rag_source/filtered_excel_files/*.xlsx`

Scripts realated to RAG development is under:
`RePORT-agent/local_data/db_rag/`

Supported DB-RAG indexing models:

- `OpenAI/text-embedding-3-small`
- `Qwen/Qwen3-Embedding-4B`
- `Qwen/Qwen3-Embedding-8B`
- `voyage-4-large`

### Build indexing
Build the DB-RAG assets from the repo root by selecting one of the supported indexing models. This creates the shared DuckDB asset plus model-specific DB-RAG index assets in `runtime/db_rag/`.:

```bash
python -m db_rag.build_index --indexing-model Qwen/Qwen3-Embedding-4B
```

To rebuild that model if the index already exists, add `--rebuild`:

```bash
python -m db_rag.build_index --indexing-model Qwen/Qwen3-Embedding-4B --rebuild
```

Build behavior:

- `--indexing-model` is required.
- The selected indexing model is written to `.env` as `DB_RAG_EMBEDDING_MODEL=...`.
- Every run prints the active `.env` value so you can confirm which indexing model is primary for DB-RAG.
- If the selected model-specific Chroma index and manifest already exist, the command exits without rebuilding and prints the exact `--rebuild` command to rerun.
- A rebuild prints step-level progress while it loads source files, builds chunks, writes DuckDB, builds Chroma, and writes the manifest.

Example for relevant `.env` keys for DB-RAG:

```env
DB_RAG_EMBEDDING_MODEL=OpenAI/text-embedding-3-small
DB_RAG_RERANKER_MODEL=cohere/rerank-v3.5
DB_RAG_SELECTION_MODEL=gpt-4o-mini
```

Supported DB-RAG reranker models:

- `cohere/rerank-v3.5`
- `cohere/rerank-4-fast`
- `cohere/rerank-4-pro`
- `voyage/rerank-2.5`

Reranking is optional and does not require rebuilding the DB-RAG index.
When `DB_RAG_RERANKER_MODEL` is set, the LangGraph DB-RAG app path uses it for column reranking as well.

`DB_RAG_SELECTION_MODEL` controls the lightweight OpenAI model used for structured
DB-RAG column ranking before human review. The current default is `gpt-4o-mini`.
This selector reuses `OPENAI_API_KEY` by default. If OpenAI is unavailable, the
app falls back to deterministic retrieval-backed column review.


### Quick test on RAG
To run a quick terminal demo against the DB-RAG assets:

```bash
python -m db_rag.quick_test "your question here"
```

If you omit the question, the script drops into interactive mode. If the runtime
assets are missing, it rebuilds them first.

### Benchmark RAG

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

### Use DB-RAG in streamlit
To use DB-RAG in the Streamlit app:

1. Build the DB-RAG index with the embedding model you want to use.
2. Set `DB_RAG_EMBEDDING_MODEL` in `.env` to the same embedding model used for the built index.
3. Optionally set `DB_RAG_RERANKER_MODEL` to one of the supported rerankers.
4. Optionally set `DB_RAG_SELECTION_MODEL` if you want to override the default `gpt-4o-mini` structured selector.
5. Launch the app with `python -m streamlit run streamlit_app.py`.
6. In the sidebar under `DB-RAG Runtime`, confirm the app shows the expected embedding index and reranker.

Example `.env` for app usage, for OpenRouter based model:

```env
DB_RAG_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B
DB_RAG_RERANKER_MODEL=cohere/rerank-v3.5
DB_RAG_SELECTION_MODEL=gpt-4o-mini
DB_RAG_OPENROUTER_API_KEY=...
OPENAI_API_KEY=...
```

Example `.env` for Voyage-backed app usage:

```env
DB_RAG_EMBEDDING_MODEL=voyage-4-large
DB_RAG_RERANKER_MODEL=voyage/rerank-2.5
DB_RAG_SELECTION_MODEL=gpt-4o-mini
VOYAGE_API_KEY=...
OPENAI_API_KEY=...
```

If `DB_RAG_RERANKER_MODEL` is unset, the app uses the default ChromaDB column ordering with reranking disabled.
If `DB_RAG_SELECTION_MODEL` is unset, the app defaults to `gpt-4o-mini` for structured DB-RAG column ranking.

### Notes:
- Synthetic demo data are included under `data/` folder.
