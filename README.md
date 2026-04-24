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

Build the DB-RAG assets from the repo root:

```bash
python -m db_rag.bootstrap --rebuild
```

Bootstrap behavior:

- Every rebuild call prints the active `DB_RAG_EMBEDDING_MODEL` and tells you to edit `.env` if you want to switch models later.
- If `DB_RAG_EMBEDDING_MODEL` is not set yet, the rebuild flow prompts you to choose one and writes that selection to `.env`.
- If the selected model index already exists, bootstrap exits without rebuilding and explains how to switch models by editing `.env`.
- If you want to rebuild that model anyway, for example because the source data changed, run:

```bash
python -m db_rag.bootstrap --rebuild --force
```

Relevant `.env` keys:

```env
DB_RAG_EMBEDDING_MODEL=OpenAI/text-embedding-3-small
DB_RAG_OPENROUTER_API_KEY=...
DB_RAG_OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

For OpenAI embeddings, use:

```env
DB_RAG_EMBEDDING_MODEL=OpenAI/text-embedding-3-small
```

For Qwen embeddings through OpenRouter, use one of:

```env
DB_RAG_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B
DB_RAG_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
```

This creates the shared DuckDB asset plus model-specific DB-RAG index assets in `runtime/db_rag/`.

To run a quick terminal demo against the DB-RAG assets:

```bash
python -m db_rag.quick_test "your question here"
```

If you omit the question, the script drops into interactive mode. If the runtime
assets are missing, it rebuilds them first.

### Notes:
- Synthetic demo data are included under `data/` folder.
