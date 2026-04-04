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
git clone -b dev-test-macOS --single-branch https://github.com/xutao-wang/RePORT-agent.git
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

### Build the sandbox image
Python analysis now runs inside a short-lived Docker sandbox by default. Build the
runner image once before starting the app:

```bash
docker build -t report-agent-sandbox:latest -f tools/sandbox/Dockerfile .
```

Optional execution settings:

```bash
export EXECUTION_MODE=docker
export SANDBOX_IMAGE=report-agent-sandbox:latest
export EXECUTION_TIMEOUT_SEC=20
export SANDBOX_MEMORY_MB=512
export SANDBOX_CPU_LIMIT=1.0
```

Local trusted mode:

```bash
# Explicit trusted local execution (no Docker isolation)
export EXECUTION_MODE=trusted_local
```

Use `trusted_local` only for trusted operators or local development. It keeps
the app workflow but runs approved Python directly on the host process without
container isolation.

If you need additional analysis packages, add them to
`tools/sandbox/requirements.txt` and rebuild the image. Runtime package
installation inside the sandbox is intentionally unsupported.

### Run the app
```bash
python -m streamlit run streamlit_app.py
```

Synthetic demo data are included under `data/` folder.
