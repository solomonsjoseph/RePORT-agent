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

### Notes:
- Synthetic demo data are included under `data/` folder.
