# RePORT-agent
A multi-agent framework for RePORT India studies.

## Requirements
Install the Python dependencies listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Orchestrator-driven workflow
The LangGraph workflow is orchestrator-driven: a lightweight planning node chooses
the next specialized agent based on the current state, recent observations, and
any human feedback. This makes the graph flexible and easy to extend with new
agents without hard-coding a strict pipeline.

### Planner modes
The orchestrator supports three planner modes via `planner_mode` in state:
- `rules`: deterministic policy routing only
- `llm`: LLM-only routing
- `hybrid`: LLM routing with rule-based fallback guardrails

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
