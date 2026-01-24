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

## MCP tool demos
MCP tools are available to agents via the tool handler. Each specialized agent
can request tools by adding entries to its own `tool_requests` list inside
`agents[<agent_name>]`. The tool handler executes those requests and stores
results under that agent's `tool_results`.

### MCP search tool
Use the `search_context` tool to search dataset context.

```python
state = {
    "agents": {
        "qa": {
            "tool_requests": [
                {
                    "tool_name": "search_context",
                    "payload": {"query": "hiv", "context": "<dataset context>"},
                }
            ]
        }
    }
}
```

### MCP custom tool
Use `custom_echo` for a custom payload demo or `calculate` for arithmetic.

```python
state = {
    "agents": {
        "qa": {
            "tool_requests": [
                {
                    "tool_name": "custom_echo",
                    "payload": {"question": "report cohort size"},
                }
            ]
        }
    }
}
```

### MCP server registry
Register MCP servers in `tools/servers_config.json`:

```json
{
  "mcpServers": {
    "calculator": {
      "command": "python",
      "args": ["tools/calculator_server.py"],
      "transport": "stdio"
    },
    "weather": {
      "command": "python",
      "args": ["tools/weather_server.py"],
      "transport": "stdio"
    },
    "search": {
      "command": "python",
      "args": ["tools/search_server.py"],
      "transport": "stdio"
    }
  }
}
```

If a tool request includes a `server` name, the tool handler will resolve the
server config from `servers_config.json` and return it in the tool result payload.

See `tools/calculator_server.py`, `tools/weather_server.py`, and
`tools/search_server.py` for standalone FastMCP server examples that match the
MCP tool style.
