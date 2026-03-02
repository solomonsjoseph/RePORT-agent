from __future__ import annotations

import json

from langchain_core.prompts import ChatPromptTemplate

from ..state import AgentState


TOOLS_CATALOG: list[dict[str, object]] = [
    {
        "tool_name": "query_weather",
        "server": "weather",
        "description": "Get weather for a city and optional date range (includes observation date/time).",
        "schema": {
            "city": "string (e.g., Boston)",
            "start_date": "optional YYYY-MM-DD",
            "end_date": "optional YYYY-MM-DD",
        },
    },
    {
        "tool_name": "get_weather_tips",
        "server": "weather",
        "description": "Get seasonal weather tips.",
        "schema": {"season": "spring|summer|autumn|winter"},
    },
    {
        "tool_name": "search",
        "server": "search",
        "description": "Run a web search (Tavily).",
        "schema": {"query": "string", "max_results": "int (default 5)"},
    },
    {
        "tool_name": "calculate",
        "server": "calculator",
        "description": "Evaluate a math expression.",
        "schema": {"expression": "string (e.g., 2 + 3 * 4)"},
    },
]


def latest_user_message(state: AgentState) -> str:
    messages = list(state.get("messages", []))
    for message in reversed(messages):
        if getattr(message, "type", None) == "human":
            return str(message.content or "")
    return ""


def format_tool_results(results: list[dict[str, object]]) -> str:
    formatted: list[str] = []
    for result in results:
        formatted.append(json.dumps(result, ensure_ascii=False, default=str))
    return "\n".join(formatted) if formatted else "none"


def format_tool_catalog(tools_catalog: list[dict[str, object]] | None = None) -> str:
    lines = []
    for tool in tools_catalog or TOOLS_CATALOG:
        schema = json.dumps(tool["schema"], ensure_ascii=False)
        lines.append(
            f"- {tool['tool_name']} (server={tool['server']}): {tool['description']} "
            f"schema={schema}"
        )
    return "\n".join(lines)


def is_tool_requested(state: AgentState) -> bool:
    agents = state.get("agents", {})
    for agent_state in agents.values():
        if agent_state.get("tool_requests"):
            return True
    return False


def parse_tool_requests(text: str) -> list[dict[str, object]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    requests = data.get("tool_requests", [])
    if not isinstance(requests, list):
        return []
    validated: list[dict[str, object]] = []
    for request in requests:
        if not isinstance(request, dict):
            continue
        tool_name = request.get("tool_name")
        payload = request.get("payload")
        if not tool_name or not isinstance(payload, dict):
            continue
        validated.append(
            {
                "tool_name": tool_name,
                "payload": payload,
            }
        )
    return validated


def request_tools_for_question(
    llm,
    question: str,
    tools_catalog: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    tool_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a tool routing assistant. Decide if the user question needs "
                "external tools. Use only the tools listed. If no tool is needed, "
                "return {{\"tool_requests\": []}}. Otherwise return JSON with tool requests "
                "in the form {{\"tool_requests\": [{{\"tool_name\": ..., \"payload\": {{...}}}}]}}. "
                "Each payload MUST include \"server\" and any required fields.",
            ),
            (
                "system",
                "Available tools:\n{tools_catalog}",
            ),
            ("human", "{question}"),
        ]
    )
    tool_response = llm.invoke(
        tool_prompt.format_prompt(
            tools_catalog=format_tool_catalog(tools_catalog),
            question=question,
        ).to_messages()
    )
    return parse_tool_requests(tool_response.content)
