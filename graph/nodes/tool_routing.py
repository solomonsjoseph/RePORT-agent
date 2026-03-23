from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from ..state import AgentState
from utils.llm_response import coerce_text_content


TOOLS_CATALOG: list[dict[str, object]] = [
    {
        "tool_name": "query_weather",
        "server": "weather",
        "description": "Get weather for a city and optional date range (includes observation date/time).",
        "required_fields": {"city": "string (e.g., Boston)"},
        "optional_fields": {"start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"},
    },
    {
        "tool_name": "get_weather_tips",
        "server": "weather",
        "description": "Get seasonal weather tips.",
        "required_fields": {"season": "spring|summer|autumn|winter"},
        "optional_fields": {},
    },
    {
        "tool_name": "search",
        "server": "search",
        "description": "Run a web search (Tavily).",
        "required_fields": {"query": "string"},
        "optional_fields": {"max_results": "int (default 5)"},
    },
    {
        "tool_name": "calculate",
        "server": "calculator",
        "description": "Evaluate a math expression.",
        "required_fields": {"expression": "string (e.g., 2 + 3 * 4)"},
        "optional_fields": {},
    },
]

# Fast lexical gate to avoid routing unrelated questions through the tool-router
# LLM call. This prevents false clarification prompts (e.g., asking for weather
# city/file fields) when the user asks for dataset analysis/code generation.
_TOOL_KEYWORDS: dict[str, tuple[str, ...]] = {
    "query_weather": ("weather", "temperature", "forecast", "rain", "snow"),
    "get_weather_tips": ("weather tips", "season", "winter", "summer", "autumn", "spring"),
    "search": ("search", "look up", "find online", "web", "internet", "news"),
    "calculate": ("calculate", "compute", "what is", "evaluate", "math"),
}

_TOOL_ROUTING_SYSTEM = (
    "You are a tool routing assistant. Decide if the user question needs external tools.\n"
    "Each tool lists required_fields (must be populated) and optional_fields (omit if not mentioned).\n"
    "\n"
    "RULES:\n"
    "- If all required_fields can be filled from the question or recent context\n"
    "  (e.g. 'today' for a date, a city named in a prior message), return tool_requests JSON.\n"
    "- If a required_field is missing and CANNOT be reasonably inferred, return a\n"
    "  clarification_question asking the user for that ONE missing field only.\n"
    "- NEVER ask about optional_fields — simply omit them from the payload.\n"
    "- If no tool is needed, return {{\"tool_requests\": []}}.\n"
    "\n"
    "Output format — choose exactly one:\n"
    "  {{\"tool_requests\": [{{\"tool_name\": \"...\", \"payload\": {{\"server\": \"...\", ...}}}}]}}\n"
    "  {{\"clarification_question\": \"Which city would you like weather for?\"}}"
)


@dataclass
class ToolRoutingResult:
    """Discriminated result from :func:`request_tools_for_question`.

    Exactly one of ``tool_requests`` (non-empty) or ``clarification_question``
    (non-None) is set per result.  Both being empty/None means no tool is needed.
    """

    tool_requests: list[dict]
    clarification_question: str | None


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
        required = json.dumps(tool.get("required_fields", {}), ensure_ascii=False)
        optional = json.dumps(tool.get("optional_fields", {}), ensure_ascii=False)
        lines.append(
            f"- {tool['tool_name']} (server={tool['server']}): {tool['description']}\n"
            f"  required: {required}\n"
            f"  optional: {optional}"
        )
    return "\n".join(lines)


def is_tool_requested(state: AgentState) -> bool:
    agents = state.get("agents", {})
    for agent_state in agents.values():
        if agent_state.get("tool_requests"):
            return True
    return False


def should_route_tools(question: str) -> bool:
    """Heuristic pre-check for whether question is likely tool-oriented.

    We intentionally keep this conservative: if no tool-domain cues are present,
    skip tool-routing and let the primary node prompt handle the request.
    """
    normalized = (question or "").strip().lower()
    if not normalized:
        return False

    for keywords in _TOOL_KEYWORDS.values():
        if any(keyword in normalized for keyword in keywords):
            return True
    return False


def parse_tool_requests(
    text: str,
    tools_catalog: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    requests = data.get("tool_requests", [])
    if not isinstance(requests, list):
        return []

    # Build tool_name -> server mapping from catalog
    tool_to_server: dict[str, str] = {}
    for t in tools_catalog or TOOLS_CATALOG:
        name = str(t.get("tool_name") or "")
        server = str(t.get("server") or "")
        if name and server:
            tool_to_server[name] = server

    validated: list[dict[str, object]] = []
    for request in requests:
        if not isinstance(request, dict):
            continue

        tool_name = request.get("tool_name")
        payload = request.get("payload")

        if not tool_name or not isinstance(payload, dict):
            continue

        tool_name = str(tool_name)

        # Enforce / inject server
        payload = dict(payload)  # copy
        payload.setdefault("server", tool_to_server.get(tool_name))

        # If still missing (unknown tool), drop it
        if not payload.get("server"):
            continue

        validated.append({"tool_name": tool_name, "payload": payload})

    return validated


def request_tools_for_question(
    llm,
    question: str,
    recent_messages: list | None = None,
    tools_catalog: list[dict[str, object]] | None = None,
) -> ToolRoutingResult:
    """Decide whether an external tool is needed for *question*.

    Returns a :class:`ToolRoutingResult` that either carries ready-to-execute
    ``tool_requests`` *or* a targeted ``clarification_question`` to ask the user
    when a required tool field cannot be inferred from context.

    Args:
        llm: The language model to invoke for routing decisions.
        question: The latest user message text.
        recent_messages: Optional short window of prior messages (typically the
            last 3 turns) so the LLM can resolve conversational references such
            as pronouns or implicit locations.
        tools_catalog: Override the default :data:`TOOLS_CATALOG` (useful in tests).
    """
    tool_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _TOOL_ROUTING_SYSTEM),
            ("system", "Available tools:\n{tools_catalog}"),
            ("system", "Recent conversation context (for resolving references):"),
            MessagesPlaceholder("recent_messages", optional=True),
            ("human", "{question}"),
        ]
    )
    tool_response = llm.invoke(
        tool_prompt.format_prompt(
            tools_catalog=format_tool_catalog(tools_catalog),
            question=question,
            recent_messages=recent_messages or [],
        ).to_messages()
    )
    response_text = coerce_text_content(getattr(tool_response, "content", ""))

    try:
        data = json.loads(response_text)
    except (TypeError, json.JSONDecodeError):
        return ToolRoutingResult(tool_requests=[], clarification_question=None)

    clarification = data.get("clarification_question")
    if isinstance(clarification, str) and clarification.strip():
        return ToolRoutingResult(
            tool_requests=[],
            clarification_question=clarification.strip(),
        )

    tool_requests = parse_tool_requests(response_text, tools_catalog=tools_catalog)
    return ToolRoutingResult(tool_requests=tool_requests, clarification_question=None)
