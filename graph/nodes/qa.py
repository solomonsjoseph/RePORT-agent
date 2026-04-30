from __future__ import annotations

import json
from functools import lru_cache

from langchain_core.messages import AIMessage

from ..conversation_events import (
    append_conversation_event,
    build_assistant_event,
    build_clarification_event,
    build_tool_call_event,
    build_tool_result_event,
)
from ..state import AgentState, MetaKeys
from ..workflow_config import QA_RECENT_TURNS
from utils.code_parser import extract_python_code
from utils.dataset_artifacts import build_dataset_context, get_active_dataset_artifact
from utils.llm_response import coerce_text_content
from utils.message_window import window_messages
from .code_guardrails import code_fingerprint, is_executable_python
from .state_helpers import clear_clarification_meta, get_agent_state, set_clarification_meta, update_agent_state
from .orchestrator.state_logic import _user_message_hash
from .tool_routing import latest_user_message, should_route_tools

NODE_NAME = "qa"
NODE_CAPABILITY = (
    "Handle direct user-facing Q&A in natural language, including factual and explanatory "
    "requests, conversation, and tool-assisted information tasks such as search, weather, "
    "and calculator queries. Prefer this over code generation unless the user explicitly "
    "wants code or dataset/programmatic work."
)


def _parse_structured_qa_response(text: str) -> tuple[str | None, bool, str | None]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None, False, None

    if not isinstance(data, dict):
        return None, False, None

    answer = data.get("answer")
    needs_clarification = data.get("needs_clarification")
    clarification_question = data.get("clarification_question")

    normalized_answer = answer.strip() if isinstance(answer, str) else None
    normalized_question = (
        clarification_question.strip() if isinstance(clarification_question, str) else None
    )
    explicit_clarification = needs_clarification is True and bool(normalized_question)
    return normalized_answer, explicit_clarification, normalized_question


def _replace_latest_human_message(messages: list, content: str) -> list:
    updated_messages = list(messages)
    for idx in range(len(updated_messages) - 1, -1, -1):
        message = updated_messages[idx]
        if getattr(message, "type", None) != "human":
            continue
        updated_messages[idx] = message.__class__(content=content)
        return updated_messages
    return updated_messages


def _build_qa_system_prompt(context: str) -> str:
    normalized_context = context or "No dataset or schema provided."
    return (
        "You are a helpful data scientist. Answer the user's question directly and clearly. "
        "If the question requires analysis, explain the recommended steps without writing code "
        "unless requested.\n"
        "If the request is dataset-specific analytical setup or column-role mapping for uploaded "
        "or local data, do not ask for dataset-specific column-role mapping here; that belongs "
        "to the code-generation workflow.\n"
        "Use the available MCP tools when they materially improve the answer. If a tool requires "
        "one specific missing field that is not present in the conversation, do not guess and do "
        "not call the tool yet. Instead, ask a single clarification question.\n"
        "When including math, use Markdown math delimiters compatible with Streamlit: inline math "
        "with $...$ and display math with $$...$$. Do not use plain parentheses around LaTeX "
        "commands.\n"
        "Return valid JSON with exactly these keys: "
        '{"answer": string, "needs_clarification": boolean, "clarification_question": string|null}.\n'
        'Set "needs_clarification" to true only when the request is blocked by one specific '
        "missing piece of required information. Do not use clarification for optional next steps "
        'or offers like asking whether the user wants code.\n'
        'When "needs_clarification" is true, put the single follow-up question in '
        '"clarification_question" and set "answer" to an empty string.\n'
        'When "needs_clarification" is false, put the full response in "answer" and set '
        '"clarification_question" to null.\n'
        "Dataset context (if available):\n"
        f"{normalized_context}"
    )


def _serialize_tool_output(result) -> str:
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except TypeError:
        return str(result)


@lru_cache(maxsize=1)
def _build_mcp_tools():
    from langchain_core.tools import StructuredTool

    from tools.mcp_pool import call_server_tool

    def query_weather(
        city: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> str:
        """Get weather for a city and optional date range."""
        return _serialize_tool_output(
            call_server_tool(
                "weather",
                "query_weather",
                city=city,
                start_date=start_date,
                end_date=end_date,
            )
        )

    def get_weather_tips(season: str) -> str:
        """Get seasonal weather tips."""
        return _serialize_tool_output(
            call_server_tool(
                "weather",
                "get_weather_tips",
                season=season,
            )
        )

    def search(query: str, max_results: int = 5) -> str:
        """Run a web search."""
        return _serialize_tool_output(
            call_server_tool(
                "search",
                "search",
                query=query,
                max_results=max_results,
            )
        )

    def calculate(expression: str) -> str:
        """Evaluate a math expression."""
        return _serialize_tool_output(
            call_server_tool(
                "calculator",
                "calculate",
                expression=expression,
            )
        )

    return [
        StructuredTool.from_function(query_weather),
        StructuredTool.from_function(get_weather_tips),
        StructuredTool.from_function(search),
        StructuredTool.from_function(calculate),
    ]


def _create_qa_agent(llm, *, context: str):
    from langchain.agents import create_agent

    return create_agent(
        model=llm,
        tools=_build_mcp_tools(),
        system_prompt=_build_qa_system_prompt(context),
        name="qa_agent",
    )


def _extract_agent_response_text(agent_result) -> str:
    if isinstance(agent_result, dict):
        messages = list(agent_result.get("messages", []))
        for message in reversed(messages):
            if getattr(message, "type", None) == "ai":
                text = coerce_text_content(getattr(message, "content", ""))
                if text:
                    return text
    return coerce_text_content(agent_result)


def _extract_agent_trace_messages(agent_result, input_messages: list) -> list:
    if not isinstance(agent_result, dict):
        return []

    agent_messages = list(agent_result.get("messages", []))
    if not agent_messages:
        return []

    if len(agent_messages) >= len(input_messages):
        candidate = agent_messages[len(input_messages):]
        if candidate:
            return [message for message in candidate if getattr(message, "type", None) != "human"]

    return [message for message in agent_messages if getattr(message, "type", None) != "human"]


def _current_user_turn_hash(state: AgentState) -> str | None:
    meta = dict(state.get("meta") or {})
    return str(meta.get(MetaKeys.LAST_USER_MESSAGE_HASH) or _user_message_hash(state) or "") or None


def _iter_tool_calls(message) -> list[tuple[str, dict]]:
    additional_kwargs = dict(getattr(message, "additional_kwargs", {}) or {})
    tool_calls = additional_kwargs.get("tool_calls") or []
    parsed_calls: list[tuple[str, dict]] = []
    if not isinstance(tool_calls, list):
        return parsed_calls

    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        tool_name = (
            tool_call.get("name")
            or tool_call.get("tool")
            or ((tool_call.get("function") or {}).get("name") if isinstance(tool_call.get("function"), dict) else None)
        )
        if not isinstance(tool_name, str) or not tool_name.strip():
            continue
        raw_args = tool_call.get("args")
        if raw_args is None and isinstance(tool_call.get("function"), dict):
            raw_args = tool_call["function"].get("arguments")
        if isinstance(raw_args, str):
            try:
                parsed_args = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed_args = {"raw": raw_args}
        elif isinstance(raw_args, dict):
            parsed_args = raw_args
        else:
            parsed_args = {}
        if not isinstance(parsed_args, dict):
            parsed_args = {"raw": parsed_args}
        parsed_calls.append((tool_name.strip(), parsed_args))
    return parsed_calls


def _tool_message_name(message) -> str:
    name = getattr(message, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()
    additional_kwargs = dict(getattr(message, "additional_kwargs", {}) or {})
    for key in ("name", "tool", "tool_name"):
        value = additional_kwargs.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    tool_call_id = getattr(message, "tool_call_id", None)
    if isinstance(tool_call_id, str) and tool_call_id.strip():
        return tool_call_id.strip()
    return "tool"


def _append_qa_semantic_events(
    state: AgentState,
    trace_messages: list,
    response_text: str,
    *,
    explicit_clarification: bool,
) -> AgentState:
    user_turn_hash = _current_user_turn_hash(state)
    updated_state = state
    for message in trace_messages:
        message_type = getattr(message, "type", None)
        if message_type == "ai":
            for tool_name, tool_args in _iter_tool_calls(message):
                updated_state = append_conversation_event(
                    updated_state,
                    build_tool_call_event(
                        actor="qa",
                        user_turn_hash=user_turn_hash,
                        tool=tool_name,
                        args=tool_args,
                    ),
                )
        elif message_type == "tool":
            updated_state = append_conversation_event(
                updated_state,
                build_tool_result_event(
                    actor="qa",
                    user_turn_hash=user_turn_hash,
                    tool=_tool_message_name(message),
                    text=coerce_text_content(getattr(message, "content", "")),
                    artifact_id=None,
                ),
            )

    if explicit_clarification:
        event = build_clarification_event(
            actor="qa",
            user_turn_hash=user_turn_hash,
            text=response_text,
        )
    else:
        event = build_assistant_event(
            actor="qa",
            user_turn_hash=user_turn_hash,
            text=response_text,
        )
    return append_conversation_event(updated_state, event)


def qa_node(
    state: AgentState,
    llm,
    context: str = "",
    question_override: str | None = None,
) -> AgentState:
    if isinstance(context, dict) and context.get("runtime_datasets"):
        context = build_dataset_context(get_active_dataset_artifact(state))
    elif callable(context):
        context = context(state)

    question = question_override if question_override is not None else latest_user_message(state)
    prompt_messages = (
        _replace_latest_human_message(list(state.get("messages", [])), question)
        if question_override is not None
        else list(state.get("messages", []))
    )
    should_attempt_tool_routing = should_route_tools(question)

    windowed = window_messages(prompt_messages, max_turns=QA_RECENT_TURNS)
    agent = _create_qa_agent(llm, context=context or "No dataset or schema provided.")
    agent_result = agent.invoke({"messages": windowed})
    raw_response_text = _extract_agent_response_text(agent_result)

    parsed_answer, explicit_clarification, clarification_question = _parse_structured_qa_response(
        raw_response_text
    )
    response_text = clarification_question if explicit_clarification else (parsed_answer or raw_response_text)
    messages = list(state.get("messages", []))
    trace_messages = _extract_agent_trace_messages(agent_result, windowed)
    if trace_messages:
        messages.extend(trace_messages)
    else:
        messages.append(AIMessage(content=response_text))

    observations = list(state.get("observations", []))
    meta = dict(state.get("meta", {}))

    if explicit_clarification:
        clarification_kind = "qa_tool" if should_attempt_tool_routing else "qa_followup"
        meta = set_clarification_meta(
            meta,
            return_node="qa",
            kind=clarification_kind,
            pending_question=question,
        )
        awaiting_tool_clarification_for_agent = True
        observations.append("qa: asked clarification (structured qa response)")
    else:
        meta = clear_clarification_meta(meta)
        awaiting_tool_clarification_for_agent = False
        observations.append("qa: responded to user question")

    updated_state = {
        **state,
        "messages": messages,
        "qa_response": response_text,
        "meta": meta,
        "observations": observations,
    }
    output = dict(updated_state.get("output") or {})
    output["qa_response"] = response_text
    updated_state = _append_qa_semantic_events(
        updated_state,
        trace_messages,
        response_text,
        explicit_clarification=explicit_clarification,
    )
    code = extract_python_code(response_text)
    if not explicit_clarification and is_executable_python(code):
        output["generated_code"] = code
        meta[MetaKeys.CURRENT_CODE_HASH] = code_fingerprint(code)
        meta.pop(MetaKeys.FINAL_APPROVED_CODE_HASH, None)
        meta.pop(MetaKeys.EXECUTION_TICKET_HASH, None)
        meta.pop(MetaKeys.ERROR_RECOVERY_ACTIVE, None)
        agents = dict(updated_state.get("agents", {}))
        review_state = dict(agents.get("human_review", {}))
        review_state["before_run_decision"] = None
        review_state["approved_code_hash"] = None
        agents["human_review"] = review_state
        updated_state["agents"] = agents
    updated_state["output"] = output
    updated_state["meta"] = meta
    return update_agent_state(
        updated_state,
        "qa",
        {
            "status": "done",
            "response": response_text,
            "awaiting_tool_clarification": awaiting_tool_clarification_for_agent,
            "tool_requests": [],
            "tool_results": [],
        },
    )
