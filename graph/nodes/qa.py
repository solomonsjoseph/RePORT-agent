from __future__ import annotations

import json

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..state import AgentState, MetaKeys
from ..workflow_config import QA_RECENT_TURNS, TOOL_ROUTER_RECENT_TURNS
from utils.code_parser import extract_python_code
from .code_guardrails import code_fingerprint, is_executable_python
from .state_helpers import (
    clear_clarification_meta,
    enqueue_tool_requester,
    get_agent_state,
    set_clarification_meta,
    update_agent_state,
)
from .tool_routing import (
    format_tool_results,
    latest_user_message,
    request_tools_for_question,
    should_route_tools,
)
from utils.message_window import window_messages
from utils.llm_response import coerce_text_content

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


def qa_node(
    state: AgentState,
    llm,
    context: str = "",
    question_override: str | None = None,
) -> AgentState:
    qa_state = get_agent_state(state, "qa")
    question = question_override if question_override is not None else latest_user_message(state)
    tool_requests = list(qa_state.get("tool_requests", []))
    tool_results = list(qa_state.get("tool_results", []))
    prompt_messages = (
        _replace_latest_human_message(list(state.get("messages", [])), question)
        if question_override is not None
        else list(state.get("messages", []))
    )

    should_attempt_tool_routing = should_route_tools(question)

    if question and not tool_results and not tool_requests and should_attempt_tool_routing:
        effective_question = question
        recent_msgs = window_messages(prompt_messages, max_turns=TOOL_ROUTER_RECENT_TURNS)

        routing_result = request_tools_for_question(llm, effective_question, recent_messages=recent_msgs)
        # Clarification needed — required tool field is missing and can't be inferred.
        if routing_result.clarification_question:
            messages = list(state.get("messages", []))
            messages.append(AIMessage(content=routing_result.clarification_question))
            meta = set_clarification_meta(
                state.get("meta", {}),
                return_node="qa",
                kind="qa_tool",
                pending_question=effective_question,
            )
            output = dict(state.get("output") or {})
            output["qa_response"] = routing_result.clarification_question
            observations = list(state.get("observations", []))
            observations.append("qa: asked clarification for missing required tool field")
            updated_state = {
                **state,
                "messages": messages,
                "meta": meta,
                "output": output,
                "observations": observations,
            }
            return update_agent_state(
                updated_state,
                "qa",
                {
                    "status": "done",
                    "awaiting_tool_clarification": True,
                },
            )

        # Tool(s) identified — clear pending clarification state and enqueue.
        if routing_result.tool_requests:
            meta = clear_clarification_meta(state.get("meta") or {})
            observations = list(state.get("observations", []))
            observations.append("qa: requested tools")
            updated_state = enqueue_tool_requester(
                {
                    **state,
                    "meta":meta,
                    "observations": observations,
                },
                "qa",
            )
            return update_agent_state(
                updated_state,
                "qa",
                {
                    "status": "pending",
                    "tool_requests": routing_result.tool_requests,
                    "awaiting_tool_clarification": False,
                },
            )

    windowed = window_messages(prompt_messages, max_turns=QA_RECENT_TURNS)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful data scientist. Answer the user's question "
                "directly and clearly. If the question requires analysis, explain the "
                "recommended steps without writing code unless requested.\n"
                "Return valid JSON with exactly these keys: "
                '{{"answer": string, "needs_clarification": boolean, "clarification_question": string|null}}.\n'
                'Set "needs_clarification" to true only when the request is blocked by one '
                "specific missing piece of required information. Do not use clarification for "
                "optional next steps or offers like asking whether the user wants code.\n"
                'When "needs_clarification" is true, put the single follow-up question in '
                '"clarification_question" and set "answer" to an empty string.\n'
                'When "needs_clarification" is false, put the full response in "answer" and set '
                '"clarification_question" to null.',
            ),
            (
                "system",
                "Dataset context (if available):\n{context}",
            ),
            (
                "system",
                "Tool results (if any):\n{tool_results}",
            ),
            MessagesPlaceholder("messages"),
        ]
    )
    response = llm.invoke(
        prompt.format_prompt(
            messages=windowed,
            context=context or "No dataset or schema provided.",
            tool_results=format_tool_results(tool_results),
        ).to_messages()
    )
    raw_response_text = coerce_text_content(response.content)
    parsed_answer, explicit_clarification, clarification_question = _parse_structured_qa_response(
        raw_response_text
    )
    response_text = clarification_question if explicit_clarification else (parsed_answer or raw_response_text)
    messages = list(state.get("messages", []))
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
    code = extract_python_code(response_text)
    if is_executable_python(code):
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
        },
    )
