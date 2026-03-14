from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..state import AgentState, MetaKeys
from .state_helpers import enqueue_tool_requester, get_agent_state, update_agent_state
from .tool_routing import (
    format_tool_results,
    latest_user_message,
    request_tools_for_question,
    should_route_tools,
)
from utils.message_window import window_messages


def qa_node(state: AgentState, llm, context: str = "") -> AgentState:
    qa_state = get_agent_state(state, "qa")
    question = latest_user_message(state)
    tool_requests = list(qa_state.get("tool_requests", []))
    tool_results = list(qa_state.get("tool_results", []))
    awaiting_tool_clarification = bool(qa_state.get("awaiting_tool_clarification"))

    should_attempt_tool_routing = should_route_tools(question) or awaiting_tool_clarification

    if question and not tool_results and not tool_requests and should_attempt_tool_routing:
          # If there is a stored pending question (original request before a
        # clarification round), merge it with the user's follow-up so the tool
        # routing LLM receives unambiguous context instead of a bare answer.
        pending_question = (state.get("meta") or {}).get(MetaKeys.PENDING_QUESTION)
        if pending_question:
            effective_question = f"{pending_question}\n\nUser clarification: {question}"
            recent_msgs = window_messages(state.get("messages", []), max_turns=5)
        else:
            effective_question = question
            recent_msgs = window_messages(state.get("messages", []), max_turns=3)
 
        routing_result = request_tools_for_question(llm, effective_question, recent_messages=recent_msgs)
        # Clarification needed — required tool field is missing and can't be inferred.
        if routing_result.clarification_question:
            messages = list(state.get("messages", []))
            messages.append(AIMessage(content=routing_result.clarification_question))
            meta = dict(state.get("meta", {}))

            meta[MetaKeys.AWAITING_USER_CLARIFICATION] = True
            # Store the original request and who handles the answer so the
            # orchestrator can route back without a full new-turn reset.
            meta[MetaKeys.PENDING_QUESTION] = effective_question
            meta[MetaKeys.CLARIFICATION_RETURN_NODE] = "qa"
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
            meta = dict(state.get("meta") or {})
            meta.pop(MetaKeys.PENDING_QUESTION, None)
            meta.pop(MetaKeys.CLARIFICATION_RETURN_NODE, None)
            observations = list(state.get("observations", []))
            observations.append("qa: requested tools")
            updated_state = enqueue_tool_requester(
                {
                    **state,
                    "meata":meta,
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

    windowed = window_messages(state.get("messages", []), max_turns=10)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful data scientist. Answer the user's question "
                "directly and clearly. If the question requires analysis, explain the "
                "recommended steps without writing code unless requested.",
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
    messages = list(state.get("messages", []))
    messages.append(AIMessage(content=response.content))

    observations = list(state.get("observations", []))
    meta = dict(state.get("meta", {}))

    # If the QA LLM produced a question on a tool-related query, treat it as a
    # clarification request so the orchestrator preserves context on the next turn.
    # This handles the case where tool routing fell through (returned neither tools
    # nor a structured clarification_question) but the LLM naturally asked for a
    # missing required field (e.g. "Which city would you like weather for?").
    if should_attempt_tool_routing and response.content.strip().endswith("?"):
        meta[MetaKeys.AWAITING_USER_CLARIFICATION] = True
        meta[MetaKeys.PENDING_QUESTION] = question
        meta[MetaKeys.CLARIFICATION_RETURN_NODE] = "qa"
        awaiting_tool_clarification_for_agent = True
        observations.append("qa: asked clarification (llm path)")
    else:
        meta.pop(MetaKeys.INTENT, None)
        meta.pop(MetaKeys.PENDING_QUESTION, None)
        meta.pop(MetaKeys.CLARIFICATION_RETURN_NODE, None)
        awaiting_tool_clarification_for_agent = False
        observations.append("qa: responded to user question")

    updated_state = {
        **state,
        "messages": messages,
        "qa_response": response.content,
        "meta": meta,
        "observations": observations,
    }
    output = dict(updated_state.get("output") or {})
    output["qa_response"] = response.content
    updated_state["output"] = output
    return update_agent_state(
        updated_state,
        "qa",
        {
            "status": "done",
            "response": response.content,
            "awaiting_tool_clarification": awaiting_tool_clarification_for_agent,
        },
    )
