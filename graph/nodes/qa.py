from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..state import AgentState
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

    if question and not tool_results and not tool_requests and should_route_tools(question):
        recent_msgs = window_messages(state.get("messages", []), max_turns=3)
        routing_result = request_tools_for_question(llm, question, recent_messages=recent_msgs)

        # Clarification needed — required tool field is missing and can't be inferred.
        if routing_result.clarification_question:
            messages = list(state.get("messages", []))
            messages.append(AIMessage(content=routing_result.clarification_question))
            meta = dict(state.get("meta", {}))
            meta["awaiting_user_clarification"] = True
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
            return update_agent_state(updated_state, "qa", {"status": "done"})

        # Tool(s) identified — enqueue and wait for tool_handler.
        if routing_result.tool_requests:
            observations = list(state.get("observations", []))
            observations.append("qa: requested tools")
            updated_state = enqueue_tool_requester(
                {
                    **state,
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
    observations.append("qa: responded to user question")

    meta = dict(state.get("meta", {}))
    meta.pop("intent", None)

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
        },
    )
