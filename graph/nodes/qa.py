from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..state import AgentState
from .state_helpers import get_agent_state, update_agent_state
from .tool_routing import (
    format_tool_results,
    latest_user_message,
    request_tools_for_question,
)


def qa_node(state: AgentState, llm) -> AgentState:
    qa_state = get_agent_state(state, "qa")
    question = latest_user_message(state)
    tool_requests = list(qa_state.get("tool_requests", []))
    tool_results = list(qa_state.get("tool_results", []))

    if question and not tool_results and not tool_requests:
        tool_requests = request_tools_for_question(llm, question)
        if tool_requests:
            observations = list(state.get("observations", []))
            observations.append("qa: requested tools")
            updated_state = {
                **state,
                "observations": observations,
            }
            return update_agent_state(
                updated_state,
                "qa",
                {
                    "status": "pending",
                    "tool_requests": tool_requests,
                },
            )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful epidemiology assistant. Answer the user's question "
                "directly and clearly. If the question requires analysis, explain the "
                "recommended steps without writing code unless requested.",
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
            messages=state.get("messages", []),
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
