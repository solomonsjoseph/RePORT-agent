from __future__ import annotations

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..state import AgentState
from .state_helpers import update_agent_state


def qa_node(state: AgentState, llm) -> AgentState:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful epidemiology assistant. Answer the user's question "
                "directly and clearly. If the question requires analysis, explain the "
                "recommended steps without writing code unless requested.",
            ),
            MessagesPlaceholder("messages"),
        ]
    )
    response = llm.invoke(
        prompt.format_prompt(
            messages=state.get("messages", []),
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
    return update_agent_state(
        updated_state,
        "qa",
        {
            "status": "done",
            "response": response.content,
        },
    )
