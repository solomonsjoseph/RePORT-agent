from typing import Annotated, List, Literal, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

    output: dict
    next_action: str | None
    last_action: str | None
    observations: List[str]
    orchestrator: dict
    agents: dict

    # extensibility
    meta: dict  # free-form (retry counts, tool info, etc.)
