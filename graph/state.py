from typing import Annotated, List, Literal, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

    generated_code: str | None
    output: str | None
    error: str | None
    figure_png: bytes
    # machine control
    run_status: Literal[
        "idle",
        "pending",
        "error",
        "done",
    ]

    human_decision: Literal[
        "approve",
        "regenerate",
    ]| None

    # extensibility
    meta: dict  # free-form (retry counts, tool info, etc.)

