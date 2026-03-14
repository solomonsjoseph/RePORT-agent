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


class MetaKeys:
    """Documented string keys used in AgentState['meta'].

    Risk-5 fix: replaces magic string literals scattered across the codebase
    with named constants so typos surface as NameErrors and the full contract
    is visible in one place.

    Usage:
        meta[MetaKeys.INTENT] = "code"
        meta.pop(MetaKeys.AWAITING_USER_CLARIFICATION, None)
    """
    INTENT = "intent"
    ERROR_ITERATIONS = "error_iterations"
    CURRENT_CODE_HASH = "current_code_hash"
    AWAITING_USER_CLARIFICATION = "awaiting_user_clarification"
    TOOL_REQUEST_QUEUE = "tool_request_queue"
    WORKFLOW_TRACE = "workflow_trace"
    LAST_USER_MESSAGE_HASH = "last_user_message_hash"
    # Set by any node that asks a clarification question.
    # Stores the original user request so it can be reconstructed on re-entry.
    PENDING_QUESTION = "pending_question"
    # Which node should receive control when the user answers the clarification.
    CLARIFICATION_RETURN_NODE = "clarification_return_node"
