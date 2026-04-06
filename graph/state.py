from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, List, TypedDict

try:
    from langgraph.graph.message import add_messages
except ModuleNotFoundError:  # pragma: no cover - import safety for test envs without langgraph
    def add_messages(left, right):
        return right

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

    output: dict
    artifacts: dict
    next_action: str | None
    last_action: str | None
    observations: List[str]
    orchestrator: dict
    planner: dict
    agents: dict
    node_data: dict

    # extensibility
    meta: dict  # free-form (retry counts, tool info, etc.)


class MetaKeys:
    """Documented string keys used in AgentState['meta'].

    Risk-5 fix: replaces magic string literals scattered across the codebase
    with named constants so typos surface as NameErrors and the full contract
    is visible in one place.

    Usage:
        meta.pop(MetaKeys.AWAITING_USER_CLARIFICATION, None)
    """
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
    # The clarification workflow kind, used by the clarification node to resume
    # the correct subworkflow.
    CLARIFICATION_KIND = "clarification_kind"
    # One-shot list of actions allowed to bypass loop guards after explicit
    # human instruction (e.g., regenerate code in review).
    LOOP_GUARD_BYPASS_ACTIONS = "loop_guard_bypass_actions"
    PROGRESS_SNAPSHOT = "progress_snapshot"
    PROGRESS_MADE_LAST_STEP = "progress_made_last_step"
    STAGNATION_COUNT = "stagnation_count"
    REPEATED_FAILURE_SIGNATURE = "repeated_failure_signature"
