from __future__ import annotations

from langchain_core.messages import AIMessage

from ..state import AgentState, MetaKeys
from .qa import qa_node
from .state_helpers import (
    clear_clarification_meta,
    enqueue_tool_requester,
    set_clarification_meta,
    update_agent_state,
)
from .tool_routing import latest_user_message, request_tools_for_question
from utils.message_window import window_messages


def _resume_qa_tool_clarification(state: AgentState, llm) -> AgentState:
    question = latest_user_message(state)
    pending_question = (state.get("meta") or {}).get(MetaKeys.PENDING_QUESTION)
    if pending_question:
        effective_question = f"{pending_question}\n\nUser clarification: {question}"
        recent_msgs = window_messages(state.get("messages", []), max_turns=5)
    else:
        effective_question = question
        recent_msgs = window_messages(state.get("messages", []), max_turns=3)

    routing_result = request_tools_for_question(llm, effective_question, recent_messages=recent_msgs)
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
        observations.append("clarification: asked follow-up for missing tool field")
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

    if routing_result.tool_requests:
        meta = clear_clarification_meta(state.get("meta") or {})
        observations = list(state.get("observations", []))
        observations.append("clarification: resolved qa tool clarification")
        updated_state = enqueue_tool_requester(
            {
                **state,
                "meta": meta,
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

    resumed_state = {
        **state,
        "meta": clear_clarification_meta(state.get("meta") or {}),
    }
    return qa_node(resumed_state, llm)


def clarification_node(state: AgentState, llm, context: str = "") -> AgentState:
    meta = dict(state.get("meta") or {})
    kind = str(meta.get(MetaKeys.CLARIFICATION_KIND) or "")

    if kind == "qa_tool":
        return _resume_qa_tool_clarification(state, llm)

    question = latest_user_message(state)
    pending_question = meta.get(MetaKeys.PENDING_QUESTION)
    effective_question = (
        f"{pending_question}\n\nUser clarification: {question}"
        if pending_question and question
        else question
    )
    resumed_state = {
        **state,
        "meta": clear_clarification_meta(meta),
    }

    return qa_node(resumed_state, llm, context, question_override=effective_question)
