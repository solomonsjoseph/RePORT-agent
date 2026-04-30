from langgraph.types import interrupt
from langchain_core.messages import HumanMessage
from ..conversation_events import append_conversation_event, build_review_decision_event
from ..state import MetaKeys
from .state_helpers import update_agent_state

NODE_NAME = "human_review_before_run"
NODE_CAPABILITY = "Ask human approval before running newly generated code."


def human_review_before_run_node(state):
    # pause here and send playload to UI
    output = dict(state.get("output") or {})
    feedback = interrupt({
        "type": "before_run_review",
        "code_summary": output.get("code_summary", ""),
        "code_assumptions": output.get("code_assumptions", ""),
        "generated_code": output.get("generated_code", ""),
    })
    decision = feedback.get("action")
    messages = list(state.get("messages", []))
    suggestion = feedback.get("suggestion", None)
    if suggestion:
        messages.append(HumanMessage(content=suggestion))
    # On resume, 'decision' becomes the user input
    meta = dict(state.get("meta", {}))
    approved_hash = meta.get("current_code_hash") if decision == "approve" else None
    updated_state = {
        **state,
        "messages": messages,
        "meta": meta,
    }
    updated_state = append_conversation_event(
        updated_state,
        build_review_decision_event(
            actor="human_review_before_run",
            user_turn_hash=str(meta.get(MetaKeys.LAST_USER_MESSAGE_HASH) or "").strip() or None,
            review_kind="before_run_review",
            decision=str(decision or ""),
            text=str(suggestion or ("Approved generated code for execution." if decision == "approve" else "Requested code regeneration before execution.")),
            status="done",
        ),
    )
    return update_agent_state(
        updated_state,
        "human_review",
        {
            "status": "done",
            "before_run_decision": decision,
            "approved_code_hash": approved_hash,
        },
    )
