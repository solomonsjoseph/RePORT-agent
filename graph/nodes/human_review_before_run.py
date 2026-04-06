from langgraph.types import interrupt
from langchain_core.messages import HumanMessage
from .state_helpers import update_agent_state

NODE_NAME = "human_review_before_run"
NODE_CAPABILITY = "Ask human approval before running newly generated code."


def human_review_before_run_node(state):
    # pause here and send playload to UI
    output = dict(state.get("output") or {})
    feedback = interrupt({
        "type": "before_run_review",
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
    return update_agent_state(
        updated_state,
        "human_review",
        {
            "status": "done",
            "before_run_decision": decision,
            "approved_code_hash": approved_hash,
        },
    )
