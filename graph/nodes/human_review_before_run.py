from langgraph.types import interrupt
from langchain_core.messages import HumanMessage
from .state_helpers import update_agent_state


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
    updated_state = {
        **state,
        "messages": messages
    }
    return update_agent_state(
        updated_state,
        "human_review",
        {
            "status": "done",
            "before_run_decision": decision,
        },
    )