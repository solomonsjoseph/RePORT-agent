from langgraph.types import interrupt
from langchain_core.messages import HumanMessage
from .state_helpers import update_agent_state


def human_review_after_error_node(state):
    output = dict(state.get("output") or {})
    feedback = interrupt({
        "type": "after_error_review",
        "generated_code": output.get("generated_code", ""),
        "error": output.get("error", {}),
    })
    decision = feedback.get("action")
    messages = list(state.get("messages", []))
    suggestion = feedback.get("suggestion", None)
    if suggestion:
        messages.append(HumanMessage(content=suggestion))

    meta = dict(state.get("meta", {}))
    meta["error_iterations"] = 0

    output["generated_code"] = ""
    updated_state = {
        **state,
        "messages": messages,
        "meta": meta,
        "output": output,
    }
    updated_state = update_agent_state(
        updated_state,
        "executor",
        {
            "run_status": "idle",
        },
    )
    return update_agent_state(
        updated_state,
        "human_review",
        {
            "status": "done",
            "after_error_decision": decision,
        },
    )
