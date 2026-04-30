from langgraph.types import interrupt
from langchain_core.messages import HumanMessage

from ..conversation_events import append_conversation_event, build_review_decision_event
from ..state import MetaKeys
from .state_helpers import update_agent_state

NODE_NAME = "human_review_after_error"
NODE_CAPABILITY = (
    "Ask human for guidance after repeated retryable code-execution failures once retry budget is exhausted."
)


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
    meta[MetaKeys.ERROR_ITERATIONS] = 0
    meta.pop(MetaKeys.CURRENT_CODE_HASH, None)
    meta.pop(MetaKeys.FINAL_APPROVED_CODE_HASH, None)
    meta.pop(MetaKeys.EXECUTION_TICKET_HASH, None)
    meta.pop(MetaKeys.ERROR_RECOVERY_ACTIVE, None)

    output["generated_code"] = ""
    updated_state = {
        **state,
        "messages": messages,
        "meta": meta,
        "output": output,
    }
    updated_state = append_conversation_event(
        updated_state,
        build_review_decision_event(
            actor="human_review_after_error",
            user_turn_hash=str(meta.get(MetaKeys.LAST_USER_MESSAGE_HASH) or "").strip() or None,
            review_kind="after_error_review",
            decision=str(decision or ""),
            text=str(suggestion or ("Requested code regeneration after execution failure." if decision == "regenerate" else "Reviewed repeated execution failure.")),
            status="done",
        ),
    )
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
            "before_run_decision": None,
            "approved_code_hash": None,
        },
    )
