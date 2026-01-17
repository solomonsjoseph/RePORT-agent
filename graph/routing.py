from .state import AgentState
from langgraph.graph import END

def route_after_generate(state: AgentState):
    # if not state.get("messages"):
    #     return END
    
    # Pause if human review requested
    # if state["review_stage"] is not None:
    #     return END
    if state.get("generated_code") is None: # Prevent infinite loop for LLM talking to itself
        return "generated_code"
    return "human_review_before_run"
    
def route_after_human_review_before_run(state: AgentState):
    # if not state.get("messages"):
    #     return END
    
    # Pause if human review requested
    # if state["review_stage"] is not None:
    #     return END

    # Human wants regeneration
    decision = state.get("human_decision", None)
    if decision and decision == "regenerate":
        return "generate_code"

    # Human approved
    if decision and decision == "approve":
        return "execute_code"
    return "human_review_before_run"


def route_after_execute(state: AgentState):
    # # Success → final review pause
    # if state["review_stage"] is not None:
    #     return END

    # if state["human_decision"] == "regenerate":
    #     return "generate_code"
    # return END
    if state["run_status"] == "error":
        return "handle_error"
    if state["run_status"] == "ok":
        return "human_review_final"
    
def route_after_human_review_final(state: AgentState):
    if state["human_decision"] == "regenerate":
        return "generate_code"
    if state["human_decision"] == "approve":
        return END
    return "human_review_final"

