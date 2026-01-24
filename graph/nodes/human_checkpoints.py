from langgraph.types import interrupt
from langchain_core.messages import HumanMessage, AIMessage
from .state_helpers import update_agent_state

def human_review_before_run_node(state):
    # pause here and send playload to UI
    feedback = interrupt({
        "type": "before_run_review",
        "generated_code": state.get("generated_code", "")
    })
    decision = feedback.get("action")
    messages = list(state.get("messages", []))
    suggestion = feedback.get("suggestion", None)
    if suggestion:
        messages.append(HumanMessage(content = suggestion))
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

def human_review_after_error_node(state):
    return state

def human_review_final_node(state):
    # pause here and send playload to UI
    feedback = interrupt({
        "type": "final_review",
        "generated_code": state.get("generated_code", ""),
        "output": state.get("output", ""),
        "figure_png": state.get("figure_png", ""),
    })
    decision = feedback.get("action")
    output = state.get("output", None)
    messages = list(state.get("messages", []))
    
    # if decision == "approve":
    code = state.get("generated_code")
    extra = {}
    figure_png = state.get("figure_png", None)   
    if figure_png:
        extra["figure_png"] = figure_png  # store raw bytes
    ai_msg = AIMessage(content = 
                        f"Generated code:\n```python\n{code}\n```", 
                        additional_kwargs=extra)
    if output:
        ai_msg.content += f"\n\nOutput:\n```\n{output}\n```"
    messages.append(ai_msg)

    suggestion = feedback.get("suggestion", None)
    if suggestion:
        messages.append(HumanMessage(content = suggestion))
                           
    updated_state = {
        **state,
        "messages": messages,
        "output": output
    }
    return update_agent_state(
        updated_state,
        "human_review",
        {
            "status": "done",
            "final_decision": decision,
        },
    )
