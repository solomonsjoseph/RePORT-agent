from langgraph.types import interrupt
from langchain_core.messages import HumanMessage, AIMessage
from .state_helpers import update_agent_state


def human_review_final_node(state):
    # pause here and send playload to UI
    output = dict(state.get("output") or {})
    feedback = interrupt({
        "type": "final_review",
        "generated_code": output.get("generated_code", ""),
        "output": output.get("text", ""),
        "figure_png": output.get("figure_png", ""),
    })
    decision = feedback.get("action")
    output_text = output.get("text", None)
    messages = list(state.get("messages", []))

    if decision == "approve":
        code = output.get("generated_code")
        extra = {}
        figure_png = output.get("figure_png", None)
        if figure_png:
            extra["figure_png"] = figure_png  # store raw bytes
        ai_msg = AIMessage(
            content=f"Generated code:\n```python\n{code}\n```",
            additional_kwargs=extra,
        )
        if output_text:
            ai_msg.content += f"\n\nOutput:\n```\n{output_text}\n```"
        messages.append(ai_msg)

    suggestion = feedback.get("suggestion", None)
    if suggestion:
        messages.append(HumanMessage(content=suggestion))

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
