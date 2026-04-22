from langgraph.types import interrupt
from langchain_core.messages import HumanMessage, AIMessage
from .state_helpers import update_agent_state

NODE_NAME = "human_review_before_output"
NODE_CAPABILITY = (
    "Ask human approval of the final successful code-execution output before ending the task."
)


def human_review_before_output_node(state):
    # pause here and send playload to UI
    output = dict(state.get("output") or {})
    feedback = interrupt({
        "type": "final_review",
        "code_summary": output.get("code_summary", ""),
        "code_assumptions": output.get("code_assumptions", ""),
        "generated_code": output.get("generated_code", ""),
        "output": output.get("text", ""),
        "figure_png": output.get("figure_png", ""),
    })
    decision = feedback.get("action")
    output_text = output.get("text", None)
    messages = list(state.get("messages", []))

    if decision == "approve":
        code = output.get("generated_code")
        summary = str(output.get("code_summary", "") or "").strip()
        assumptions = str(output.get("code_assumptions", "") or "").strip()
        extra = {}
        figure_png = output.get("figure_png", None)
        if figure_png:
            extra["figure_png"] = figure_png  # store raw bytes
        body_parts = []
        if summary:
            body_parts.append(summary)
        if assumptions:
            body_parts.append(f"Assumptions: {assumptions}")
        body_parts.append(f"Generated code:\n```python\n{code}\n```")
        ai_msg = AIMessage(
            content="\n\n".join(body_parts),
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
