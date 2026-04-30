from langgraph.types import interrupt
from langchain_core.messages import HumanMessage, AIMessage
from .state_helpers import update_agent_state
from ..state_views import get_artifact_files
from ..conversation_events import (
    append_conversation_event,
    build_assistant_event,
    build_figure_event,
)
from ..state import MetaKeys
from .orchestrator.state_logic import _user_message_hash

NODE_NAME = "human_review_before_output"
NODE_CAPABILITY = (
    "Ask human approval of the final successful code-execution output before ending the task."
)


def human_review_before_output_node(state):
    # pause here and send playload to UI
    output = dict(state.get("output") or {})
    artifact_files = get_artifact_files(state)
    figure_artifact_id = str(output.get("figure_artifact_id") or "").strip()
    figure_path = ""
    if figure_artifact_id and figure_artifact_id in artifact_files:
        content = dict(artifact_files[figure_artifact_id].get("content") or {})
        path_value = content.get("path")
        if isinstance(path_value, str):
            figure_path = path_value
    feedback = interrupt({
        "type": "final_review",
        "code_summary": output.get("code_summary", ""),
        "code_assumptions": output.get("code_assumptions", ""),
        "generated_code": output.get("generated_code", ""),
        "output": output.get("text", ""),
        "figure_artifact_id": figure_artifact_id,
        "figure_path": figure_path,
    })
    decision = feedback.get("action")
    output_text = output.get("text", None)
    messages = list(state.get("messages", []))

    if decision == "approve":
        code = output.get("generated_code")
        summary = str(output.get("code_summary", "") or "").strip()
        assumptions = str(output.get("code_assumptions", "") or "").strip()
        extra = {}
        if figure_path:
            extra["figure_path"] = figure_path
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
        meta = dict(state.get("meta") or {})
        user_turn_hash = str(meta.get(MetaKeys.LAST_USER_MESSAGE_HASH) or _user_message_hash(state) or "") or None
        updated_event_state = append_conversation_event(
            state,
            build_assistant_event(
                actor="human_review_before_output",
                user_turn_hash=user_turn_hash,
                text=ai_msg.content,
                status="done",
            ),
        )
        assistant_event_id = str(
            (updated_event_state.get("artifacts") or {}).get("conversation_events", [])[-1]["event_id"]
        )
        if figure_artifact_id and figure_path:
            updated_event_state = append_conversation_event(
                updated_event_state,
                build_figure_event(
                    actor="human_review_before_output",
                    user_turn_hash=user_turn_hash,
                    artifact_id=figure_artifact_id,
                    text="Approved figure output.",
                    status="done",
                    parent_event_id=assistant_event_id,
                ),
            )
        state = updated_event_state

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
