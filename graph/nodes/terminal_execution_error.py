from __future__ import annotations

from langchain_core.messages import AIMessage

from .state_helpers import update_agent_state

NODE_NAME = "terminal_execution_error"
NODE_CAPABILITY = (
    "Explain terminal execution failures directly to the user and end the turn without retrying."
)


def _terminal_error_message(error_payload: dict | None) -> str:
    error = dict(error_payload or {})
    category = str(error.get("category") or "")
    message = str(error.get("message") or "Unknown execution error.")

    if category == "policy_blocked":
        return (
            "The sandbox stopped this request because it asked for an operation "
            f"that is not allowed in this environment.\n\nDetails: {message}"
        )
    if category == "unsupported_runtime":
        return (
            "This request needs a package or runtime capability that is not "
            f"available in the current runtime image.\n\nDetails: {message}"
        )
    if category == "timeout":
        return (
            "This analysis ran longer than the allowed execution time, so it was "
            f"stopped before completion.\n\nDetails: {message}"
        )
    if category == "infrastructure":
        return (
            "The execution environment could not complete this run because of a "
            f"sandbox or infrastructure problem.\n\nDetails: {message}"
        )
    return f"Execution stopped with an unrecoverable error.\n\nDetails: {message}"


def terminal_execution_error_node(state):
    output = dict(state.get("output") or {})
    error = dict(output.get("error") or {})
    response_text = _terminal_error_message(error)

    messages = list(state.get("messages", []))
    messages.append(AIMessage(content=response_text))
    output["qa_response"] = response_text

    updated_state = {
        **state,
        "messages": messages,
        "output": output,
    }
    return update_agent_state(
        updated_state,
        "terminal_execution_error",
        {
            "status": "done",
            "response": response_text,
        },
    )
