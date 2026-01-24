from langchain_core.messages import AIMessage
from utils.code_parser import extract_python_code
from prompts.fix_prompt import make_fix_code_prompt
from .state_helpers import update_agent_state

def error_handler_node(state, llm, context):
    code = state.get("generated_code") or ""
    error = state.get("error") or "Unknown error"

    prompt = make_fix_code_prompt().invoke(
        {
            "messages": state["messages"],
            "context": context,
            "code": code,
            "error_type": error["type"],
            "error_message":error["message"]
        }
    )
    response = llm.invoke(prompt)
    new_code = extract_python_code(response.content)

    updated_state = {
        **state,
        "generated_code": new_code,
    }
    updated_state = update_agent_state(
        updated_state,
        "error_handler",
        {
            "status": "done",
            "generated_code": new_code,
        },
    )
    return update_agent_state(
        updated_state,
        "executor",
        {
            "run_status": "pending",
        },
    )
