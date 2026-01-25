from langchain_core.messages import AIMessage
from utils.code_parser import extract_python_code
from prompts.fix_prompt import make_fix_code_prompt
from .state_helpers import update_agent_state

def error_handler_node(state, llm, context):
    output = dict(state.get("output") or {})
    code = output.get("generated_code") or ""
    error = output.get("error") or {"type": "Unknown", "message": "Unknown error"}
    meta = dict(state.get("meta", {}))
    error_iterations = int(meta.get("error_iterations", 0))
    if error_iterations == 0:
        review_state = dict(state.get("agents", {}).get("human_review", {}))
        review_state["after_error_decision"] = None
        agents = dict(state.get("agents", {}))
        agents["human_review"] = review_state
        state = {**state, "agents": agents}
    meta["error_iterations"] = error_iterations + 1

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

    output["generated_code"] = new_code
    updated_state = {
        **state,
        "output": output,
        "meta": meta,
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
