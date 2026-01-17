from langchain_core.messages import AIMessage
from utils.code_parser import extract_python_code
from prompts.fix_prompt import make_fix_code_prompt

def handle_error_node(state, llm, context):
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

    return {
        **state,
        "generated_code": new_code,
        "run_status": "pending",
    }
