from langchain_core.messages import AIMessage, HumanMessage
from utils.code_parser import extract_python_code
from prompts.generate_prompt import make_generate_code_prompt
from .state_helpers import update_agent_state

def generate_code_node(state, llm, context):
    messages = state.get("messages", [])

    # Safety: must have human input
    if not any(isinstance(m, HumanMessage) for m in messages):
        return state
    output = state.get("output", None)
    prompt = make_generate_code_prompt().invoke(
        {"messages": messages, "context": context, "output": output}
    )
    response = llm.invoke(prompt)
    code = extract_python_code(response.content)

    output = dict(state.get("output") or {})
    output["generated_code"] = code
    updated_state = {
        **state,
        "output": output,
    }
    return update_agent_state(
        updated_state,
        "generate_code",
        {
            "status": "done",
            "generated_code": code,
        },
    )
