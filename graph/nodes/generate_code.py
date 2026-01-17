from langchain_core.messages import AIMessage, HumanMessage
from utils.code_parser import extract_python_code
from prompts.generate_prompt import make_generate_code_prompt

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

    return {
        **state,
        "generated_code": code,
        "human_decision": None,
        "run_status": "pending",
    }
