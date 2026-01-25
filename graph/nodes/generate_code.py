from langchain_core.messages import HumanMessage
from utils.code_parser import extract_python_code
from prompts.generate_prompt import make_generate_code_prompt
from .state_helpers import get_agent_state, update_agent_state
from .tool_routing import format_tool_results, latest_user_message, request_tools_for_question

def generate_code_node(state, llm, context):
    generate_state = get_agent_state(state, "generate_code")
    messages = state.get("messages", [])
    question = latest_user_message(state)
    tool_requests = list(generate_state.get("tool_requests", []))
    tool_results = list(generate_state.get("tool_results", []))

    # Safety: must have human input
    if not any(isinstance(m, HumanMessage) for m in messages):
        return state
    if question and not tool_results and not tool_requests:
        tool_requests = request_tools_for_question(llm, question)
        if tool_requests:
            observations = list(state.get("observations", []))
            observations.append("generate_code: requested tools")
            updated_state = {
                **state,
                "observations": observations,
            }
            return update_agent_state(
                updated_state,
                "generate_code",
                {
                    "status": "pending",
                    "tool_requests": tool_requests,
                },
            )
    output = dict(state.get("output") or {})
    if tool_results:
        output["tool_results"] = format_tool_results(tool_results)
    prompt = make_generate_code_prompt().invoke(
        {
            "messages": messages,
            "context": context,
            "output": output,
            "tool_results": output.get("tool_results", ""),
        }
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
