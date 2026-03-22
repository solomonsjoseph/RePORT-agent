from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from utils.code_parser import extract_python_code
from utils.message_window import window_messages
from prompts.generate_prompt import make_generate_code_prompt
from .state_helpers import get_agent_state, update_agent_state
from .tool_routing import format_tool_results
from .code_guardrails import code_fingerprint, is_executable_python
from utils.llm_response import coerce_text_content

def generate_code_node(state, llm, context):
    generate_state = get_agent_state(state, "generate_code")
    messages = state.get("messages", [])
    tool_results = list(generate_state.get("tool_results", []))

    # Safety: must have human input
    if not any(isinstance(m, HumanMessage) for m in messages):
        return state

    output = dict(state.get("output") or {})
    if tool_results:
        output["tool_results"] = format_tool_results(tool_results)
    windowed = window_messages(messages, max_turns=10)
    prompt = make_generate_code_prompt().invoke(
        {
            "messages": windowed,
            "context": context,
            "output": output,
            "tool_results": output.get("tool_results", ""),
        }
    )
    response = llm.invoke(prompt)
    response_text = coerce_text_content(response.content)
    code = extract_python_code(response_text)

    # If the model returned guidance text (not executable Python), return that
    # guidance to the user and stop the turn rather than attempting execution.
    if not is_executable_python(code):
        msgs = list(state.get("messages", []))
        msgs.append(AIMessage(content=response_text))
        output = dict(state.get("output") or {})
        output["generated_code"] = ""
        output["qa_response"] = response_text
        meta = dict(state.get("meta", {}))
        # Not an interrupt: this flag marks model-driven clarification needed
        # before a new generation attempt can proceed.
        meta["awaiting_user_clarification"] = True
        updated_state = {
            **state,
            "messages": msgs,
            "output": output,
            "meta": meta,
        }
        return update_agent_state(
            updated_state,
            "generate_code",
            {
                "status": "done",
                "generated_code": "",
                "notes": ["Model returned non-code guidance; awaiting user clarification."],
            },
        )

    output = dict(state.get("output") or {})
    output["generated_code"] = code
    human_review_state = dict(state.get("agents", {}).get("human_review", {}))
    human_review_state["before_run_decision"] = None
    human_review_state["approved_code_hash"] = None
    agents = dict(state.get("agents", {}))
    agents["human_review"] = human_review_state
    meta = dict(state.get("meta", {}))
    meta["current_code_hash"] = code_fingerprint(code)
    meta.pop("awaiting_user_clarification", None)
    updated_state = {
        **state,
        "output": output,
        "agents": agents,
        "meta": meta,
    }
    return update_agent_state(
        updated_state,
        "generate_code",
        {
            "status": "done",
            "generated_code": code,
        },
    )
