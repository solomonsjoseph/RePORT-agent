from utils.code_parser import extract_python_code
from utils.message_window import window_messages
from prompts.fix_prompt import make_fix_code_prompt
from .state_helpers import clear_clarification_meta, update_agent_state
from .code_guardrails import code_fingerprint, is_executable_python

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

    windowed = window_messages(state.get("messages", []), max_turns=10)
    prompt = make_fix_code_prompt().invoke(
        {
            "messages": windowed,
            "context": context,
            "code": code,
            "error_type": error["type"],
            "error_message":error["message"]
        }
    )
    response = llm.invoke(prompt)
    new_code = extract_python_code(response.content)

    if not is_executable_python(new_code):
        output["generated_code"] = ""
        output["qa_response"] = response.content
        meta = clear_clarification_meta(meta)
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
                "generated_code": "",
                "notes": ["Fix model returned non-code guidance."],
            },
        )
        return update_agent_state(
            updated_state,
            "executor",
            {
                "run_status": "idle",
            },
        )

    output["generated_code"] = new_code
    review_state = dict(state.get("agents", {}).get("human_review", {}))
    new_hash = code_fingerprint(new_code)
    # Preserve human approval across auto-fix retries so execution can continue
    # without repeatedly asking for review in the error-recovery loop.
    if review_state.get("before_run_decision") == "approve":
        review_state["before_run_decision"] = "approve"
        review_state["approved_code_hash"] = new_hash
    else:
        review_state["before_run_decision"] = None
        review_state["approved_code_hash"] = None
    agents = dict(state.get("agents", {}))
    agents["human_review"] = review_state
    meta = clear_clarification_meta(meta)
    meta["current_code_hash"] = new_hash
    updated_state = {
        **state,
        "output": output,
        "meta": meta,
        "agents": agents,
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
