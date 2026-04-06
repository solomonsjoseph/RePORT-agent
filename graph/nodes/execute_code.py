from tools.execution import run_python_user
from .state_helpers import update_agent_state

NODE_NAME = "execute_code"
NODE_CAPABILITY = (
    "Execute previously generated Python code after approval and collect outputs or errors."
)


def execute_code_node(state, df):
    code = (state.get("output") or {}).get("generated_code")
    if not code:
        output = dict(state.get("output") or {})
        output["error"] = {
            "type": "NoCode",
            "message": "No code available to execute.",
        }
        updated_state = {
            **state,
            "output": output,
        }
        return update_agent_state(
            updated_state,
            "executor",
            {
                "status": "error",
                "run_status": "error",
                "error": "No code available to execute.",
            },
        )

    # --------------------------------------------------
    # Execute code
    # --------------------------------------------------
    result, stdout, figure_png, error = run_python_user(code, df)

    # --------------------------------------------------
    # Handle execution error
    # --------------------------------------------------
    if error:

        output = dict(state.get("output") or {})
        output["error"] = error
        updated_state = {
            **state,
            "output": output,
        }
        return update_agent_state(
            updated_state,
            "executor",
            {
                "status": "error",
                "run_status": "error",
                "error": error,
            },
        )

    # --------------------------------------------------
    # Execution succeeded
    # --------------------------------------------------
    output_text = stdout if stdout else (str(result) if result is not None else "")

    output_payload = dict(state.get("output") or {})
    output_payload["text"] = output_text
    if figure_png:
        output_payload["figure_png"] = figure_png
    meta = dict(state.get("meta", {}))
    meta["error_iterations"] = 0
    updated_state = {
        **state,
        "output": output_payload,
        "meta": meta,
    }
    return update_agent_state(
        updated_state,
        "executor",
        {
            "status": "done",
            "run_status": "ok",
            "output": output_text,
        },
    )

    

    
