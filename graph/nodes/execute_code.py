from tools.execution import run_python_user
from .state_helpers import update_agent_state

def execute_code_node(state, df):
    code = state.get("generated_code")
    if not code:
        updated_state = {
            **state,
            "error": {
                "type": "NoCode",
                "message": "No code available to execute."
            },
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

        updated_state = {
            **state,
            "error": error,
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
    output = stdout if stdout else (str(result) if result is not None else "")

    updated_state = {
        **state,
        "output": output,
        "error": None,
        "figure_png": figure_png,
    }
    return update_agent_state(
        updated_state,
        "executor",
        {
            "status": "done",
            "run_status": "ok",
            "output": output,
        },
    )

    

    
