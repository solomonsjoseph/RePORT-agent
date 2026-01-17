from tools.execution import run_python_user

def execute_code_node(state, df):
    code = state.get("generated_code")
    if not code:
        return {
            **state,
            "error": {
                "type": "NoCode",
                "message": "No code available to execute."
            },
            "run_status": "error",
        }

    # --------------------------------------------------
    # Execute code
    # --------------------------------------------------
    result, stdout, figure_png, error = run_python_user(code, df)

    # --------------------------------------------------
    # Handle execution error
    # --------------------------------------------------
    if error:

        return {
            **state,
            "error": error,
            "run_status": "error",
        }

    # --------------------------------------------------
    # Execution succeeded
    # --------------------------------------------------
    output = stdout if stdout else (str(result) if result is not None else "")

    return {
        **state,
        "output": output,
        "error": None,
        "run_status": "ok",
        "human_decision": None,
        "figure_png": figure_png,
    }

    

    
