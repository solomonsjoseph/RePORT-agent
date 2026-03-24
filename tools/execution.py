import io
import os
import pickle
import subprocess
import sys
import tempfile
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact
import matplotlib.pyplot as plt

def _execute_user_code(code: str, df: pd.DataFrame):
    """Execute user code quietly and return structured errors."""

    global_env = {
        "pd": pd,
        "np": np,
        "KaplanMeierFitter": KaplanMeierFitter,
        "CoxPHFitter": CoxPHFitter,
        "df": df,
        "chi2_contingency": chi2_contingency,
        "fisher_exact": fisher_exact,
        "plt": plt,
        "__name__": "__main__",
    }
    local_env = {}

    stdout_buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_buf

    # default: no figure
    figure_png = b""

    try:
        # run
        exec(code, global_env, local_env)
        result = local_env.get("result")
        error = None

        # Capture figure
        fig = plt.gcf()
        if fig and fig.axes:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            figure_png = buf.getvalue()

        # clear so next calls don't reuse plot
        plt.close("all")

    except Exception as e:
        result = None
        error = {
            "type": type(e).__name__,
            "message": str(e),
        }

    finally:
        sys.stdout = old_stdout

    stdout = stdout_buf.getvalue()

    return result, stdout, figure_png, error

def run_python_user(code: str, df: pd.DataFrame):
    """Execute user code quietly and return structured errors."""

    execution_mode = os.getenv("EXECUTION_MODE", "sandbox").lower()
    if execution_mode == "inline":
        return _execute_user_code(code, df)

    timeout_seconds = float(os.getenv("EXECUTION_TIMEOUT_SEC", "20"))
    with tempfile.TemporaryDirectory(prefix="report-agent-exec-") as tmpdir:
        input_path = os.path.join(tmpdir, "input.pkl")
        output_path = os.path.join(tmpdir, "output.pkl")

        with open(input_path, "wb") as f:
            pickle.dump({"code": code, "df": df}, f)

        command = [sys.executable, "-m", "tools.execution_worker", input_path, output_path]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None, "", b"", {
                "type": "TimeoutError",
                "message": f"Execution exceeded {timeout_seconds:.0f}s sandbox timeout.",
            }

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            return None, "", b"", {
                "type": "ExecutionError",
                "message": stderr or "Sandboxed execution failed to return a result.",
            }

        if not os.path.exists(output_path):
            return None, "", b"", {
                "type": "ExecutionError",
                "message": "Sandboxed execution failed to return a result.",
            }

        with open(output_path, "rb") as f:
            payload = pickle.load(f)

    return (
        payload.get("result"),
        payload.get("stdout", ""),
        payload.get("figure_png", b""),
        payload.get("error"),
    )
