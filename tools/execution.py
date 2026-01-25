import io
import os
import sys
import multiprocessing
from multiprocessing.connection import Connection
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


def _subprocess_worker(code: str, df: pd.DataFrame, conn: Connection):
    result, stdout, figure_png, error = _execute_user_code(code, df)
    conn.send(
        {
            "result": result,
            "stdout": stdout,
            "figure_png": figure_png,
            "error": error,
        }
    )
    conn.close()

def run_python_user(code: str, df: pd.DataFrame):
    """Execute user code quietly and return structured errors."""

    execution_mode = os.getenv("EXECUTION_MODE", "sandbox").lower()
    if execution_mode == "inline":
        return _execute_user_code(code, df)

    timeout_seconds = float(os.getenv("EXECUTION_TIMEOUT_SEC", "20"))
    parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.Process(
        target=_subprocess_worker,
        args=(code, df, child_conn),
        daemon=True,
    )
    process.start()
    process.join(timeout=timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        return None, "", b"", {
            "type": "TimeoutError",
            "message": f"Execution exceeded {timeout_seconds:.0f}s sandbox timeout.",
        }

    if not parent_conn.poll():
        return None, "", b"", {
            "type": "ExecutionError",
            "message": "Sandboxed execution failed to return a result.",
        }

    payload = parent_conn.recv()
    return (
        payload.get("result"),
        payload.get("stdout", ""),
        payload.get("figure_png", b""),
        payload.get("error"),
    )