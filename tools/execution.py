import io
import sys
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter, CoxPHFitter
from scipy.stats import chi2_contingency
from scipy.stats import fisher_exact
import matplotlib.pyplot as plt

def run_python_user(code: str, df: pd.DataFrame):
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
