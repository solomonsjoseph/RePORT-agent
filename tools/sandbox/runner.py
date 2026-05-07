from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from scipy.stats import chi2_contingency, fisher_exact


def _safe_json_value(value: Any) -> Any:
    try:
        json.dumps(value)
    except TypeError:
        return None
    return value


def _execute_user_code(code: str, df: pd.DataFrame, *, dataset_id: str | None = None):
    dataset_key = str(dataset_id or "").strip()
    global_env = {
        "pd": pd,
        "np": np,
        "KaplanMeierFitter": KaplanMeierFitter,
        "CoxPHFitter": CoxPHFitter,
        "datasets": {dataset_key: df} if dataset_key else {},
        "selected_dataset_id": dataset_key,
        "chi2_contingency": chi2_contingency,
        "fisher_exact": fisher_exact,
        "plt": plt,
        "__name__": "__main__",
    }
    local_env: dict[str, Any] = {}

    stdout_buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_buf
    figure_png = b""

    try:
        exec(code, global_env, local_env)
        result = _safe_json_value(local_env.get("result"))
        error = None

        fig = plt.gcf()
        if fig and fig.axes:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            figure_png = buf.getvalue()

        plt.close("all")

    except Exception as exc:
        result = None
        error = {
            "type": "PythonRuntimeError",
            "message": str(exc),
        }

    finally:
        sys.stdout = old_stdout

    return result, stdout_buf.getvalue(), figure_png, error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    code = (input_dir / "code.py").read_text(encoding="utf-8")
    df = pd.read_csv(input_dir / "dataset.csv")
    dataset_id_path = input_dir / "dataset_id.txt"
    dataset_id = dataset_id_path.read_text(encoding="utf-8").strip() if dataset_id_path.exists() else ""

    result, stdout, figure_png, error = _execute_user_code(code, df, dataset_id=dataset_id)

    payload = {
        "status": "error" if error else "ok",
        "text": stdout if stdout else (str(result) if result is not None else ""),
        "result": result,
        "error": error,
        "has_figure": bool(figure_png),
    }
    (output_dir / "result.json").write_text(json.dumps(payload), encoding="utf-8")

    if stdout:
        (output_dir / "stdout.txt").write_text(stdout, encoding="utf-8")
    if figure_png:
        (output_dir / "figure.png").write_bytes(figure_png)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
