from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.execution import run_python_user


def test_run_python_user_exposes_selected_dataset_by_exact_id(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "inline")
    lifelines = ModuleType("lifelines")
    lifelines.CoxPHFitter = object
    lifelines.KaplanMeierFitter = object
    monkeypatch.setitem(sys.modules, "lifelines", lifelines)
    df = pd.DataFrame({"gender": ["Female", "Male"], "tb_outcome": ["Cured", "Failed"]})

    _result, stdout, _figure_png, error = run_python_user(
        'tmp = datasets["subset-0e11007d"][["gender", "tb_outcome"]].copy()\n'
        "print(len(tmp))",
        df,
        dataset_id="subset-0e11007d",
    )

    assert error is None
    assert stdout.strip() == "2"
