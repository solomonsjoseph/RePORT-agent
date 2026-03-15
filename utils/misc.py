import numpy as np
import pandas as pd


def yesno_to_bool(value):
    if pd.isna(value):
        return np.nan

    normalized = str(value).strip().lower()
    if normalized in ["yes", "y", "1", "true"]:
        return 1
    if normalized in ["no", "n", "0", "false"]:
        return 0
    return np.nan
