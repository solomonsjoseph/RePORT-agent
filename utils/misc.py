import pandas as pd
import numpy as np
def yesno_to_bool(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().lower()
    if s in ["yes","y","1","true"]: return 1
    if s in ["no","n","0","false"]: return 0
    return np.nan