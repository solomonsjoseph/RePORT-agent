from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

import pandas as pd

from .config import DUCKDB_PATH, EXCEL_DIR, RUNTIME_ROOT, SCHEMA_DIR


def find_excel(prefix: str) -> Path | None:
    for excel_file in EXCEL_DIR.glob("*.xlsx"):
        if excel_file.name.startswith(prefix):
            return excel_file
    return None


def profile_column(df: pd.DataFrame, column: str) -> dict[str, object]:
    profile: dict[str, object] = {"dtype": str(df[column].dtype)}
    total = len(df)
    nulls = int(df[column].isna().sum())
    profile["null_rate"] = f"{nulls}/{total}"
    profile["samples"] = ", ".join(str(value) for value in df[column].dropna().unique()[:8].tolist())
    profile["distinct_count"] = int(df[column].nunique())
    if pd.api.types.is_numeric_dtype(df[column]):
        profile["min"] = df[column].min()
        profile["max"] = df[column].max()
    return profile


def load_excel_data() -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for schema_file in sorted(SCHEMA_DIR.glob("*.json")):
        prefix = schema_file.name.split(" ")[0] + "_"
        excel_file = find_excel(prefix)
        data[prefix] = pd.read_excel(excel_file) if excel_file else pd.DataFrame()
    return data


def build_duckdb(excel_data: dict[str, pd.DataFrame]) -> None:
    import duckdb

    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
    if DUCKDB_PATH.exists():
        DUCKDB_PATH.unlink()
    db = duckdb.connect(str(DUCKDB_PATH))
    for schema_file in sorted(SCHEMA_DIR.glob("*.json")):
        schema = json.loads(schema_file.read_text(encoding="utf-8"))
        table_name = schema["form_name"]
        prefix = schema_file.name.split(" ")[0] + "_"
        df = excel_data.get(prefix, pd.DataFrame())
        if df.empty:
            continue
        db.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM df')
    db.close()


def source_fingerprint() -> str:
    digest = sha256()
    for path in sorted([*SCHEMA_DIR.glob("*.json"), *EXCEL_DIR.glob("*.xlsx")]):
        stat = path.stat()
        digest.update(str(path.relative_to(path.parents[2])).encode())
        digest.update(str(stat.st_mtime_ns).encode())
        digest.update(str(stat.st_size).encode())
    return digest.hexdigest()
