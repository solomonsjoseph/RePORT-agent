from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from .service import (
    CHROMA_DIR,
    DUCKDB_PATH,
    EXCEL_DIR,
    MANIFEST_PATH,
    RUNTIME_ROOT,
    SCHEMA_DIR,
    OpenAIEmbeddingFunction,
)


def _find_excel(prefix: str) -> Path | None:
    for excel_file in EXCEL_DIR.glob("*.xlsx"):
        if excel_file.name.startswith(prefix):
            return excel_file
    return None


def _profile_column(df: pd.DataFrame, column: str) -> dict[str, object]:
    profile: dict[str, object] = {"dtype": str(df[column].dtype)}
    total = len(df)
    nulls = int(df[column].isna().sum())
    profile["null_rate"] = f"{nulls}/{total}"
    profile["samples"] = ", ".join(str(value) for value in df[column].dropna().unique()[:8].tolist())
    profile["distinct_count"] = int(df[column].nunique())
    return profile


def _load_excel_data() -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for schema_file in sorted(SCHEMA_DIR.glob("*.json")):
        prefix = schema_file.name.split(" ")[0] + "_"
        excel_file = _find_excel(prefix)
        data[prefix] = pd.read_excel(excel_file) if excel_file else pd.DataFrame()
    return data


def _build_table_chunks(excel_data: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    for schema_file in sorted(SCHEMA_DIR.glob("*.json")):
        schema = json.loads(schema_file.read_text(encoding="utf-8"))
        table_name = schema["form_name"]
        variables = schema.get("variables", {})
        summary = schema.get("summary", "")
        prefix = schema_file.name.split(" ")[0] + "_"
        df = excel_data.get(prefix, pd.DataFrame())
        subjid_col = next((c for c in df.columns if "SUBJID" in c.upper()), None)
        fid_col = next((c for c in df.columns if c.upper().endswith("FID")), None)
        join_keys = " and ".join([c for c in (subjid_col, fid_col) if c]) or "None"
        var_lines = [f"- {name}: {info.get('description', '(no description)')}" for name, info in variables.items()]
        parts = [
            f"Table: {table_name}",
            f"Summary: {summary}",
            f"Total variables: {len(variables)}",
            f"Join key: {join_keys}",
        ]
        if not df.empty:
            parts.append(f"Row count: {len(df)}")
        parts.append("Columns:\n" + "\n".join(var_lines))
        chunks.append(
            {
                "id": f"{table_name}.summary",
                "text": "\n".join(parts),
                "metadata": {"table": table_name},
            }
        )
    return chunks


def _build_column_chunks(excel_data: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    for schema_file in sorted(SCHEMA_DIR.glob("*.json")):
        schema = json.loads(schema_file.read_text(encoding="utf-8"))
        table_name = schema["form_name"]
        variables = schema.get("variables", {})
        prefix = schema_file.name.split(" ")[0] + "_"
        df = excel_data.get(prefix, pd.DataFrame())
        for column, info in variables.items():
            parts = [
                f"Table: {table_name}",
                f"Column: {column}",
                f"Description: {info.get('description') or '(no description)'}",
            ]
            values = info.get("values")
            if values:
                parts.append("Allowed values: " + ", ".join(f"{k}={v}" for k, v in values.items()))
            if not df.empty and column in df.columns:
                profile = _profile_column(df, column)
                parts.append("Profile: " + " | ".join(f"{k}: {v}" for k, v in profile.items()))
            chunks.append(
                {
                    "id": f"{table_name}.{column}",
                    "text": "\n".join(parts),
                    "metadata": {"table": table_name, "column": column},
                }
            )
    return chunks


def _build_duckdb(excel_data: dict[str, pd.DataFrame]) -> None:
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


def _build_chroma(table_chunks: list[dict[str, object]], column_chunks: list[dict[str, object]]) -> None:
    import chromadb

    if CHROMA_DIR.exists():
        for child in CHROMA_DIR.iterdir():
            if child.is_file():
                child.unlink()
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    for collection_name in ("table_summaries", "column_chunks"):
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    ef = OpenAIEmbeddingFunction()
    table_collection = client.create_collection("table_summaries", embedding_function=ef)
    column_collection = client.create_collection("column_chunks", embedding_function=ef)
    table_collection.add(
        ids=[chunk["id"] for chunk in table_chunks],
        documents=[chunk["text"] for chunk in table_chunks],
        metadatas=[chunk["metadata"] for chunk in table_chunks],
    )
    column_collection.add(
        ids=[chunk["id"] for chunk in column_chunks],
        documents=[chunk["text"] for chunk in column_chunks],
        metadatas=[chunk["metadata"] for chunk in column_chunks],
    )


def _source_fingerprint() -> str:
    digest = sha256()
    for path in sorted([*SCHEMA_DIR.glob("*.json"), *EXCEL_DIR.glob("*.xlsx")]):
        stat = path.stat()
        digest.update(str(path.relative_to(path.parents[2])).encode())
        digest.update(str(stat.st_mtime_ns).encode())
        digest.update(str(stat.st_size).encode())
    return digest.hexdigest()


def rebuild() -> None:
    excel_data = _load_excel_data()
    table_chunks = _build_table_chunks(excel_data)
    column_chunks = _build_column_chunks(excel_data)
    _build_duckdb(excel_data)
    _build_chroma(table_chunks, column_chunks)
    manifest = {
        "source_fingerprint": _source_fingerprint(),
        "embedding_model": "text-embedding-3-small",
        "duckdb_path": str(DUCKDB_PATH),
        "chroma_path": str(CHROMA_DIR),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Rebuild DuckDB and Chroma assets from local source data.")
    args = parser.parse_args()
    if not args.rebuild:
        parser.error("Use --rebuild to initialize DB-RAG assets.")
    rebuild()


if __name__ == "__main__":
    main()
