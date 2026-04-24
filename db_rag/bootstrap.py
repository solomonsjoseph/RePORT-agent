from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import os

import pandas as pd

from .service import (
    CHROMA_DIR,
    DUCKDB_PATH,
    EMBEDDING_MODEL,
    EXCEL_DIR,
    INDEX_ROOT,
    MANIFEST_PATH,
    MANIFEST_ROOT,
    PROJECT_ROOT,
    RUNTIME_ROOT,
    SCHEMA_DIR,
    SOURCE_ROOT,
    OpenAIEmbeddingFunction,
    SUPPORTED_DB_RAG_EMBEDDING_MODELS,
    chroma_dir_for_model,
    manifest_path_for_model,
    resolve_db_rag_embedding_model,
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
    if pd.api.types.is_numeric_dtype(df[column]):
        profile["min"] = df[column].min()
        profile["max"] = df[column].max()
    return profile


def _write_env_key(project_root: Path, key: str, value: str) -> Path:
    env_path = project_root / ".env"
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    replacement = f"{key}={value}"
    updated = False
    new_lines: list[str] = []
    for line in lines:
        if line.startswith(f"{key}="):
            new_lines.append(replacement)
            updated = True
        else:
            new_lines.append(line)
    if not updated:
        new_lines.append(replacement)
    env_path.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")
    return env_path


def _prompt_for_embedding_model() -> str:
    for index, model in enumerate(SUPPORTED_DB_RAG_EMBEDDING_MODELS):
        print(f"[{index}] {model}")
    choice = (input("Select DB-RAG embedding model: ").strip() or "0")
    return SUPPORTED_DB_RAG_EMBEDDING_MODELS[int(choice)]


def _print_env_guidance(model: str, env_path: Path) -> None:
    print(f"Active DB_RAG_EMBEDDING_MODEL: {model}")
    print(f"Set DB_RAG_EMBEDDING_MODEL in {env_path}. Edit that file to change models later.")


def _print_existing_index_message(model: str, chroma_dir: Path, manifest_path: Path, env_path: Path) -> None:
    print(f"DB-RAG index already exists for the selected embedding model: {model}")
    print(f"Existing index: {chroma_dir}")
    print(f"Manifest: {manifest_path}")
    print()
    print("No rebuild was performed.")
    print()
    print(f"To switch to a different embedding model, edit {env_path} and change:")
    print("DB_RAG_EMBEDDING_MODEL=...")
    print()
    print("If your source data changed, or you want to rebuild this model anyway, run:")
    print("python -m db_rag.bootstrap --rebuild --force")


def resolve_build_embedding_model() -> tuple[str, Path]:
    env_path = PROJECT_ROOT / ".env"
    try:
        model = resolve_db_rag_embedding_model()
    except ValueError:
        model = _prompt_for_embedding_model()
        os.environ["DB_RAG_EMBEDDING_MODEL"] = model
        env_path = _write_env_key(PROJECT_ROOT, "DB_RAG_EMBEDDING_MODEL", model)
    _print_env_guidance(model, env_path)
    return model, env_path


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


def _build_chroma(
    table_chunks: list[dict[str, object]],
    column_chunks: list[dict[str, object]],
    *,
    model: str,
    chroma_dir: Path,
) -> None:
    import chromadb

    if chroma_dir.exists():
        for child in chroma_dir.iterdir():
            if child.is_file():
                child.unlink()
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    for collection_name in ("table_summaries", "column_chunks"):
        try:
            client.delete_collection(collection_name)
        except Exception:
            pass
    ef = OpenAIEmbeddingFunction(model=model)
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


def _display_path(path: object) -> str:
    candidate = Path(str(path)).resolve()
    try:
        return str(candidate.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(candidate)


def _format_rebuild_success(manifest: dict[str, object]) -> str:
    return "\n".join(
        [
            "DB-RAG rebuild completed successfully.",
            f"Embedding model: {manifest.get('embedding_model', EMBEDDING_MODEL)}",
            f"Source data: {_display_path(manifest.get('source_root', SOURCE_ROOT))}",
            f"DuckDB database: {_display_path(manifest.get('duckdb_path', DUCKDB_PATH))}",
            f"Chroma index: {_display_path(manifest.get('chroma_path', CHROMA_DIR))}",
            f"Manifest: {_display_path(manifest.get('manifest_path', MANIFEST_PATH))}",
            (
                "Indexed chunks: "
                f"{manifest.get('table_chunk_count', 0)} table summaries, "
                f"{manifest.get('column_chunk_count', 0)} column chunks"
            ),
            f"Source fingerprint: {manifest.get('source_fingerprint', 'unknown')}",
        ]
    )


def rebuild(*, force: bool = False) -> dict[str, object] | None:
    model, env_path = resolve_build_embedding_model()
    chroma_dir = chroma_dir_for_model(model)
    manifest_path = manifest_path_for_model(model)
    if chroma_dir.exists() and manifest_path.exists() and not force:
        _print_existing_index_message(model, chroma_dir, manifest_path, env_path)
        return None
    excel_data = _load_excel_data()
    table_chunks = _build_table_chunks(excel_data)
    column_chunks = _build_column_chunks(excel_data)
    _build_duckdb(excel_data)
    _build_chroma(table_chunks, column_chunks, model=model, chroma_dir=chroma_dir)
    manifest = {
        "source_fingerprint": _source_fingerprint(),
        "embedding_model": model,
        "source_root": str(SOURCE_ROOT),
        "duckdb_path": str(DUCKDB_PATH),
        "chroma_path": str(chroma_dir),
        "manifest_path": str(manifest_path),
        "table_chunk_count": len(table_chunks),
        "column_chunk_count": len(column_chunks),
    }
    INDEX_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Rebuild DuckDB and Chroma assets from local source data.")
    parser.add_argument("--force", action="store_true", help="Force rebuild even if the selected embedding model index already exists.")
    args = parser.parse_args()
    if not args.rebuild:
        parser.error("Use --rebuild to initialize DB-RAG assets.")
    manifest = rebuild(force=args.force)
    if manifest is None:
        return
    print(_format_rebuild_success(manifest))


if __name__ == "__main__":
    main()
