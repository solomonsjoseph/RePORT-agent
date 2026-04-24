from __future__ import annotations

import argparse
import os
from pathlib import Path

from .chunker import build_column_chunks, build_table_chunks
from .config import (
    CHROMA_DIR,
    DUCKDB_PATH,
    EMBEDDING_MODEL,
    INDEX_ROOT,
    MANIFEST_PATH,
    MANIFEST_ROOT,
    PROJECT_ROOT,
    SOURCE_ROOT,
    SUPPORTED_DB_RAG_EMBEDDING_MODELS,
    chroma_dir_for_model,
    env_path_for_project,
    manifest_path_for_model,
    resolve_db_rag_embedding_model,
)
from .data_loader import build_duckdb, load_excel_data, profile_column, source_fingerprint
from .vectorstore import build_chroma, write_manifest


def _write_env_key(project_root: Path, key: str, value: str) -> Path:
    env_path = env_path_for_project(project_root)
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
    print("python -m db_rag.build_index --rebuild --force")


def resolve_build_embedding_model() -> tuple[str, Path]:
    env_path = env_path_for_project(PROJECT_ROOT)
    try:
        model = resolve_db_rag_embedding_model()
    except ValueError:
        model = _prompt_for_embedding_model()
        os.environ["DB_RAG_EMBEDDING_MODEL"] = model
        env_path = _write_env_key(PROJECT_ROOT, "DB_RAG_EMBEDDING_MODEL", model)
    _print_env_guidance(model, env_path)
    return model, env_path


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


def rebuild(*, force: bool = False) -> dict[str, object]:
    model, env_path = resolve_build_embedding_model()
    chroma_dir = chroma_dir_for_model(model)
    manifest_path = manifest_path_for_model(model)
    if not force and chroma_dir.exists() and manifest_path.exists():
        _print_existing_index_message(model, chroma_dir, manifest_path, env_path)
        return {"embedding_model": model, "chroma_path": str(chroma_dir), "manifest_path": str(manifest_path)}

    excel_data = load_excel_data()
    table_chunks = build_table_chunks(excel_data)
    column_chunks = build_column_chunks(excel_data)
    build_duckdb(excel_data)
    build_chroma(table_chunks, column_chunks, model=model, chroma_dir=chroma_dir)
    manifest = {
        "source_fingerprint": source_fingerprint(),
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
    write_manifest(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Rebuild DuckDB and Chroma assets from local source data.")
    parser.add_argument("--force", action="store_true", help="Force a rebuild even if the selected embedding index already exists.")
    args = parser.parse_args()
    if not args.rebuild:
        parser.error("Use --rebuild to initialize DB-RAG assets.")
    manifest = rebuild(force=args.force)
    if manifest.get("source_fingerprint"):
        print(_format_rebuild_success(manifest))


if __name__ == "__main__":
    main()
