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
)
from .data_loader import build_duckdb, load_excel_data, source_fingerprint
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


def _read_env_key(project_root: Path, key: str) -> str:
    env_path = env_path_for_project(project_root)
    if not env_path.exists():
        return ""
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}="):
            return line.split("=", 1)[1].strip()
    return ""


def _format_available_models() -> str:
    return "\n".join(f"- {model}" for model in SUPPORTED_DB_RAG_EMBEDDING_MODELS)


def _build_rebuild_command(model: str, *, force: bool = False) -> str:
    command = f"python -m db_rag.build_index --indexing-model {model}"
    if force:
        command = f"{command} --rebuild"
    return command


def _validate_indexing_model(parser: argparse.ArgumentParser, indexing_model: str | None) -> str:
    model = str(indexing_model or "").strip()
    if not model:
        parser.exit(
            2,
            (
                "Error: --indexing-model is required.\n\n"
                f"Available indexing models:\n{_format_available_models()}\n\n"
                f"Rerun with:\n{_build_rebuild_command('Qwen/Qwen3-Embedding-4B')}\n"
            ),
        )
    if model not in SUPPORTED_DB_RAG_EMBEDDING_MODELS:
        parser.exit(
            2,
            (
                f"Error: unsupported indexing model: {model}\n\n"
                f"Available indexing models:\n{_format_available_models()}\n\n"
                f"Rerun with:\n{_build_rebuild_command('Qwen/Qwen3-Embedding-4B')}\n"
            ),
        )
    return model


def _print_existing_index_message(model: str, chroma_dir: Path, manifest_path: Path) -> None:
    print(f"DB-RAG index already exists for the selected embedding model: {model}")
    print(f"Existing index: {chroma_dir}")
    print(f"Manifest: {manifest_path}")
    print()
    print("No rebuild was performed.")
    print()
    print("If your source data changed, or you want to rebuild this model anyway, run:")
    print(_build_rebuild_command(model, force=True))


def resolve_build_embedding_model(indexing_model: str) -> tuple[str, Path]:
    env_path = env_path_for_project(PROJECT_ROOT)
    previous_value = _read_env_key(PROJECT_ROOT, "DB_RAG_EMBEDDING_MODEL")
    os.environ["DB_RAG_EMBEDDING_MODEL"] = indexing_model
    env_path = _write_env_key(PROJECT_ROOT, "DB_RAG_EMBEDDING_MODEL", indexing_model)
    if previous_value and previous_value != indexing_model:
        print(
            "Updated .env: DB_RAG_EMBEDDING_MODEL changed "
            f"from {previous_value} to {indexing_model}"
        )
    print(
        f".env:DB_RAG_EMBEDDING_MODEL={indexing_model}, "
        "this will be used as your primary indexing model in the RAG"
    )
    return indexing_model, env_path


def _display_path(path: object) -> str:
    candidate = Path(str(path)).resolve()
    try:
        return str(candidate.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(candidate)


def _format_rebuild_success(manifest: dict[str, object]) -> str:
    return "\n".join(
        [
            "DB-RAG build completed successfully.",
            f"Indexing model: {manifest.get('embedding_model', EMBEDDING_MODEL)}",
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


def _print_rebuild_progress(message: str) -> None:
    print(message, flush=True)


def rebuild(*, indexing_model: str, force: bool = False) -> dict[str, object]:
    model, _env_path = resolve_build_embedding_model(indexing_model)
    chroma_dir = chroma_dir_for_model(model)
    manifest_path = manifest_path_for_model(model)
    if not force and chroma_dir.exists() and manifest_path.exists():
        _print_existing_index_message(model, chroma_dir, manifest_path)
        return {"embedding_model": model, "chroma_path": str(chroma_dir), "manifest_path": str(manifest_path)}

    _print_rebuild_progress(f"Starting DB-RAG build for indexing model: {model}")
    _print_rebuild_progress("[1/5] Loading source Excel data...")
    excel_data = load_excel_data()
    _print_rebuild_progress("[2/5] Building table and column chunks...")
    table_chunks = build_table_chunks(excel_data)
    column_chunks = build_column_chunks(excel_data)
    _print_rebuild_progress("[3/5] Building DuckDB database...")
    build_duckdb(excel_data)
    _print_rebuild_progress("[4/5] Building Chroma index...")
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
    _print_rebuild_progress("[5/5] Writing manifest...")
    INDEX_ROOT.mkdir(parents=True, exist_ok=True)
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest_path, manifest)
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m db_rag.build_index")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force a rebuild even if the selected embedding index already exists.",
    )
    parser.add_argument(
        "--indexing-model",
        default=None,
        help="Embedding model used to build the DB-RAG index.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    indexing_model = _validate_indexing_model(parser, args.indexing_model)
    manifest = rebuild(indexing_model=indexing_model, force=args.rebuild)
    if manifest.get("source_fingerprint"):
        print(_format_rebuild_success(manifest))


if __name__ == "__main__":
    main()
