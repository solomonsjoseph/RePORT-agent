from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pandas as pd
import pytest


def _install_langchain_message_stubs(monkeypatch) -> None:
    messages_mod = ModuleType("langchain_core.messages")

    class _Message:
        def __init__(self, content: str = "") -> None:
            self.content = content

    messages_mod.HumanMessage = _Message
    messages_mod.SystemMessage = _Message
    monkeypatch.setitem(sys.modules, "langchain_core.messages", messages_mod)


def _fresh_build_index_module(monkeypatch):
    _install_langchain_message_stubs(monkeypatch)
    for module_name in (
        "db_rag.build_index",
        "db_rag.service",
        "db_rag.vectorstore",
        "db_rag.data_loader",
        "db_rag.chunker",
        "db_rag.config",
        "db_rag",
    ):
        sys.modules.pop(module_name, None)
    return importlib.import_module("db_rag.build_index")


def test_build_index_main_prints_success_summary(monkeypatch, capsys) -> None:
    build_index = _fresh_build_index_module(monkeypatch)
    manifest = {
        "source_fingerprint": "abc123",
        "embedding_model": "OpenAI/text-embedding-3-small",
        "duckdb_path": str(build_index.DUCKDB_PATH),
        "chroma_path": str(build_index.CHROMA_DIR),
        "manifest_path": str(build_index.MANIFEST_PATH),
        "source_root": str(build_index.SOURCE_ROOT),
        "table_chunk_count": 2,
        "column_chunk_count": 30,
    }

    monkeypatch.setattr(build_index, "rebuild", lambda *, indexing_model, force=False: manifest)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "python -m db_rag.build_index",
            "--indexing-model",
            "OpenAI/text-embedding-3-small",
        ],
    )

    build_index.main()

    output = capsys.readouterr().out
    assert "DB-RAG build completed successfully." in output
    assert "Indexing model: OpenAI/text-embedding-3-small" in output
    assert "Source data:" in output
    assert "local_data/db_rag_source" in output
    assert "DuckDB database:" in output
    assert "runtime/db_rag/report.duckdb" in output
    assert "Chroma index:" in output
    assert "Manifest:" in output
    assert "Indexed chunks: 2 table summaries, 30 column chunks" in output


def test_build_index_main_requires_indexing_model(monkeypatch, capsys) -> None:
    build_index = _fresh_build_index_module(monkeypatch)

    monkeypatch.setattr(sys, "argv", ["python -m db_rag.build_index"])

    with pytest.raises(SystemExit) as excinfo:
        build_index.main()

    error_output = capsys.readouterr().err
    assert excinfo.value.code == 2
    assert "--indexing-model is required" in error_output
    assert "Available indexing models:" in error_output
    assert "python -m db_rag.build_index --indexing-model Qwen/Qwen3-Embedding-4B" in error_output


def test_build_index_main_lists_voyage_embedding_model(monkeypatch, capsys) -> None:
    build_index = _fresh_build_index_module(monkeypatch)

    monkeypatch.setattr(sys, "argv", ["python -m db_rag.build_index"])

    with pytest.raises(SystemExit):
        build_index.main()

    error_output = capsys.readouterr().err
    assert "voyage-4-large" in error_output


def test_build_index_main_rejects_unsupported_indexing_model(monkeypatch, capsys) -> None:
    build_index = _fresh_build_index_module(monkeypatch)

    monkeypatch.setattr(
        sys,
        "argv",
        ["python -m db_rag.build_index", "--indexing-model", "Foo/Bar"],
    )

    with pytest.raises(SystemExit) as excinfo:
        build_index.main()

    error_output = capsys.readouterr().err
    assert excinfo.value.code == 2
    assert "unsupported indexing model: Foo/Bar" in error_output
    assert "Available indexing models:" in error_output


def test_profile_column_adds_min_and_max_for_numeric_data() -> None:
    from db_rag.data_loader import profile_column

    frame = pd.DataFrame({"AGE": [18, 21, 34]})

    profile = profile_column(frame, "AGE")

    assert profile["min"] == 18
    assert profile["max"] == 34


def test_resolve_build_embedding_model_writes_env_from_explicit_selection(monkeypatch, tmp_path, capsys) -> None:
    build_index = _fresh_build_index_module(monkeypatch)

    monkeypatch.setattr(build_index, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(build_index, "env_path_for_project", lambda project_root=tmp_path: tmp_path / ".env")

    model, env_path = build_index.resolve_build_embedding_model("OpenAI/text-embedding-3-small")

    assert model == "OpenAI/text-embedding-3-small"
    assert env_path == tmp_path / ".env"
    assert "DB_RAG_EMBEDDING_MODEL=OpenAI/text-embedding-3-small" in env_path.read_text()
    output = capsys.readouterr().out
    assert (
        ".env:DB_RAG_EMBEDDING_MODEL=OpenAI/text-embedding-3-small, "
        "this will be used as your primary indexing model in the RAG"
    ) in output


def test_resolve_build_embedding_model_reports_when_env_value_changes(monkeypatch, tmp_path, capsys) -> None:
    build_index = _fresh_build_index_module(monkeypatch)

    env_path = tmp_path / ".env"
    env_path.write_text("DB_RAG_EMBEDDING_MODEL=OpenAI/text-embedding-3-small\n", encoding="utf-8")

    monkeypatch.setattr(build_index, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(build_index, "env_path_for_project", lambda project_root=tmp_path: env_path)

    build_index.resolve_build_embedding_model("Qwen/Qwen3-Embedding-4B")

    output = capsys.readouterr().out
    assert "Updated .env: DB_RAG_EMBEDDING_MODEL changed from OpenAI/text-embedding-3-small to Qwen/Qwen3-Embedding-4B" in output
    assert (
        ".env:DB_RAG_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B, "
        "this will be used as your primary indexing model in the RAG"
    ) in output


def test_rebuild_skips_existing_index_without_force(monkeypatch, tmp_path, capsys) -> None:
    build_index = _fresh_build_index_module(monkeypatch)

    chroma_dir = tmp_path / "indexes" / "qwen"
    manifest_path = tmp_path / "manifests" / "qwen.json"
    chroma_dir.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(build_index, "resolve_build_embedding_model", lambda indexing_model: (indexing_model, tmp_path / ".env"))
    monkeypatch.setattr(build_index, "chroma_dir_for_model", lambda model: chroma_dir)
    monkeypatch.setattr(build_index, "manifest_path_for_model", lambda model: manifest_path)

    manifest = build_index.rebuild(indexing_model="Qwen/Qwen3-Embedding-4B")

    output = capsys.readouterr().out
    assert manifest["embedding_model"] == "Qwen/Qwen3-Embedding-4B"
    assert "DB-RAG index already exists" in output
    assert "--rebuild" in output
    assert "Edit that file to change models later." not in output


def test_rebuild_flag_bypasses_existing_index_skip(monkeypatch, tmp_path) -> None:
    build_index = _fresh_build_index_module(monkeypatch)

    chroma_dir = tmp_path / "indexes" / "openai_text_embedding_3_small"
    manifest_path = tmp_path / "manifests" / "openai_text_embedding_3_small.json"
    chroma_dir.mkdir(parents=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")

    calls: list[str] = []
    monkeypatch.setattr(
        build_index,
        "resolve_build_embedding_model",
        lambda indexing_model: (indexing_model, tmp_path / ".env"),
    )
    monkeypatch.setattr(build_index, "chroma_dir_for_model", lambda _: chroma_dir)
    monkeypatch.setattr(build_index, "manifest_path_for_model", lambda _: manifest_path)
    monkeypatch.setattr(build_index, "load_excel_data", lambda: {})
    monkeypatch.setattr(build_index, "build_table_chunks", lambda _: [])
    monkeypatch.setattr(build_index, "build_column_chunks", lambda _: [])
    monkeypatch.setattr(build_index, "build_duckdb", lambda _: calls.append("duckdb"))
    monkeypatch.setattr(build_index, "build_chroma", lambda *args, **kwargs: calls.append("chroma"))
    monkeypatch.setattr(build_index, "source_fingerprint", lambda: "fp")

    manifest = build_index.rebuild(indexing_model="OpenAI/text-embedding-3-small", force=True)

    assert manifest["embedding_model"] == "OpenAI/text-embedding-3-small"
    assert calls == ["duckdb", "chroma"]
