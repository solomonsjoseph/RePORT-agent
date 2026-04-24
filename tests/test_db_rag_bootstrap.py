from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pandas as pd


def _install_langchain_message_stubs(monkeypatch) -> None:
    messages_mod = ModuleType("langchain_core.messages")

    class _Message:
        def __init__(self, content: str = "") -> None:
            self.content = content

    messages_mod.HumanMessage = _Message
    messages_mod.SystemMessage = _Message
    monkeypatch.setitem(sys.modules, "langchain_core.messages", messages_mod)


def _fresh_bootstrap_module(monkeypatch):
    _install_langchain_message_stubs(monkeypatch)
    for module_name in ("db_rag.bootstrap", "db_rag.service", "db_rag"):
        sys.modules.pop(module_name, None)
    return importlib.import_module("db_rag.bootstrap")


def test_bootstrap_main_prints_success_summary(monkeypatch, capsys) -> None:
    bootstrap = _fresh_bootstrap_module(monkeypatch)
    manifest = {
        "source_fingerprint": "abc123",
        "embedding_model": "text-embedding-3-small",
        "duckdb_path": str(bootstrap.DUCKDB_PATH),
        "chroma_path": str(bootstrap.CHROMA_DIR),
        "manifest_path": str(bootstrap.MANIFEST_PATH),
        "source_root": str(bootstrap.SCHEMA_DIR.parent),
        "table_chunk_count": 2,
        "column_chunk_count": 30,
    }

    monkeypatch.setattr(bootstrap, "rebuild", lambda: manifest)
    monkeypatch.setattr(sys, "argv", ["python -m db_rag.bootstrap", "--rebuild"])

    bootstrap.main()

    output = capsys.readouterr().out
    assert "DB-RAG rebuild completed successfully." in output
    assert "Embedding model: text-embedding-3-small" in output
    assert "Source data:" in output
    assert "local_data/db_rag_source" in output
    assert "DuckDB database:" in output
    assert "runtime/db_rag/report.duckdb" in output
    assert "Chroma index:" in output
    assert "runtime/db_rag/chroma_db" in output
    assert "Manifest:" in output
    assert "runtime/db_rag/manifest.json" in output
    assert "Indexed chunks: 2 table summaries, 30 column chunks" in output


def test_profile_column_adds_min_and_max_for_numeric_data(monkeypatch) -> None:
    bootstrap = _fresh_bootstrap_module(monkeypatch)
    frame = pd.DataFrame({"AGE": [18, 21, 34]})

    profile = bootstrap._profile_column(frame, "AGE")

    assert profile["min"] == 18
    assert profile["max"] == 34


def test_rebuild_prompts_for_missing_embedding_model_and_writes_env(monkeypatch, tmp_path, capsys) -> None:
    bootstrap = _fresh_bootstrap_module(monkeypatch)

    monkeypatch.delenv("DB_RAG_EMBEDDING_MODEL", raising=False)
    monkeypatch.setattr("builtins.input", lambda _: "0")
    monkeypatch.setattr(bootstrap, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(bootstrap, "_load_excel_data", lambda: {})
    monkeypatch.setattr(bootstrap, "_build_table_chunks", lambda _: [])
    monkeypatch.setattr(bootstrap, "_build_column_chunks", lambda _: [])
    monkeypatch.setattr(bootstrap, "_build_duckdb", lambda _: None)
    monkeypatch.setattr(bootstrap, "_build_chroma", lambda *args, **kwargs: None)
    monkeypatch.setattr(bootstrap, "_source_fingerprint", lambda: "fp")

    manifest = bootstrap.rebuild()

    assert manifest["embedding_model"] == "OpenAI/text-embedding-3-small"
    assert "DB_RAG_EMBEDDING_MODEL=OpenAI/text-embedding-3-small" in (tmp_path / ".env").read_text()
    assert ".env" in capsys.readouterr().out
