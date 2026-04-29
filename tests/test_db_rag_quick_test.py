from __future__ import annotations

from types import SimpleNamespace


def test_quick_test_parser_accepts_debug_flag():
    from db_rag import quick_test

    parser = quick_test._build_parser()
    args = parser.parse_args(["what columns track age", "--debug"])

    assert args.query == "what columns track age"
    assert args.debug is True


def test_quick_test_parser_leaves_reranker_unset_by_default():
    from db_rag import quick_test

    parser = quick_test._build_parser()
    args = parser.parse_args(["what columns track age"])

    assert args.reranker is None


def test_quick_test_parser_accepts_voyage_reranker():
    from db_rag import quick_test

    parser = quick_test._build_parser()
    args = parser.parse_args(["question", "--reranker", "voyage/rerank-2.5"])

    assert args.reranker == "voyage/rerank-2.5"


def test_run_query_passes_debug_to_service(monkeypatch):
    from db_rag import quick_test

    calls: dict[str, object] = {}

    class _Service:
        def __init__(self, llm):
            calls["llm"] = llm

        def execute_sql_flow(self, question: str, *, debug: bool = False):
            calls["question"] = question
            calls["debug"] = debug
            return {"answer": "ok"}

    monkeypatch.setattr(quick_test, "_get_db_rag_service_class", lambda: _Service)
    monkeypatch.setattr(quick_test, "ensure_assets_ready", lambda rebuild_if_missing=True: {"ready": True})
    monkeypatch.setattr(quick_test, "build_runtime_llm", lambda **kwargs: "llm")

    result = quick_test.run_query("age and outcome", debug=True)

    assert result == {"answer": "ok"}
    assert calls["question"] == "age and outcome"
    assert calls["debug"] is True


def test_main_prints_runtime_banner_without_reranker(monkeypatch, capsys):
    from db_rag import quick_test

    monkeypatch.setattr(quick_test, "load_dotenv", lambda: None)
    monkeypatch.setattr(quick_test, "_runtime_assets_ready", lambda: True)
    monkeypatch.setattr(quick_test, "resolve_db_rag_embedding_model", lambda: "Qwen/Qwen3-Embedding-4B")
    monkeypatch.setattr(quick_test, "_default_model_name", lambda provider, base_url: "gpt-5.4", raising=False)
    monkeypatch.setattr(
        quick_test,
        "run_query",
        lambda *args, **kwargs: {"answer": "ok", "sql": "", "dataframe": None, "source_tables": []},
    )

    exit_code = quick_test.main(["what columns track age"])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Running quick test:" in output
    assert "Question: what columns track age" in output
    assert "Indexing model: Qwen/Qwen3-Embedding-4B" in output
    assert "Reranker: none (ChromaDB ordering)" in output
    assert "To enable reranking, pass --reranker <model>." in output
    assert (
        "Available reranker models: cohere/rerank-v3.5, cohere/rerank-4-fast, "
        "cohere/rerank-4-pro, voyage/rerank-2.5"
    ) in output
    assert "Query LLM: OpenAI / gpt-5.4" in output


def test_main_prints_runtime_banner_with_reranker(monkeypatch, capsys):
    from db_rag import quick_test

    monkeypatch.setattr(quick_test, "load_dotenv", lambda: None)
    monkeypatch.setattr(quick_test, "_runtime_assets_ready", lambda: True)
    monkeypatch.setattr(quick_test, "resolve_db_rag_embedding_model", lambda: "Qwen/Qwen3-Embedding-4B")
    monkeypatch.setattr(quick_test, "_default_model_name", lambda provider, base_url: "gpt-5.4", raising=False)
    monkeypatch.setattr(
        quick_test,
        "run_query",
        lambda *args, **kwargs: {"answer": "ok", "sql": "", "dataframe": None, "source_tables": []},
    )

    exit_code = quick_test.main(
        ["what columns track age", "--reranker", "cohere/rerank-v3.5", "--provider", "openai"]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Running quick test:" in output
    assert "Indexing model: Qwen/Qwen3-Embedding-4B" in output
    assert "Reranker: cohere/rerank-v3.5" in output
    assert "To enable reranking, pass --reranker <model>." not in output
    assert "Query LLM: OpenAI / gpt-5.4" in output


def test_print_debug_shows_sql_sections(capsys):
    from db_rag import quick_test

    quick_test._print_debug(
        {
            "debug": {
                "question": "age and outcome",
                "retrieved_tables": ["Form 1A"],
                "retrieved_columns": ["Form 1A.AGE"],
                "sql_tables": ["Form 1A"],
                "sql_columns": ["Form 1A.AGE"],
            }
        }
    )

    output = capsys.readouterr().out
    assert "SQL preparation:" in output
    assert "Question: age and outcome" in output
    assert "Tables: ['Form 1A']" in output
    assert "Columns: ['Form 1A.AGE']" in output


def test_print_debug_shows_retrieval_sections(capsys):
    from db_rag import quick_test

    quick_test._print_debug(
        {
            "debug": {
                "question": "age and outcome",
                "retrieved_tables": ["Form 1A", "Final Outcome"],
                "retrieved_columns": ["Form 1A.AGE", "Final Outcome.OUTCOME"],
                "sql_tables": ["Form 1A"],
                "sql_columns": ["Form 1A.AGE"],
            }
        }
    )

    output = capsys.readouterr().out
    assert "Retrieval summary:" in output
    assert "Retrieved tables: ['Form 1A', 'Final Outcome']" in output
    assert "Retrieved columns: ['Form 1A.AGE', 'Final Outcome.OUTCOME']" in output


def test_resolve_api_key_uses_provider_specific_env(monkeypatch):
    from db_rag import quick_test

    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")

    assert quick_test._resolve_api_key("openai", None) == "openai-key"
    assert quick_test._resolve_api_key("anthropic", None) == "anthropic-key"
    assert quick_test._resolve_api_key("gemini", None) == "google-key"
    assert quick_test._resolve_api_key("vllm", None) == ""


def test_resolve_api_key_prefers_explicit_override(monkeypatch):
    from db_rag import quick_test

    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")

    assert quick_test._resolve_api_key("anthropic", "override-key") == "override-key"
