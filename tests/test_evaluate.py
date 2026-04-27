from __future__ import annotations

import json
import pytest


def test_evaluate_parser_accepts_benchmark_and_debug():
    from db_rag.benchmark import evaluate

    parser = evaluate._build_parser()
    args = parser.parse_args(["--benchmark", "custom.csv", "--indexing-model", "Qwen/Qwen3-Embedding-4B", "--debug"])

    assert args.benchmark == "custom.csv"
    assert args.indexing_model == "Qwen/Qwen3-Embedding-4B"
    assert args.debug is True


def test_resolve_output_tag_includes_reranker():
    from db_rag.benchmark import evaluate

    args = type(
        "_Args",
        (),
        {
            "indexing_model": "Qwen/Qwen3-Embedding-4B",
            "reranker": "Qwen/Qwen3-Reranker-4B",
            "provider": "openai",
            "model": None,
        },
    )()

    assert evaluate.resolve_output_tag(args) == "Qwen_Qwen3-Embedding-4B__Qwen_Qwen3-Reranker-4B__openai_default"


def test_resolve_reranker_model_requires_explicit_choice():
    from db_rag.benchmark import evaluate

    args = type(
        "_Args",
        (),
        {
            "indexing_model": "Qwen/Qwen3-Embedding-4B",
            "reranker": None,
        },
    )()

    assert evaluate.resolve_reranker_model(args) is None


def test_evaluate_main_requires_indexing_model(capsys):
    from db_rag.benchmark import evaluate

    with pytest.raises(SystemExit) as excinfo:
        evaluate.main([])

    error_output = capsys.readouterr().err
    assert excinfo.value.code == 2
    assert "--indexing-model is required" in error_output
    assert "python -m db_rag.benchmark.evaluate" in error_output


def test_evaluate_main_rejects_unsupported_indexing_model(capsys):
    from db_rag.benchmark import evaluate

    with pytest.raises(SystemExit) as excinfo:
        evaluate.main(["--indexing-model", "Foo/Bar"])

    error_output = capsys.readouterr().err
    assert excinfo.value.code == 2
    assert "unsupported indexing model: Foo/Bar" in error_output
    assert "python -m db_rag.benchmark.evaluate" in error_output


def test_evaluate_retrieval_computes_summary(tmp_path):
    from db_rag.benchmark import evaluate

    benchmark = tmp_path / "benchmark.csv"
    benchmark.write_text(
        "\n".join(
            [
                "question,expected_tables,expected_columns,difficulty",
                "Which fields track age?,Form 1A|Form 1B,AGE|SEX,easy",
                "This is unanswerable,NONE,NONE,hard",
            ]
        ),
        encoding="utf-8",
    )

    def retriever(question: str, *, debug: bool = False):
        assert question == "Which fields track age?"
        assert debug is False
        return (
            [
                {"table": "Form 1B", "text": "table context"},
                {"table": "Other Form", "text": "other"},
            ],
            [
                {"table": "Form 1B", "column": "AGE", "text": "age column"},
                {"table": "Other Form", "column": "OTHER", "text": "other column"},
            ],
        )

    summary, details = evaluate.evaluate_retrieval(benchmark, retriever=retriever)

    assert summary["total_questions"] == 2
    assert summary["scored_questions"] == 1
    assert summary["unanswerable"] == 1
    assert summary["table_recall@k"] == 1.0
    assert summary["table_mrr"] == 1.0
    assert summary["column_recall@k"] == 0.5
    assert summary["column_precision@k"] == 0.5
    assert summary["column_mrr"] == 1.0
    assert summary["easy_table_recall"] == 1.0
    assert len(details.index) == 2
    assert details.iloc[0]["predicted_tables"] == "Form 1B|Other Form"
    assert details.iloc[0]["matched_columns"] == "AGE"
    assert details.iloc[1]["table_hit"] == "N/A"


def test_evaluate_retrieval_prints_question_progress(tmp_path, capsys):
    from db_rag.benchmark import evaluate

    benchmark = tmp_path / "benchmark.csv"
    benchmark.write_text(
        "\n".join(
            [
                "question,expected_tables,expected_columns,difficulty",
                "Question one,Form 1A,AGE,easy",
                "Question two,Form 1A,AGE,easy",
            ]
        ),
        encoding="utf-8",
    )

    def retriever(question: str, *, debug: bool = False):
        return (
            [{"table": "Form 1A", "text": "table context"}],
            [{"table": "Form 1A", "column": "AGE", "text": "age column"}],
        )

    evaluate.evaluate_retrieval(benchmark, retriever=retriever)

    output = capsys.readouterr().out
    assert "Evaluating question [1/2]" in output
    assert "Evaluating question [2/2]" in output


def test_evaluate_retrieval_marks_failed_question_as_na_and_continues(tmp_path):
    from db_rag.benchmark import evaluate

    benchmark = tmp_path / "benchmark.csv"
    benchmark.write_text(
        "\n".join(
            [
                "question,expected_tables,expected_columns,difficulty",
                "Broken question,Form 1A,AGE,easy",
                "Working question,Form 1A,AGE,easy",
            ]
        ),
        encoding="utf-8",
    )

    def retriever(question: str, *, debug: bool = False):
        if question == "Broken question":
            raise RuntimeError("temporary embedding failure")
        return (
            [{"table": "Form 1A", "text": "table context"}],
            [{"table": "Form 1A", "column": "AGE", "text": "age column"}],
        )

    summary, details = evaluate.evaluate_retrieval(benchmark, retriever=retriever)

    assert summary["total_questions"] == 2
    assert summary["scored_questions"] == 1
    assert summary["unanswerable"] == 0
    assert summary["errored_questions"] == 1
    assert summary["table_recall@k"] == 1.0
    assert summary["column_recall@k"] == 1.0
    assert details.iloc[0]["table_hit"] == "N/A"
    assert details.iloc[0]["table_rank"] == "N/A"
    assert details.iloc[0]["column_recall"] == "N/A"
    assert details.iloc[0]["column_precision"] == "N/A"
    assert details.iloc[0]["column_mrr"] == "N/A"
    assert details.iloc[0]["error"] == "temporary embedding failure"
    assert details.iloc[1]["table_hit"] is True


def test_main_writes_summary_and_detail_files(monkeypatch, tmp_path, capsys):
    from db_rag.benchmark import evaluate

    benchmark = tmp_path / "benchmark.csv"
    benchmark.write_text(
        "\n".join(
            [
                "question,expected_tables,expected_columns,difficulty",
                "Which fields track age?,Form 1A,AGE,easy",
            ]
        ),
        encoding="utf-8",
    )

    def retriever(question: str, *, debug: bool = False):
        return (
            [{"table": "Form 1A", "text": "table context"}],
            [{"table": "Form 1A", "column": "AGE", "text": "age column"}],
        )

    monkeypatch.setattr(evaluate, "build_runtime_retriever", lambda args: retriever)
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(evaluate, "resolve_output_tag", lambda args: "test-tag")
    monkeypatch.setattr(evaluate, "_default_model_name", lambda provider, base_url: "gpt-4.1-mini", raising=False)
    monkeypatch.setattr(evaluate, "ensure_prebuilt_assets", lambda indexing_model: None)
    exit_code = evaluate.main(
        [
            "--benchmark",
            str(benchmark),
            "--indexing-model",
            "Qwen/Qwen3-Embedding-4B",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Indexing model: Qwen/Qwen3-Embedding-4B" in output
    assert "Reranker: none (ChromaDB ordering)" in output
    assert "To enable reranking, pass --reranker <model>." in output
    assert "Available reranker models: Qwen/Qwen3-Reranker-4B, Qwen/Qwen3-Reranker-8B" in output
    assert "Query LLM: OpenAI / gpt-4.1-mini" in output
    assert "Embedding:" not in output
    assert "Provider:" not in output
    assert "Model:" not in output
    detail_path = evaluate.RESULTS_DIR / "eval_detail__test-tag.csv"
    summary_path = evaluate.RESULTS_DIR / "eval_summary__test-tag.json"
    assert detail_path.exists()
    assert summary_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["table_recall@k"] == 1.0
    assert summary["column_recall@k"] == 1.0
    assert summary["reranker_model"] is None


def test_main_prints_selected_reranker_without_no_rerank_hint(monkeypatch, tmp_path, capsys):
    from db_rag.benchmark import evaluate

    benchmark = tmp_path / "benchmark.csv"
    benchmark.write_text(
        "\n".join(
            [
                "question,expected_tables,expected_columns,difficulty",
                "Which fields track age?,Form 1A,AGE,easy",
            ]
        ),
        encoding="utf-8",
    )

    def retriever(question: str, *, debug: bool = False):
        return (
            [{"table": "Form 1A", "text": "table context"}],
            [{"table": "Form 1A", "column": "AGE", "text": "age column"}],
        )

    monkeypatch.setattr(evaluate, "build_runtime_retriever", lambda args: retriever)
    monkeypatch.setattr(evaluate, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(evaluate, "resolve_output_tag", lambda args: "test-tag")
    monkeypatch.setattr(evaluate, "_default_model_name", lambda provider, base_url: "gpt-4.1-mini", raising=False)
    monkeypatch.setattr(evaluate, "ensure_prebuilt_assets", lambda indexing_model: None)

    exit_code = evaluate.main(
        [
            "--benchmark",
            str(benchmark),
            "--indexing-model",
            "Qwen/Qwen3-Embedding-4B",
            "--reranker",
            "Qwen/Qwen3-Reranker-4B",
        ]
    )
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Reranker: Qwen/Qwen3-Reranker-4B" in output
    assert "To enable reranking, pass --reranker <model>." not in output
    assert "Available reranker models:" not in output


def test_build_runtime_retriever_uses_current_db_rag_runtime(monkeypatch):
    from db_rag.benchmark import adapter

    calls: dict[str, object] = {}

    class _QuickTest:
        @staticmethod
        def build_runtime_llm(**kwargs):
            calls["llm_kwargs"] = kwargs
            return "runtime-llm"

    class _PersistentClient:
        def __init__(self, *, path):
            calls["client_path"] = path

        def get_collection(self, name, embedding_function):
            calls.setdefault("collections", []).append((name, embedding_function))
            return f"{name}-collection"

    def fake_embedding_function(*, model):
        calls["embedding_model"] = model
        return "embedding-function"

    def fake_retrieve_context_records(
        llm,
        table_collection,
        column_collection,
        question,
        *,
        reranker_model="none",
        debug=False,
    ):
        calls["retrieval_args"] = {
            "llm": llm,
            "table_collection": table_collection,
            "column_collection": column_collection,
            "question": question,
            "reranker_model": reranker_model,
            "debug": debug,
        }
        return ([{"table": "Form 1A", "text": "table context"}], [])

    monkeypatch.setattr(adapter, "_get_quick_test_module", lambda: _QuickTest)
    monkeypatch.setattr(adapter, "_get_chromadb_module", lambda: type("_Chromadb", (), {"PersistentClient": _PersistentClient}))
    monkeypatch.setattr(adapter, "_get_openai_embedding_function_class", lambda: fake_embedding_function)
    monkeypatch.setattr(adapter, "_get_retrieve_context_records", lambda: fake_retrieve_context_records)
    monkeypatch.setattr(adapter, "chroma_dir_for_model", lambda model: f"/tmp/{model}")
    monkeypatch.setattr(adapter, "load_manifest_for_model", lambda model: {"embedding_model": model})

    retriever = adapter.build_runtime_retriever(indexing_model="Qwen/Qwen3-Embedding-4B")
    table_hits, column_hits = retriever("age", debug=True)

    assert table_hits == [{"table": "Form 1A", "text": "table context"}]
    assert column_hits == []
    assert calls["embedding_model"] == "Qwen/Qwen3-Embedding-4B"
    assert calls["client_path"] == "/tmp/Qwen/Qwen3-Embedding-4B"
    assert calls["retrieval_args"] == {
        "llm": "runtime-llm",
        "table_collection": "table_summaries-collection",
        "column_collection": "column_chunks-collection",
        "question": "age",
        "reranker_model": None,
        "debug": True,
    }


def test_evaluate_main_exits_when_prebuilt_assets_are_missing(monkeypatch, capsys):
    from db_rag.benchmark import evaluate

    monkeypatch.setattr(
        evaluate,
        "ensure_prebuilt_assets",
        lambda indexing_model: (_ for _ in ()).throw(RuntimeError("missing assets message")),
    )

    with pytest.raises(SystemExit) as excinfo:
        evaluate.main(["--indexing-model", "Qwen/Qwen3-Embedding-4B"])

    assert excinfo.value.code == 2
    assert "missing assets message" in capsys.readouterr().err
