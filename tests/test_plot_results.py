from __future__ import annotations

import json


def test_infer_summary_and_output_paths(tmp_path):
    from db_rag.benchmark import plot_results

    detail_path = tmp_path / "test-tag" / "details.csv"

    summary_path = plot_results.infer_summary_path(detail_path)
    output_path = plot_results.infer_output_path(detail_path)

    assert summary_path == tmp_path / "test-tag" / "summary.json"
    assert output_path == tmp_path / "test-tag" / "figure.png"


def test_infer_detail_path(tmp_path):
    from db_rag.benchmark import plot_results

    run_dir = tmp_path / "test-tag"

    assert plot_results.infer_detail_path(run_dir) == run_dir / "details.csv"


def test_compute_plot_metrics_by_difficulty(tmp_path):
    from db_rag.benchmark import plot_results

    detail_path = tmp_path / "test-tag" / "details.csv"
    detail_path.parent.mkdir(parents=True)
    detail_path.write_text(
        "\n".join(
            [
                "question,difficulty,expected_tables,predicted_tables,table_hit,table_rank,expected_columns,predicted_columns,matched_columns,column_recall,column_precision,column_mrr,error",
                "Q1,easy,T1,T1,True,1,C1,C1,C1,1.0,0.5,1.0,",
                "Q2,easy,T2,T2,True,2,C2,CX,,0.0,0.0,0.0,",
                "Q3,hard,T3,T3,True,1,C3,C3,C3,1.0,1.0,1.0,",
                "Q4,subset,T4,,N/A,N/A,C4,,,N/A,N/A,N/A,connection error",
            ]
        ),
        encoding="utf-8",
    )

    metrics = plot_results.compute_plot_metrics(detail_path)

    assert metrics["overall"]["count"] == 3
    assert round(metrics["overall"]["column_recall"], 4) == 0.6667
    assert round(metrics["overall"]["column_precision"], 4) == 0.5
    assert round(metrics["overall"]["column_f1"], 4) == 0.5556

    assert metrics["by_difficulty"]["easy"]["count"] == 2
    assert metrics["by_difficulty"]["hard"]["count"] == 1
    assert "subset" not in metrics["by_difficulty"]
    assert round(metrics["by_difficulty"]["easy"]["column_f1"], 4) == 0.3333


def test_render_benchmark_figure_writes_png(tmp_path):
    from db_rag.benchmark import plot_results

    run_dir = tmp_path / "test-tag"
    run_dir.mkdir()
    detail_path = run_dir / "details.csv"
    summary_path = run_dir / "summary.json"
    output_path = run_dir / "figure.png"

    detail_path.write_text(
        "\n".join(
            [
                "question,difficulty,expected_tables,predicted_tables,table_hit,table_rank,expected_columns,predicted_columns,matched_columns,column_recall,column_precision,column_mrr,error",
                "Q1,easy,T1,T1,True,1,C1,C1,C1,1.0,0.5,1.0,",
                "Q2,medium,T2,T2,True,2,C2,C2,C2,1.0,0.25,0.5,",
                "Q3,hard,T3,T3,False,,C3,CX,,0.0,0.0,0.0,",
                "Q4,subset,T4,T4,True,3,C4,C4,C4,1.0,0.2,0.3333,",
            ]
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "total_questions": 4,
                "scored_questions": 4,
                "unanswerable": 0,
                "table_recall@4": 0.75,
                "table_mrr": 0.4583,
                "column_recall@12": 0.75,
                "column_precision@12": 0.2375,
                "column_mrr": 0.4583,
                "embedding_model": "Qwen/Qwen3-Embedding-4B",
                "reranker_model": "cohere/rerank-v3.5",
                "provider": "openai",
                "model": "gpt-5.4",
            }
        ),
        encoding="utf-8",
    )

    plot_results.render_benchmark_figure(detail_path=detail_path, summary_path=summary_path, output_path=output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_render_benchmark_figure_uses_explicit_metric_keys_and_titles(tmp_path, monkeypatch):
    from db_rag.benchmark import plot_results

    run_dir = tmp_path / "test-tag"
    run_dir.mkdir()
    detail_path = run_dir / "details.csv"
    summary_path = run_dir / "summary.json"
    output_path = run_dir / "figure.png"

    detail_path.write_text(
        "\n".join(
            [
                "question,difficulty,expected_tables,predicted_tables,table_hit,table_rank,expected_columns,predicted_columns,matched_columns,column_recall,column_precision,column_mrr,error",
                "Q1,easy,T1,T1,True,1,C1,C1,C1,1.0,0.5,1.0,",
                "Q2,medium,T2,T2,True,2,C2,C2,C2,1.0,0.25,0.5,",
                "Q3,hard,T3,T3,False,,C3,CX,,0.0,0.0,0.0,",
            ]
        ),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            {
                "total_questions": 3,
                "scored_questions": 3,
                "unanswerable": 0,
                "table_recall@4": 0.67,
                "table_mrr": 0.5,
                "column_recall@12": 0.6667,
                "column_precision@12": 0.25,
                "column_mrr": 0.5,
                "embedding_model": "voyage-4-large",
                "reranker_model": "voyage/rerank-2.5",
                "provider": "anthropic",
                "model": "claude-opus-4-1-20250805",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(plot_results.plt, "close", lambda fig: None)

    plot_results.render_benchmark_figure(detail_path=detail_path, summary_path=summary_path, output_path=output_path)

    figure = plot_results.plt.gcf()
    header_text = " ".join(text.get_text() for text in figure.texts)
    axis_titles = [axis.get_title(loc="left") for axis in figure.axes]

    assert "Table Recall@4 = 0.67" in header_text
    assert axis_titles == ["Column Recall@12", "Column Precision@12", "Column F1@12"]


def test_main_supports_run_dir(tmp_path, capsys):
    from db_rag.benchmark import plot_results

    run_dir = tmp_path / "test-tag"
    run_dir.mkdir()
    (run_dir / "details.csv").write_text(
        "\n".join(
            [
                "question,difficulty,expected_tables,predicted_tables,table_hit,table_rank,expected_columns,predicted_columns,matched_columns,column_recall,column_precision,column_mrr,error",
                "Q1,easy,T1,T1,True,1,C1,C1,C1,1.0,1.0,1.0,",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "total_questions": 1,
                "scored_questions": 1,
                "unanswerable": 0,
                "table_recall@4": 1.0,
                "table_mrr": 1.0,
                "column_recall@12": 1.0,
                "column_precision@12": 1.0,
                "column_mrr": 1.0,
                "embedding_model": "Qwen/Qwen3-Embedding-4B",
                "reranker_model": None,
                "provider": "openai",
                "model": "gpt-5.4",
            }
        ),
        encoding="utf-8",
    )

    exit_code = plot_results.main(["--run-dir", str(run_dir)])
    output = capsys.readouterr().out

    assert exit_code == 0
    assert "Figure saved to" in output
    assert (run_dir / "figure.png").exists()
