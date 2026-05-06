from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db_rag.service import DbRagService
from db_rag.service.models import ColumnSelectionCandidate


class _CaptureSqlLLM:
    def __init__(self) -> None:
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return type("Response", (), {"content": "SELECT 1"})()


def test_prepare_sql_candidate_includes_stored_column_profile(monkeypatch) -> None:
    def fake_profile(table: str, column: str) -> dict[str, object]:
        assert table == "Form 2A - INDEX CASE: Clinical/Demographic Form"
        assert column == "IC_DMDX"
        return {
            "stored_dtype": "VARCHAR",
            "stored_samples": ["Don't know", "No", "Yes"],
        }

    monkeypatch.setattr("db_rag.service.sql_service._duckdb_column_profile", fake_profile, raising=False)

    llm = _CaptureSqlLLM()
    service = DbRagService(llm=llm)
    selection = ColumnSelectionCandidate(
        selection_id="sel-diabetes",
        question="Subset index cases by diabetes status.",
        tables=["Form 2A - INDEX CASE: Clinical/Demographic Form"],
        columns=[
            {
                "table": "Form 2A - INDEX CASE: Clinical/Demographic Form",
                "column": "IC_DMDX",
                "description": "Have you ever been diagnosed with Diabetes Mellitus?",
            }
        ],
        rationale="Needed diabetes status.",
        status="approved",
    )

    service.prepare_sql_candidate("Subset index cases by diabetes status.", selection)

    prompt = llm.messages[1].content
    assert '"stored_dtype": "VARCHAR"' in prompt
    assert '"stored_samples": [' in prompt
    assert '"Don\'t know"' in prompt
    assert '"No"' in prompt
    assert '"Yes"' in prompt
