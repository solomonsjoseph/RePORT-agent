from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _install_langchain_message_stubs() -> None:
    langchain_core = sys.modules.get("langchain_core")
    if langchain_core is None:
        langchain_core = ModuleType("langchain_core")
    messages = sys.modules.get("langchain_core.messages")
    if messages is None:
        messages = ModuleType("langchain_core.messages")

    class BaseMessage:
        def __init__(self, content: str = "", additional_kwargs: dict | None = None) -> None:
            self.content = content
            self.additional_kwargs = dict(additional_kwargs or {})

    class HumanMessage(BaseMessage):
        type = "human"

    class SystemMessage(BaseMessage):
        type = "system"

    messages.BaseMessage = BaseMessage
    messages.HumanMessage = HumanMessage
    messages.SystemMessage = SystemMessage
    langchain_core.messages = messages
    sys.modules["langchain_core"] = langchain_core
    sys.modules["langchain_core.messages"] = messages


_install_langchain_message_stubs()

from db_rag.service.models import DbRagColumnHit, DbRagContext, DbRagTableHit, FeedbackConstraintSet
from db_rag.service.selection_service import DbRagSelectionMixin, _OpenAIStructuredColumnSelector


class _SelectionService(DbRagSelectionMixin):
    llm = object()

    def __init__(self, ranked_output=None) -> None:
        self.ranked_output = ranked_output

    def ground_feedback_constraints(self, *args, **kwargs) -> FeedbackConstraintSet:  # type: ignore[override]
        return FeedbackConstraintSet(goal_text="")

    def _rank_columns_with_openai(self, **kwargs):  # type: ignore[override]
        del kwargs
        return self.ranked_output


class _FakeStructuredSelectorClient:
    last_request = None

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        _FakeStructuredSelectorClient.last_request = kwargs
        response_content = (
            '{"columns":["Form 2A||IC_AGE","Form 6||HIV_STATUS"],'
            '"rationale":"Ranked exact identifiers."}'
        )
        message = type("Message", (), {"content": response_content})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class _FakeOpenAISelector(_OpenAIStructuredColumnSelector):
    @staticmethod
    def _resolve_client():
        return _FakeStructuredSelectorClient


def _context() -> DbRagContext:
    return DbRagContext(
        tables=[
            DbRagTableHit(table="Form 2A", text="Index case demographics"),
            DbRagTableHit(table="Form 6", text="HIV treatment form"),
        ],
        columns=[
            DbRagColumnHit(table="Form 2A", column="IC_AGE", text="Age in years"),
            DbRagColumnHit(table="Form 2A", column="IC_GENDER", text="Gender"),
            DbRagColumnHit(table="Form 6", column="HIV_STATUS", text="HIV status"),
        ],
        table_context="Form 2A\n\nForm 6",
        column_context="IC_AGE\n\nIC_GENDER\n\nHIV_STATUS",
    )


def test_openai_structured_selector_constrains_columns_to_exact_candidate_identifier_enum(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    selector = _FakeOpenAISelector(resolve_model=lambda: "gpt-4o-mini")

    result = selector.rank_columns(
        question="Rank variables.",
        candidate_columns=[
            {"table": "Form 2A", "column": "IC_AGE", "description": "Age in years"},
            {"table": "Form 6", "column": "HIV_STATUS", "description": "HIV status"},
        ],
        feedback_history=[],
    )

    schema = _FakeStructuredSelectorClient.last_request["response_format"]["json_schema"]["schema"]
    item_schema = schema["properties"]["columns"]["items"]
    assert item_schema["enum"] == ["Form 2A||IC_AGE", "Form 6||HIV_STATUS"]
    assert result["columns"] == ["Form 2A||IC_AGE", "Form 6||HIV_STATUS"]


def test_prepare_column_selection_skips_selector_for_tiny_candidate_pool() -> None:
    service = _SelectionService(
        ranked_output={
            "columns": ["Form 2A||IC_GENDER", "Form 2A||IC_AGE"],
            "rationale": "ignored",
            "raw_model_output": "{}",
        }
    )
    context = DbRagContext(
        tables=[DbRagTableHit(table="Form 2A", text="Index case demographics")],
        columns=[
            DbRagColumnHit(table="Form 2A", column="IC_AGE", text="Age in years"),
            DbRagColumnHit(table="Form 2A", column="IC_GENDER", text="Gender"),
        ],
        table_context="Form 2A",
        column_context="IC_AGE\n\nIC_GENDER",
    )

    selection = service.prepare_column_selection("Subset index cases.", context)

    assert selection.selection_source == "deterministic_constrained"
    assert [column["column"] for column in selection.columns] == ["IC_AGE", "IC_GENDER"]
    assert selection.tables == ["Form 2A"]


def test_prepare_column_selection_uses_valid_ranked_columns_and_marks_partial_output() -> None:
    service = _SelectionService(
        ranked_output={
            "columns": [
                "Form 2A||IC_GENDER",
                "Missing||BAD",
                "Form 2A||IC_GENDER",
                "Form 2A||IC_AGE",
            ],
            "rationale": "",
            "raw_model_output": '{"columns":["Form 2A||IC_GENDER","Missing||BAD"]}',
        }
    )

    selection = service.prepare_column_selection("Study treatment failure.", _context())

    assert selection.selection_source == "openai_ranked_partial"
    assert [column["column"] for column in selection.columns] == ["IC_GENDER", "IC_AGE"]
    assert selection.tables == ["Form 2A", "Form 6"]
    assert selection.raw_model_output
    assert selection.rationale


def test_prepare_column_selection_falls_back_to_retrieval_pool_when_ranked_output_is_unusable() -> None:
    service = _SelectionService(
        ranked_output={
            "columns": ["Missing||BAD"],
            "rationale": "bad output",
            "raw_model_output": '{"columns":["Missing||BAD"]}',
        }
    )

    selection = service.prepare_column_selection("Study treatment failure.", _context())

    assert selection.selection_source == "retrieval_fallback"
    assert [f'{column["table"]}||{column["column"]}' for column in selection.columns] == [
        "Form 2A||IC_AGE",
        "Form 2A||IC_GENDER",
        "Form 6||HIV_STATUS",
    ]
    assert selection.fallback_reason
    assert "Missing||BAD" in selection.fallback_reason
    assert selection.raw_model_output == '{"columns":["Missing||BAD"]}'


def test_prepare_column_selection_includes_rank_failure_reason_in_fallback() -> None:
    service = _SelectionService(
        ranked_output={
            "columns": [],
            "rationale": "",
            "raw_model_output": "",
            "error": "selection ranking request failed: rate limit",
        }
    )

    selection = service.prepare_column_selection("Study treatment failure.", _context())

    assert selection.selection_source == "retrieval_fallback"
    assert "rate limit" in selection.fallback_reason


def test_prepare_column_selection_excludes_columns_missing_from_runtime_duckdb_schema(monkeypatch, tmp_path) -> None:
    duckdb_path = tmp_path / "report.duckdb"
    duckdb_path.touch()
    table_name = "Form 12A - Index Case Follow-Up Visit Form"

    class FakeDuckDbConnection:
        def execute(self, sql: str):
            if "FUA_VISTYPE" in sql:
                raise Exception('Binder Error: Referenced column "FUA_VISTYPE" not found in FROM clause!')
            return self

        def close(self) -> None:
            pass

    fake_duckdb = ModuleType("duckdb")
    fake_duckdb.connect = lambda *_args, **_kwargs: FakeDuckDbConnection()
    monkeypatch.setattr("db_rag.service.runtime_schema.DUCKDB_PATH", duckdb_path, raising=False)
    monkeypatch.setitem(sys.modules, "duckdb", fake_duckdb)

    service = _SelectionService()
    context = DbRagContext(
        tables=[DbRagTableHit(table=table_name, text="Index case follow-up")],
        columns=[
            DbRagColumnHit(table=table_name, column="FUA_VISTYPE", text="Schema-only visit type"),
            DbRagColumnHit(table=table_name, column="FUA_VISIT", text="Runtime visit type"),
        ],
        table_context=table_name,
        column_context="FUA_VISTYPE\n\nFUA_VISIT",
    )

    selection = service.prepare_column_selection("Extract visit type observations.", context)

    assert [column["column"] for column in selection.columns] == ["FUA_VISIT"]
    assert selection.tables == [table_name]
