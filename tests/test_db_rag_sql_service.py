from __future__ import annotations

from pathlib import Path
from types import ModuleType
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _ensure_langchain_core_stubs() -> None:
    langchain_core = sys.modules.get("langchain_core") or ModuleType("langchain_core")
    messages = sys.modules.get("langchain_core.messages") or ModuleType("langchain_core.messages")

    class BaseMessage:
        def __init__(self, content: str, additional_kwargs: dict | None = None) -> None:
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


_ensure_langchain_core_stubs()

from db_rag.service import DbRagService
from db_rag.service.models import ColumnSelectionCandidate
from utils.performance import collect_timings


class _CaptureSqlLLM:
    def __init__(self) -> None:
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        table_name = "Form 2A - INDEX CASE: Clinical/Demographic Form"
        return type("Response", (), {"content": f'SELECT "IC_DMDX" FROM "{table_name}"'})()


class _SequenceSqlLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.messages = []

    def invoke(self, messages):
        self.messages.append(messages)
        return type("Response", (), {"content": self.responses.pop(0)})()


class _FailingSqlLLM:
    def invoke(self, _messages):
        raise AssertionError("LLM should not be called")


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


def test_prepare_sql_candidate_repairs_metadata_only_sql(monkeypatch) -> None:
    profile_calls = []

    def fake_profile(table: str, column: str) -> dict[str, object]:
        profile_calls.append((table, column))
        return {}

    monkeypatch.setattr("db_rag.service.sql_service._duckdb_column_profile", fake_profile, raising=False)

    table_name = "Form 2A - INDEX CASE: Clinical/Demographic Form"
    llm = _SequenceSqlLLM(
        [
            (
                "SELECT 'Diabetes status' AS variable, "
                "'Form 2A - INDEX CASE: Clinical/Demographic Form' AS table_name, "
                "'IC_DMDX' AS column_name"
            ),
            f'SELECT "IC_DMDX" FROM "{table_name}"',
        ]
    )
    service = DbRagService(llm=llm)
    selection = ColumnSelectionCandidate(
        selection_id="sel-diabetes",
        question="Extract observations for diabetes status.",
        tables=[table_name],
        columns=[
            {
                "table": table_name,
                "column": "IC_DMDX",
                "description": "Have you ever been diagnosed with Diabetes Mellitus?",
            }
        ],
        rationale="Needed diabetes status.",
        status="approved",
    )

    candidate = service.prepare_sql_candidate("Extract observations for diabetes status.", selection)

    assert candidate.sql == f'SELECT "IC_DMDX" FROM "{table_name}"'
    repair_prompt = llm.messages[1][1].content
    assert "row-level observations" in repair_prompt
    assert "must read from at least one approved source table" in repair_prompt
    assert profile_calls == [(table_name, "IC_DMDX")]


def test_prepare_sql_candidate_repairs_sql_with_missing_runtime_column(monkeypatch, tmp_path) -> None:
    duckdb_path = tmp_path / "report.duckdb"
    duckdb_path.touch()
    table_name = "Form 12A - Index Case Follow-Up Visit Form"

    class FakeDuckDbConnection:
        def execute(self, sql: str):
            if "FUA_VISTYPE" in sql:
                raise Exception(
                    'Binder Error: Referenced column "FUA_VISTYPE" not found in FROM clause! '
                    'Candidate bindings: "FUA_VISIT"'
                )
            return self

        def close(self) -> None:
            pass

    fake_duckdb = ModuleType("duckdb")
    fake_duckdb.connect = lambda *_args, **_kwargs: FakeDuckDbConnection()

    monkeypatch.setattr("db_rag.service.sql_service.DUCKDB_PATH", duckdb_path, raising=False)
    monkeypatch.setattr("db_rag.service.sql_service._duckdb_column_profile", lambda *_args: {}, raising=False)
    monkeypatch.setitem(sys.modules, "duckdb", fake_duckdb)

    llm = _SequenceSqlLLM(
        [
            f'SELECT "FUA_VISTYPE" FROM "{table_name}"',
            f'SELECT "FUA_VISIT" FROM "{table_name}"',
        ]
    )
    service = DbRagService(llm=llm)
    selection = ColumnSelectionCandidate(
        selection_id="sel-follow-up",
        question="Extract visit type observations.",
        tables=[table_name],
        columns=[
            {
                "table": table_name,
                "column": "FUA_VISTYPE",
                "description": "Visit type from schema metadata.",
            },
            {
                "table": table_name,
                "column": "FUA_VISIT",
                "description": "Visit type from DuckDB runtime.",
            }
        ],
        rationale="Needed visit type.",
        status="approved",
    )

    candidate = service.prepare_sql_candidate("Extract visit type observations.", selection)

    assert candidate.sql == f'SELECT "FUA_VISIT" FROM "{table_name}"'
    assert len(llm.messages) == 2
    repair_prompt = llm.messages[1][1].content
    assert "FUA_VISTYPE" in repair_prompt
    assert "DuckDB runtime schema" in repair_prompt


def test_prepare_sql_candidate_preflights_multi_table_without_join_keys(monkeypatch) -> None:
    monkeypatch.setattr("db_rag.service.sql_service._duckdb_column_profile", lambda *_args: {}, raising=False)

    service = DbRagService(llm=_FailingSqlLLM())
    selection = ColumnSelectionCandidate(
        selection_id="sel-multi-no-join",
        question="Extract diabetes and final outcome observations.",
        tables=[
            "Form 2A - INDEX CASE: Clinical/Demographic Form",
            "Final Outcome Determination Form - Cohort A (Active Pulmonary TB, index case)",
        ],
        columns=[
            {
                "table": "Form 2A - INDEX CASE: Clinical/Demographic Form",
                "column": "IC_DMDX",
                "description": "Have you ever been diagnosed with Diabetes Mellitus?",
            },
            {
                "table": "Final Outcome Determination Form - Cohort A (Active Pulmonary TB, index case)",
                "column": "FOA_COHAOUT",
                "description": "Cohort A final outcome.",
            },
        ],
        rationale="Needed diabetes status and outcome.",
        status="approved",
    )

    try:
        service.prepare_sql_candidate("Extract diabetes and final outcome observations.", selection)
    except Exception as exc:
        error = exc
    else:  # pragma: no cover
        raise AssertionError("Expected preflight error")

    assert "approved join key" in str(error)


def test_prepare_sql_candidate_allows_multi_table_with_join_keys(monkeypatch) -> None:
    monkeypatch.setattr("db_rag.service.sql_service._duckdb_column_profile", lambda *_args: {}, raising=False)

    first_table = "Form 2A - INDEX CASE: Clinical/Demographic Form"
    second_table = "Final Outcome Determination Form - Cohort A (Active Pulmonary TB, index case)"
    llm = _SequenceSqlLLM(
        [
            (
                f'SELECT a."SUBJID_PSEUDO", a."IC_DMDX", b."FOA_COHAOUT" '
                f'FROM "{first_table}" a JOIN "{second_table}" b '
                f'ON a."SUBJID_PSEUDO" = b."SUBJID_PSEUDO"'
            )
        ]
    )
    service = DbRagService(llm=llm)
    selection = ColumnSelectionCandidate(
        selection_id="sel-multi-join",
        question="Extract diabetes and final outcome observations.",
        tables=[first_table, second_table],
        columns=[
            {"table": first_table, "column": "SUBJID_PSEUDO", "description": "Subject ID."},
            {"table": first_table, "column": "IC_DMDX", "description": "Diabetes status."},
            {"table": second_table, "column": "SUBJID_PSEUDO", "description": "Subject ID."},
            {"table": second_table, "column": "FOA_COHAOUT", "description": "Cohort A final outcome."},
        ],
        rationale="Needed diabetes status and outcome.",
        status="approved",
    )

    candidate = service.prepare_sql_candidate("Extract diabetes and final outcome observations.", selection)

    assert "JOIN" in candidate.sql


def test_prepare_sql_candidate_records_preflight_and_repair_timers(monkeypatch) -> None:
    monkeypatch.setattr("db_rag.service.sql_service._duckdb_column_profile", lambda *_args: {}, raising=False)

    table_name = "Form 2A - INDEX CASE: Clinical/Demographic Form"
    llm = _SequenceSqlLLM(
        [
            (
                "SELECT 'Diabetes status' AS variable, "
                "'Form 2A - INDEX CASE: Clinical/Demographic Form' AS table_name, "
                "'IC_DMDX' AS column_name"
            ),
            f'SELECT "IC_DMDX" FROM "{table_name}"',
        ]
    )
    service = DbRagService(llm=llm)
    selection = ColumnSelectionCandidate(
        selection_id="sel-diabetes",
        question="Extract observations for diabetes status.",
        tables=[table_name],
        columns=[
            {
                "table": table_name,
                "column": "IC_DMDX",
                "description": "Have you ever been diagnosed with Diabetes Mellitus?",
            }
        ],
        rationale="Needed diabetes status.",
        status="approved",
    )

    with collect_timings() as records:
        service.prepare_sql_candidate("Extract observations for diabetes status.", selection)

    stages = [record["stage"] for record in records]
    assert "db_rag.sql.preflight" in stages
    assert "db_rag.sql.generate_sql" in stages
    assert "db_rag.sql.repair_sql" in stages
    assert "db_rag.sql.repair_generate_sql" in stages
