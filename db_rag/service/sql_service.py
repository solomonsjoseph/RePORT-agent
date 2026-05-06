from __future__ import annotations

import json
from typing import Any

from utils.llm_response import coerce_text_content

from .errors import DbRagUnanswerableError
from .models import ColumnSelectionCandidate, PreparedSqlCandidate, SqlExecutionResult
from ..config import DUCKDB_PATH
from ..generation import (
    build_sql_policy_text,
    default_selection_id,
    extract_sql,
    is_unanswerable_response,
    validate_sql,
)
from .schema import _lookup_schema_variable_metadata


def _quote_duckdb_identifier(identifier: str) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'


def _duckdb_column_profile(table: str, column: str) -> dict[str, object]:
    import duckdb

    if not DUCKDB_PATH.exists():
        return {}

    quoted_table = _quote_duckdb_identifier(table)
    quoted_column = _quote_duckdb_identifier(column)
    db = duckdb.connect(str(DUCKDB_PATH), read_only=True)
    try:
        described = db.execute(f"DESCRIBE SELECT {quoted_column} FROM {quoted_table} LIMIT 0").fetchall()
        samples = db.execute(
            f"SELECT DISTINCT {quoted_column} FROM {quoted_table} "
            f"WHERE {quoted_column} IS NOT NULL LIMIT 8"
        ).fetchall()
    finally:
        db.close()

    profile: dict[str, object] = {}
    if described:
        profile["stored_dtype"] = str(described[0][1])
    if samples:
        profile["stored_samples"] = [row[0] for row in samples]
    return profile


def _approved_columns_for_sql_prompt(columns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    approved_columns: list[dict[str, Any]] = []
    for column in columns:
        table = str(column["table"])
        column_name = str(column["column"])
        entry: dict[str, Any] = {
            "table": table,
            "column": column_name,
            "description": str(column.get("description", "") or ""),
        }
        schema_meta = _lookup_schema_variable_metadata(table, column_name) or {}
        for field in ("values", "depends_on", "condition", "section_context"):
            value = schema_meta.get(field)
            if value is not None and value != "":
                entry[field] = value
        try:
            entry.update(_duckdb_column_profile(table, column_name))
        except Exception:
            pass
        approved_columns.append(entry)
    return approved_columns


class DbRagSqlMixin:
    llm: Any

    def prepare_sql_candidate(
        self,
        question: str,
        approved_selection: ColumnSelectionCandidate,
    ) -> PreparedSqlCandidate:
        from langchain_core.messages import HumanMessage, SystemMessage

        if approved_selection.status != "approved":
            raise ValueError("prepare_sql_candidate requires an approved column selection.")

        approved_tables = list(approved_selection.tables)
        candidate_columns = [
            {
                "table": str(column["table"]),
                "column": str(column["column"]),
                "description": str(column.get("description", "") or ""),
            }
            for column in approved_selection.columns
        ]
        prompt_columns = _approved_columns_for_sql_prompt(candidate_columns)
        response = self.llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are a DuckDB SQL expert for the RePORT clinical research database. "
                        "Use only the approved tables and approved columns listed in the prompt. "
                        "Do not reference any other tables or columns. "
                        "Return only read-only DuckDB SQL using SELECT or WITH. "
                        "Never emit mutating or DDL statements. "
                        "If the approved schema cannot answer the question, return exactly "
                        "'UNANSWERABLE: <brief reason>'.\n\n"
                        f"{build_sql_policy_text()}"
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question:\n{question}\n\n"
                        f"Approved tables:\n{json.dumps(approved_tables, indent=2)}\n\n"
                        f"Approved columns:\n{json.dumps(prompt_columns, indent=2, sort_keys=True)}"
                    )
                ),
            ]
        )
        sql = extract_sql(coerce_text_content(getattr(response, "content", "")))
        if is_unanswerable_response(sql):
            raise DbRagUnanswerableError(sql)
        valid, error = validate_sql(sql)
        if not valid:
            repair_seed = PreparedSqlCandidate(
                question=question,
                sql=sql,
                tables=approved_tables,
                columns=candidate_columns,
                selection_id=approved_selection.selection_id,
            )
            return self.repair_prepared_sql_candidate(repair_seed, error or "SQL validation failed.")
        return PreparedSqlCandidate(
            question=question,
            sql=sql,
            tables=approved_tables,
            columns=candidate_columns,
            selection_id=approved_selection.selection_id,
        )

    def execute_prepared_sql(self, candidate: PreparedSqlCandidate) -> SqlExecutionResult:
        import duckdb

        valid, error = validate_sql(candidate.sql)
        if not valid:
            raise ValueError(error or "SQL validation failed.")

        db = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        try:
            dataframe = db.execute(candidate.sql).fetchdf()
        finally:
            db.close()

        return SqlExecutionResult(
            answer=f"Read-only SQL execution completed with {len(dataframe)} result row(s).",
            sql=candidate.sql,
            dataframe=dataframe,
            source_tables=list(candidate.tables),
        )

    def repair_prepared_sql_candidate(self, candidate: PreparedSqlCandidate, error_message: str) -> PreparedSqlCandidate:
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt_columns = _approved_columns_for_sql_prompt([dict(column) for column in candidate.columns])
        response = self.llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are fixing a DuckDB SQL query for the RePORT clinical research database. "
                        "Use only the approved tables and approved columns. "
                        "Return only read-only DuckDB SQL using SELECT or WITH. "
                        "Never emit mutating or DDL statements.\n\n"
                        f"{build_sql_policy_text()}"
                    )
                ),
                HumanMessage(
                    content=(
                        f"Original question:\n{candidate.question}\n\n"
                        f"Approved tables:\n{json.dumps(list(candidate.tables), indent=2)}\n\n"
                        f"Approved columns:\n{json.dumps(prompt_columns, indent=2, sort_keys=True)}\n\n"
                        f"SQL that failed:\n{candidate.sql}\n\n"
                        f"Execution/validation error:\n{error_message}\n\n"
                        "Fix the SQL and return only the corrected SQL."
                    )
                ),
            ]
        )
        repaired_sql = extract_sql(coerce_text_content(getattr(response, "content", "")))
        if is_unanswerable_response(repaired_sql):
            raise DbRagUnanswerableError(repaired_sql)
        valid, error = validate_sql(repaired_sql)
        if not valid:
            raise ValueError(error or "SQL validation failed.")
        return PreparedSqlCandidate(
            question=candidate.question,
            sql=repaired_sql,
            tables=list(candidate.tables),
            columns=[dict(column) for column in candidate.columns],
            selection_id=candidate.selection_id,
            status=candidate.status,
        )

    def execute_sql_flow(
        self,
        question: str,
        *,
        debug: bool = False,
        reranker_model: str | None = None,
    ) -> dict[str, Any]:
        context = self.retrieve_context(question, debug=debug, reranker_model=reranker_model)
        approved_selection = ColumnSelectionCandidate(
            selection_id=default_selection_id(
                question,
                context.table_names,
                [{"table": entry.table, "column": entry.column} for entry in context.columns],
                [],
            ),
            question=question,
            tables=context.table_names,
            columns=[
                {"table": entry.table, "column": entry.column, "description": entry.text}
                for entry in context.columns
            ],
            rationale="Legacy execute_sql_flow compatibility path.",
            status="approved",
        )
        debug_payload = (
            {
                "question": question,
                "retrieved_tables": context.table_names,
                "retrieved_columns": [entry.as_prompt_line() for entry in context.columns],
            }
            if debug
            else {}
        )
        max_retries = 2
        candidate: PreparedSqlCandidate | None = None
        result: SqlExecutionResult | None = None
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                if candidate is None:
                    candidate = self.prepare_sql_candidate(question, approved_selection)
                result = self.execute_prepared_sql(candidate)
                break
            except DbRagUnanswerableError as exc:
                return {
                    "answer": str(exc),
                    "sql": "",
                    "dataframe": None,
                    "source_tables": context.table_names,
                    "debug": debug_payload,
                }
            except ValueError as exc:
                last_error = exc
                if attempt >= max_retries:
                    raise
                repair_seed = candidate or PreparedSqlCandidate(
                    question=question,
                    sql="SELECT 1",
                    tables=list(approved_selection.tables),
                    columns=[dict(column) for column in approved_selection.columns],
                    selection_id=approved_selection.selection_id,
                )
                candidate = self.repair_prepared_sql_candidate(repair_seed, str(exc))
            except Exception as exc:
                last_error = exc
                if candidate is None or attempt >= max_retries:
                    raise
                candidate = self.repair_prepared_sql_candidate(candidate, str(exc))
        if result is None or candidate is None:
            if last_error is not None:
                raise last_error
            raise ValueError("SQL execution did not produce a result.")
        if debug:
            debug_payload["sql_tables"] = list(candidate.tables)
            debug_payload["sql_columns"] = [f"{column['table']}.{column['column']}" for column in candidate.columns]
        return {
            "answer": result.answer,
            "sql": result.sql,
            "dataframe": result.dataframe,
            "source_tables": result.source_tables,
            "debug": debug_payload,
        }
