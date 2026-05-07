from __future__ import annotations

import json
import re
from typing import Any

from utils.llm_response import coerce_text_content
from utils.performance import timing_stage

from .errors import DbRagUnanswerableError
from .models import ColumnSelectionCandidate, PreparedSqlCandidate, SqlExecutionResult
from ..config import DUCKDB_PATH, PRIMARY_JOIN_KEY_ALIAS, SECONDARY_JOIN_KEY_ALIAS
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


def _mask_single_quoted_literals_and_comments(sql: str) -> str:
    masked: list[str] = []
    quote_char: str | None = None
    i = 0
    while i < len(sql):
        char = sql[i]
        if quote_char:
            masked.append(" ")
            if char == quote_char:
                if i + 1 < len(sql) and sql[i + 1] == quote_char:
                    masked.append(" ")
                    i += 2
                    continue
                quote_char = None
            i += 1
            continue
        if char == "-" and i + 1 < len(sql) and sql[i + 1] == "-":
            masked.extend("  ")
            i += 2
            while i < len(sql) and sql[i] not in "\r\n":
                masked.append(" ")
                i += 1
            continue
        if char == "/" and i + 1 < len(sql) and sql[i + 1] == "*":
            masked.extend("  ")
            i += 2
            while i < len(sql):
                if sql[i] == "*" and i + 1 < len(sql) and sql[i + 1] == "/":
                    masked.extend("  ")
                    i += 2
                    break
                masked.append(" ")
                i += 1
            continue
        if char == "'":
            quote_char = char
            masked.append(" ")
            i += 1
            continue
        masked.append(char)
        i += 1
    return "".join(masked)


def _contains_sql_identifier(sql: str, identifier: str) -> bool:
    value = str(identifier or "").strip()
    if not value:
        return False
    masked_sql = _mask_single_quoted_literals_and_comments(sql)
    quoted = _quote_duckdb_identifier(value)
    if quoted in masked_sql:
        return True
    if re.search(rf"(?<![A-Za-z0-9_]){re.escape(value)}(?![A-Za-z0-9_])", masked_sql):
        return True
    return False


def _validate_observation_sql(
    sql: str,
    *,
    approved_tables: list[str],
    approved_columns: list[dict[str, Any]],
) -> tuple[bool, str | None]:
    masked_sql = _mask_single_quoted_literals_and_comments(sql)
    if not re.search(r"\bFROM\b", masked_sql, re.IGNORECASE):
        return False, (
            "SQL must extract row-level observations and must read from at least one approved source table "
            "with a FROM clause; metadata-only SELECT literals are not allowed."
        )

    if approved_tables and not any(_contains_sql_identifier(sql, table) for table in approved_tables):
        return False, "SQL must read from at least one approved source table, not only return metadata labels."

    approved_column_names = [str(column.get("column") or "").strip() for column in approved_columns]
    if approved_column_names and not any(_contains_sql_identifier(sql, column) for column in approved_column_names):
        return False, "SQL must select or filter on at least one approved source column."

    return True, None


def _validate_runtime_schema_sql(sql: str) -> tuple[bool, str | None]:
    if not DUCKDB_PATH.exists():
        return True, None

    try:
        import duckdb
    except ModuleNotFoundError:
        return True, None

    sql_to_describe = str(sql or "").strip().rstrip(";")
    if not sql_to_describe:
        return False, "SQL is empty."

    with timing_stage("db_rag.sql.runtime_schema_preflight"):
        db = duckdb.connect(str(DUCKDB_PATH), read_only=True)
        try:
            db.execute(f"DESCRIBE {sql_to_describe}")
        except Exception as exc:
            return False, (
                "SQL failed DuckDB runtime schema validation before execution: "
                f"{type(exc).__name__}: {exc}. "
                "Use only tables and columns that exist in the DuckDB runtime schema."
            )
        finally:
            db.close()
    return True, None


_JOIN_KEY_ALIASES = {
    PRIMARY_JOIN_KEY_ALIAS.upper(),
    SECONDARY_JOIN_KEY_ALIAS.upper(),
    "SUBJID_PSEUDO",
    "FID_PSEUDO",
    "SUBJID",
    "FID",
}


def _is_approved_join_key(column_name: str) -> bool:
    normalized = str(column_name or "").upper()
    if not normalized:
        return False
    if normalized in _JOIN_KEY_ALIASES:
        return True
    if "SUBJID" in normalized:
        return True
    return bool(re.search(r"(^|_)FID(_|$)", normalized))


def _validate_multi_table_join_preflight(columns: list[dict[str, Any]]) -> tuple[bool, str | None]:
    tables = {str(column.get("table") or "").strip() for column in columns}
    tables.discard("")
    if len(tables) <= 1:
        return True, None

    tables_with_join_keys = {
        str(column.get("table") or "").strip()
        for column in columns
        if _is_approved_join_key(str(column.get("column") or ""))
    }
    missing_tables = sorted(tables - tables_with_join_keys)
    if not missing_tables:
        return True, None

    return False, (
        "Multi-table extraction requires an approved join key column for every selected source table "
        "before SQL generation. Missing approved subject/family ID join keys for: "
        f"{', '.join(missing_tables)}. Include approved SUBJID or FID columns in the column selection, "
        "or use a single-table extraction."
    )


def _duckdb_column_profile(table: str, column: str) -> dict[str, object]:
    import duckdb

    if not DUCKDB_PATH.exists():
        return {}

    quoted_table = _quote_duckdb_identifier(table)
    quoted_column = _quote_duckdb_identifier(column)
    with timing_stage("db_rag.sql.column_profile", table=table, column=column):
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


def _approved_columns_for_sql_prompt(
    columns: list[dict[str, Any]],
    *,
    profile_cache: dict[tuple[str, str], dict[str, object]] | None = None,
) -> list[dict[str, Any]]:
    cache = profile_cache if profile_cache is not None else {}
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
            cache_key = (table, column_name)
            if cache_key not in cache:
                cache[cache_key] = _duckdb_column_profile(table, column_name)
            entry.update(cache[cache_key])
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
        with timing_stage("db_rag.sql.prepare_sql_candidate", columns=len(candidate_columns)):
            with timing_stage(
                "db_rag.sql.preflight",
                tables=len({column["table"] for column in candidate_columns}),
                columns=len(candidate_columns),
            ):
                preflight_valid, preflight_error = _validate_multi_table_join_preflight(candidate_columns)
            if not preflight_valid:
                raise DbRagUnanswerableError(preflight_error or "SQL preflight failed.")

            profile_cache: dict[tuple[str, str], dict[str, object]] = {}
            prompt_columns = _approved_columns_for_sql_prompt(candidate_columns, profile_cache=profile_cache)
            with timing_stage("db_rag.sql.generate_sql"):
                response = self.llm.invoke(
                    [
                        SystemMessage(
                            content=(
                                "You are a DuckDB SQL expert for the RePORT clinical research database. "
                                "Use only the approved tables and approved columns listed in the prompt. "
                                "Do not reference any other tables or columns. "
                                "Generate SQL that extracts row-level observations from the source database tables. "
                                "Do not return a metadata catalog of variable labels, table names, column names, "
                                "descriptions, dtypes, or sample values. SELECTing literal strings as rows is invalid. "
                                "When the request asks for variables or variable information, interpret that as selecting "
                                "the actual observation columns for a subset dataset. "
                                "When multiple approved source tables are needed, join/integrate rows using subject ID "
                                "or family ID columns when those exact join-key columns are approved; otherwise return "
                                "UNANSWERABLE rather than inventing unapproved join columns. "
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
        if valid:
            valid, error = _validate_observation_sql(
                sql,
                approved_tables=approved_tables,
                approved_columns=candidate_columns,
            )
        if valid:
            valid, error = _validate_runtime_schema_sql(sql)
        if not valid:
            repair_seed = PreparedSqlCandidate(
                question=question,
                sql=sql,
                tables=approved_tables,
                columns=candidate_columns,
                selection_id=approved_selection.selection_id,
            )
            return self.repair_prepared_sql_candidate(
                repair_seed,
                error or "SQL validation failed.",
                profiled_columns=prompt_columns,
                profile_cache=profile_cache,
            )
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
        valid, error = _validate_runtime_schema_sql(candidate.sql)
        if not valid:
            raise ValueError(error or "SQL runtime schema validation failed.")

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

    def repair_prepared_sql_candidate(
        self,
        candidate: PreparedSqlCandidate,
        error_message: str,
        *,
        profiled_columns: list[dict[str, Any]] | None = None,
        profile_cache: dict[tuple[str, str], dict[str, object]] | None = None,
    ) -> PreparedSqlCandidate:
        from langchain_core.messages import HumanMessage, SystemMessage

        with timing_stage("db_rag.sql.repair_sql", columns=len(candidate.columns)):
            prompt_columns = (
                list(profiled_columns)
                if profiled_columns is not None
                else _approved_columns_for_sql_prompt(
                    [dict(column) for column in candidate.columns],
                    profile_cache=profile_cache,
                )
            )
            with timing_stage("db_rag.sql.repair_generate_sql"):
                response = self.llm.invoke(
                    [
                        SystemMessage(
                            content=(
                                "You are fixing a DuckDB SQL query for the RePORT clinical research database. "
                                "Use only the approved tables and approved columns. "
                                "The corrected SQL must extract row-level observations from approved source tables. "
                                "Do not return metadata rows containing variable labels, table names, column names, "
                                "descriptions, dtypes, or sample values. SELECTing literal strings as rows is invalid. "
                                "When multiple approved source tables are needed, join/integrate rows using subject ID "
                                "or family ID columns when those exact join-key columns are approved; otherwise return "
                                "UNANSWERABLE rather than inventing unapproved join columns. "
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
            if valid:
                valid, error = _validate_observation_sql(
                    repaired_sql,
                    approved_tables=list(candidate.tables),
                    approved_columns=[dict(column) for column in candidate.columns],
                )
            if valid:
                valid, error = _validate_runtime_schema_sql(repaired_sql)
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
