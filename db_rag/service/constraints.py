from __future__ import annotations

from typing import Any

from db_rag.generation import DB_RAG_CONTEXT_FALLBACK_RATIONALE, default_selection_id

from .models import ColumnSelectionCandidate, DbRagContext, DbRagIntent, FeedbackConstraintSet
from .schema import _lookup_schema_column


def _constraint_set_from_payload(payload: DbRagIntent | dict[str, Any] | None) -> FeedbackConstraintSet:
    if isinstance(payload, DbRagIntent):
        source = payload.__dict__
    elif isinstance(payload, dict):
        source = payload
    else:
        source = {}
    return FeedbackConstraintSet(
        goal_text=str(source.get("goal_text") or "").strip(),
        required_tables=tuple(str(value or "").strip() for value in list(source.get("required_tables") or []) if str(value or "").strip()),
        required_columns=tuple(str(value or "").strip() for value in list(source.get("required_columns") or []) if str(value or "").strip()),
        excluded_tables=tuple(str(value or "").strip() for value in list(source.get("excluded_tables") or []) if str(value or "").strip()),
        excluded_columns=tuple(str(value or "").strip() for value in list(source.get("excluded_columns") or []) if str(value or "").strip()),
    )


def _filter_and_inject_context_columns(
    column_rows: list[dict[str, str]],
    constraints: FeedbackConstraintSet,
) -> list[dict[str, str]]:
    filtered: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for entry in list(column_rows or []):
        table = str(entry.get("table") or "").strip()
        column = str(entry.get("column") or "").strip()
        if not table or not column:
            continue
        if table in constraints.excluded_tables:
            continue
        qualified = f"{table}.{column}"
        if qualified in constraints.excluded_columns:
            continue
        pair = (table, column)
        if pair in seen_pairs:
            continue
        filtered.append(entry)
        seen_pairs.add(pair)

    for qualified in constraints.required_columns:
        if "." not in qualified:
            continue
        table, column = qualified.split(".", 1)
        pair = (table, column)
        if pair in seen_pairs:
            continue
        schema_entry = _lookup_schema_column(table, column)
        if schema_entry is None:
            continue
        filtered.append({"table": table, "column": column, "text": schema_entry.get("description", "")})
        seen_pairs.add(pair)

    return filtered


def _enforce_selection_constraints(
    candidate: ColumnSelectionCandidate,
    context: DbRagContext,
    constraints: FeedbackConstraintSet,
) -> ColumnSelectionCandidate:
    valid_tables = set(context.table_names)
    valid_columns = {(entry.table, entry.column) for entry in context.columns}

    columns: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for entry in list(candidate.columns or []):
        table = str(entry.get("table") or "").strip()
        column = str(entry.get("column") or "").strip()
        if not table or not column:
            continue
        if table in constraints.excluded_tables:
            continue
        qualified = f"{table}.{column}"
        if qualified in constraints.excluded_columns:
            continue
        schema_entry = _lookup_schema_column(table, column)
        if (table, column) not in valid_columns and schema_entry is None:
            continue
        description = str(entry.get("description") or "").strip()
        if not description and schema_entry is not None:
            description = schema_entry["description"]
        pair = (table, column)
        if pair in seen_pairs:
            continue
        columns.append({"table": table, "column": column, "description": description})
        seen_pairs.add(pair)

    for qualified in constraints.required_columns:
        if "." not in qualified:
            continue
        table, column = qualified.split(".", 1)
        pair = (table, column)
        if pair in seen_pairs:
            continue
        schema_entry = _lookup_schema_column(table, column)
        if (table, column) not in valid_columns and schema_entry is None:
            continue
        description = schema_entry["description"] if schema_entry is not None else ""
        columns.append({"table": table, "column": column, "description": description})
        seen_pairs.add(pair)

    tables: list[str] = []
    for table in list(candidate.tables or []):
        table_name = str(table or "").strip()
        if not table_name or table_name not in valid_tables:
            continue
        if table_name in constraints.excluded_tables:
            continue
        if table_name not in tables:
            tables.append(table_name)

    for table in constraints.required_tables:
        if table in constraints.excluded_tables:
            continue
        if table in valid_tables and table not in tables:
            tables.append(table)

    for column in columns:
        table = column["table"]
        if table not in tables and table not in constraints.excluded_tables:
            tables.append(table)

    return ColumnSelectionCandidate(
        selection_id=candidate.selection_id,
        question=candidate.question,
        tables=tables,
        columns=columns,
        rationale=candidate.rationale,
        feedback_history=list(candidate.feedback_history),
        status=candidate.status,
        selection_source=getattr(candidate, "selection_source", "legacy"),
        fallback_reason=str(getattr(candidate, "fallback_reason", "") or ""),
        raw_model_output=str(getattr(candidate, "raw_model_output", "") or ""),
    )


def _merge_constraint_sets(*constraints_sets: FeedbackConstraintSet) -> FeedbackConstraintSet:
    goal_text = ""
    required_tables: list[str] = []
    required_columns: list[str] = []
    excluded_tables: list[str] = []
    excluded_columns: list[str] = []

    for constraints in constraints_sets:
        if constraints.goal_text:
            goal_text = constraints.goal_text
        for table in constraints.required_tables:
            if table not in required_tables:
                required_tables.append(table)
        for column in constraints.required_columns:
            if column not in required_columns:
                required_columns.append(column)
        for table in constraints.excluded_tables:
            if table not in excluded_tables:
                excluded_tables.append(table)
        for column in constraints.excluded_columns:
            if column not in excluded_columns:
                excluded_columns.append(column)

    required_tables = [table for table in required_tables if table not in excluded_tables]
    required_columns = [column for column in required_columns if column not in excluded_columns]
    return FeedbackConstraintSet(
        goal_text=goal_text,
        required_tables=tuple(required_tables),
        required_columns=tuple(required_columns),
        excluded_tables=tuple(excluded_tables),
        excluded_columns=tuple(excluded_columns),
    )


def _normalize_previous_selection(previous_selection: Any) -> dict[str, Any]:
    if isinstance(previous_selection, ColumnSelectionCandidate):
        return {
            "selection_id": previous_selection.selection_id,
            "question": previous_selection.question,
            "tables": list(previous_selection.tables),
            "columns": [dict(column) for column in previous_selection.columns],
            "rationale": previous_selection.rationale,
            "feedback_history": list(previous_selection.feedback_history),
            "status": previous_selection.status,
        }
    if isinstance(previous_selection, dict):
        return {
            "selection_id": str(previous_selection.get("selection_id", "") or "").strip(),
            "question": str(previous_selection.get("question", "") or "").strip(),
            "tables": [str(value or "").strip() for value in list(previous_selection.get("tables") or []) if str(value or "").strip()],
            "columns": [
                {
                    "table": str(column.get("table", "") or "").strip(),
                    "column": str(column.get("column", "") or "").strip(),
                    "description": str(column.get("description", "") or "").strip(),
                }
                for column in list(previous_selection.get("columns") or [])
                if isinstance(column, dict)
                and str(column.get("table", "") or "").strip()
                and str(column.get("column", "") or "").strip()
            ],
            "rationale": str(previous_selection.get("rationale", "") or "").strip(),
            "feedback_history": list(previous_selection.get("feedback_history") or []),
            "status": str(previous_selection.get("status", "") or "").strip(),
        }
    return {
        "selection_id": "",
        "question": "",
        "tables": [],
        "columns": [],
        "rationale": "",
        "feedback_history": [],
        "status": "",
    }


def _build_invalid_column_selection_candidate(
    question: str,
    context: DbRagContext,
    feedback_history: list[dict[str, Any]],
) -> ColumnSelectionCandidate:
    columns = [{"table": entry.table, "column": entry.column} for entry in context.columns]
    return ColumnSelectionCandidate(
        selection_id=default_selection_id(question, context.table_names, columns, feedback_history),
        question=question,
        tables=[],
        columns=[],
        rationale=DB_RAG_CONTEXT_FALLBACK_RATIONALE,
        feedback_history=feedback_history,
    )


def _default_selection_id(question: str, context: DbRagContext, feedback_history: list[dict[str, Any]]) -> str:
    columns = [{"table": entry.table, "column": entry.column} for entry in context.columns]
    return default_selection_id(question, context.table_names, columns, feedback_history)
