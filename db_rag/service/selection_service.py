from __future__ import annotations

import json
from typing import Any

from utils.llm_response import coerce_text_content

from .constraints import (
    _build_invalid_column_selection_candidate,
    _constraint_set_from_payload,
    _enforce_selection_constraints,
    _merge_constraint_sets,
    _normalize_previous_selection,
)
from .models import ColumnSelectionCandidate, DbRagContext, DbRagIntent, FeedbackConstraintSet
from .schema import _lookup_schema_column, _resolve_explicit_schema_mentions
from ..generation import default_selection_id, parse_json_object


class DbRagSelectionMixin:
    llm: Any

    def ground_feedback_constraints(
        self,
        question: str,
        context: DbRagContext,
        *,
        feedback_history: list[dict[str, Any]] | None = None,
        previous_selection: Any = None,
        intent_snapshot: dict[str, Any] | None = None,
    ) -> FeedbackConstraintSet:
        from langchain_core.messages import HumanMessage, SystemMessage

        normalized_feedback_history = list(feedback_history or [])
        normalized_previous_selection = _normalize_previous_selection(previous_selection)
        response = self.llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are grounding human review feedback for a RePORT DB-RAG column selection task. "
                        "Map concept-level feedback onto exact schema table names and exact table.column strings "
                        "using only the provided retrieved context and previous selection. "
                        "Return JSON only with keys: "
                        '{"goal_text": string, "required_tables": [string], "required_columns": [string], '
                        '"excluded_tables": [string], "excluded_columns": [string]}.'
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question:\n{question}\n\n"
                        f"Table context:\n{context.table_context or 'none'}\n\n"
                        f"Column context:\n{context.column_context or 'none'}\n\n"
                        f"Intent snapshot:\n{json.dumps(intent_snapshot or {}, indent=2, sort_keys=True)}\n\n"
                        f"Feedback history:\n{json.dumps(normalized_feedback_history, indent=2, sort_keys=True)}\n\n"
                        f"Previous selection candidate:\n{json.dumps(normalized_previous_selection, indent=2, sort_keys=True)}"
                    )
                ),
            ]
        )
        parsed = parse_json_object(coerce_text_content(getattr(response, "content", ""))) or {}
        return _constraint_set_from_payload(parsed)

    def validate_selection_against_intent(
        self,
        candidate: ColumnSelectionCandidate,
        intent: DbRagIntent,
    ) -> tuple[bool, str]:
        selected_tables = set(candidate.tables)
        selected_columns = {f'{entry["table"]}.{entry["column"]}' for entry in candidate.columns}
        for table in intent.required_tables:
            if table not in selected_tables:
                return False, f"Missing required table: {table}"
        for column in intent.required_columns:
            if column not in selected_columns:
                return False, f"Missing required column: {column}"
        for table in intent.excluded_tables:
            if table in selected_tables:
                return False, f"Selection includes excluded table: {table}"
        for column in intent.excluded_columns:
            if column in selected_columns:
                return False, f"Selection includes excluded column: {column}"
        return True, ""

    def prepare_column_selection(
        self,
        question: str,
        context: DbRagContext,
        feedback_history: list[dict[str, Any]] | None = None,
        previous_selection: Any = None,
        intent_snapshot: dict[str, Any] | None = None,
    ) -> ColumnSelectionCandidate:
        from langchain_core.messages import HumanMessage, SystemMessage

        normalized_feedback_history = list(feedback_history or [])
        normalized_previous_selection = _normalize_previous_selection(previous_selection)
        base_constraints = _constraint_set_from_payload(intent_snapshot)
        grounded_constraints = self.ground_feedback_constraints(
            question,
            context,
            feedback_history=normalized_feedback_history,
            previous_selection=previous_selection,
            intent_snapshot=intent_snapshot,
        )
        constraints = _merge_constraint_sets(base_constraints, grounded_constraints)
        valid_tables = set(context.table_names)
        valid_columns = {(entry.table, entry.column) for entry in context.columns}
        explicit_schema_columns = _resolve_explicit_schema_mentions(question, normalized_feedback_history)
        explicit_valid_columns = [
            column
            for column in explicit_schema_columns
            if _lookup_schema_column(column["table"], column["column"]) is not None
        ]
        constrained_columns = [
            column
            for column in normalized_previous_selection.get("columns", [])
            if _lookup_schema_column(column["table"], column["column"]) is not None
        ]
        response = self.llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are preparing a column selection candidate for a RePORT database question. "
                        "Use the retrieved table and column context plus any reviewer feedback to select only "
                        "exact tables and exact table.column pairs relevant to the question. "
                        "If the prompt includes a previous reviewed selection or explicit schema constraints, "
                        "preserve those exact table.column pairs unless the reviewer feedback explicitly replaces them. "
                        "Do not drop required exact columns just because they were not top-ranked in retrieval. "
                        "Return only a JSON object with exactly these keys: "
                        '{"selection_id": string, "rationale": string, "tables": [string], '
                        '"columns": [{"table": string, "column": string, "description": string}]}.'
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question:\n{question}\n\n"
                        f"Table context:\n{context.table_context or 'none'}\n\n"
                        f"Column context:\n{context.column_context or 'none'}\n\n"
                        f"Intent snapshot:\n{json.dumps(intent_snapshot or {}, indent=2, sort_keys=True)}\n\n"
                        f"Grounded constraints:\n{json.dumps(constraints.__dict__, indent=2, sort_keys=True)}\n\n"
                        f"Feedback history:\n{json.dumps(normalized_feedback_history, indent=2, sort_keys=True)}\n\n"
                        f"Previous selection candidate:\n{json.dumps(normalized_previous_selection, indent=2, sort_keys=True)}\n\n"
                        f"Explicit schema constraints:\n{json.dumps(explicit_valid_columns, indent=2, sort_keys=True)}"
                    )
                ),
            ]
        )
        parsed = parse_json_object(coerce_text_content(getattr(response, "content", "")))
        if not parsed:
            return _build_invalid_column_selection_candidate(question, context, normalized_feedback_history)

        raw_tables = parsed.get("tables")
        tables: list[str] = []
        if isinstance(raw_tables, list):
            for value in raw_tables:
                table = str(value or "").strip()
                if table and table in valid_tables and table not in tables and table not in constraints.excluded_tables:
                    tables.append(table)

        raw_columns = parsed.get("columns")
        columns: list[dict[str, str]] = []
        if isinstance(raw_columns, list):
            for item in raw_columns:
                if not isinstance(item, dict):
                    continue
                table = str(item.get("table", "") or "").strip()
                column = str(item.get("column", "") or "").strip()
                if not table or not column:
                    continue
                if table in constraints.excluded_tables:
                    continue
                if f"{table}.{column}" in constraints.excluded_columns:
                    continue
                schema_entry = _lookup_schema_column(table, column)
                if (table, column) not in valid_columns and schema_entry is None:
                    continue
                description = str(item.get("description", "") or "").strip()
                if not description and schema_entry is not None:
                    description = schema_entry["description"]
                columns.append({"table": table, "column": column, "description": description})

        constrained_pairs = {
            (column["table"], column["column"]): column
            for column in [*explicit_valid_columns, *constrained_columns]
            if column["table"] not in constraints.excluded_tables
            and f'{column["table"]}.{column["column"]}' not in constraints.excluded_columns
        }
        current_pairs = {(column["table"], column["column"]) for column in columns}
        for pair, column in constrained_pairs.items():
            if pair in current_pairs:
                continue
            columns.append(dict(column))
            current_pairs.add(pair)

        columns = [
            column
            for column in columns
            if column["table"] not in constraints.excluded_tables
            and f'{column["table"]}.{column["column"]}' not in constraints.excluded_columns
        ]

        if not tables and columns:
            for column in columns:
                table = column["table"]
                if table not in tables:
                    tables.append(table)

        for column in columns:
            table = column["table"]
            if table not in tables:
                tables.append(table)

        selection_id = str(parsed.get("selection_id", "") or "").strip() or default_selection_id(
            question,
            context.table_names,
            [{"table": entry.table, "column": entry.column} for entry in context.columns],
            normalized_feedback_history,
        )
        rationale = str(parsed.get("rationale", "") or "").strip()

        if not columns:
            return _build_invalid_column_selection_candidate(question, context, normalized_feedback_history)

        candidate = ColumnSelectionCandidate(
            selection_id=selection_id,
            question=question,
            tables=tables,
            columns=columns,
            rationale=rationale,
            feedback_history=normalized_feedback_history,
        )
        return _enforce_selection_constraints(candidate, context, constraints)
