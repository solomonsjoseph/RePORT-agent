from __future__ import annotations

import json
import os
from typing import Any

from utils.llm_response import coerce_text_content

from .constraints import (
    _constraint_set_from_payload,
    _enforce_selection_constraints,
    _merge_constraint_sets,
    _normalize_previous_selection,
)
from .models import ColumnSelectionCandidate, DbRagContext, DbRagIntent, FeedbackConstraintSet
from ..config import resolve_db_rag_selection_model
from .schema import _lookup_schema_column, _resolve_explicit_schema_mentions
from ..generation import default_selection_id, parse_json_object

_DETERMINISTIC_CONSTRAINED_RATIONALE = "Using constrained retrieved candidates directly for human review."
_OPENAI_RANKED_DEFAULT_RATIONALE = "Ranked from constrained retrieved candidates."
_RETRIEVAL_FALLBACK_RATIONALE = (
    "Structured ranking output was unavailable. Showing retrieved candidate tables and columns directly for human review."
)


class _OpenAIStructuredColumnSelector:
    def __init__(self, *, resolve_model=resolve_db_rag_selection_model) -> None:
        self._resolve_model = resolve_model

    @staticmethod
    def _resolve_client():
        try:
            from openai import OpenAI
        except ModuleNotFoundError:
            return None
        return OpenAI

    def rank_columns(
        self,
        *,
        question: str,
        candidate_columns: list[dict[str, str]],
        feedback_history: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        model = self._resolve_model()
        if not model:
            return None

        api_key = str(os.getenv("DB_RAG_SELECTION_API_KEY", "") or "").strip()
        if not api_key:
            api_key = str(os.getenv("OPENAI_API_KEY", "") or "").strip()
        if not api_key:
            return None

        client_cls = self._resolve_client()
        if client_cls is None:
            return None

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        base_url = str(os.getenv("DB_RAG_SELECTION_BASE_URL", "") or "").strip()
        if base_url:
            client_kwargs["base_url"] = base_url

        candidate_lines = []
        for entry in candidate_columns:
            identifier = f'{entry["table"]}||{entry["column"]}'
            description = str(entry.get("description") or "").strip()
            if description:
                candidate_lines.append(f"- {identifier}: {description}")
            else:
                candidate_lines.append(f"- {identifier}")

        try:
            client = client_cls(**client_kwargs)
            response = client.chat.completions.create(
                model=model,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "db_rag_column_ranking",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "columns": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "rationale": {"type": "string"},
                            },
                            "required": ["columns", "rationale"],
                            "additionalProperties": False,
                        },
                    },
                },
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Rank the provided RePORT database columns from most to least relevant for the question. "
                            "Use only the exact identifiers from the candidate list. Return every ranked identifier once. "
                            "Do not invent identifiers."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Question:\n{question}\n\n"
                            f"Candidate columns:\n{chr(10).join(candidate_lines) or 'none'}\n\n"
                            f"Feedback history:\n{json.dumps(list(feedback_history or []), indent=2, sort_keys=True)}"
                        ),
                    },
                ],
            )
            content = coerce_text_content(getattr(response.choices[0].message, "content", ""))
            parsed = parse_json_object(content) or {}
        except Exception:
            return None

        columns = parsed.get("columns")
        if not isinstance(columns, list):
            return {
                "columns": [],
                "rationale": "",
                "raw_model_output": content,
            }
        return {
            "columns": [str(value or "").strip() for value in columns if str(value or "").strip()],
            "rationale": str(parsed.get("rationale") or "").strip(),
            "raw_model_output": content,
        }


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

    def _rank_columns_with_openai(
        self,
        *,
        question: str,
        candidate_columns: list[dict[str, str]],
        feedback_history: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        selector = _OpenAIStructuredColumnSelector()
        return selector.rank_columns(
            question=question,
            candidate_columns=candidate_columns,
            feedback_history=feedback_history,
        )

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
        constrained_previous_columns = [
            column
            for column in normalized_previous_selection.get("columns", [])
            if _lookup_schema_column(column["table"], column["column"]) is not None
        ]
        constrained_pool: list[dict[str, str]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for entry in list(context.columns or []):
            table = str(entry.table or "").strip()
            column = str(entry.column or "").strip()
            if not table or not column:
                continue
            if table in constraints.excluded_tables:
                continue
            qualified = f"{table}.{column}"
            if qualified in constraints.excluded_columns:
                continue
            schema_entry = _lookup_schema_column(table, column)
            description = str((schema_entry or {}).get("description") or str(entry.text or "") or "").strip()
            pair = (table, column)
            if pair in seen_pairs:
                continue
            constrained_pool.append({"table": table, "column": column, "description": description})
            seen_pairs.add(pair)

        forced_columns = [
            column
            for column in [*explicit_valid_columns, *constrained_previous_columns]
            if column["table"] not in constraints.excluded_tables
            and f'{column["table"]}.{column["column"]}' not in constraints.excluded_columns
        ]
        for column in forced_columns:
            pair = (column["table"], column["column"])
            if pair in seen_pairs:
                continue
            constrained_pool.append(
                {
                    "table": column["table"],
                    "column": column["column"],
                    "description": str(column.get("description") or "").strip(),
                }
            )
            seen_pairs.add(pair)

        selection_id = default_selection_id(
            question,
            context.table_names,
            [{"table": entry.table, "column": entry.column} for entry in context.columns],
            normalized_feedback_history,
        )
        retrieval_tables = [
            table_name
            for table_name in context.table_names
            if table_name in valid_tables and table_name not in constraints.excluded_tables
        ]

        if not constrained_pool:
            candidate = ColumnSelectionCandidate(
                selection_id=selection_id,
                question=question,
                tables=[],
                columns=[],
                rationale=_RETRIEVAL_FALLBACK_RATIONALE,
                feedback_history=normalized_feedback_history,
                selection_source="retrieval_fallback",
                fallback_reason=_RETRIEVAL_FALLBACK_RATIONALE,
            )
            return _enforce_selection_constraints(candidate, context, constraints)

        if len(constrained_pool) <= 2:
            candidate = ColumnSelectionCandidate(
                selection_id=selection_id,
                question=question,
                tables=retrieval_tables,
                columns=list(constrained_pool),
                rationale=_DETERMINISTIC_CONSTRAINED_RATIONALE,
                feedback_history=normalized_feedback_history,
                selection_source="deterministic_constrained",
            )
            return _enforce_selection_constraints(candidate, context, constraints)

        ranked = self._rank_columns_with_openai(
            question=question,
            candidate_columns=constrained_pool,
            feedback_history=normalized_feedback_history,
        )

        ranked_ids = list((ranked or {}).get("columns") or [])
        raw_model_output = str((ranked or {}).get("raw_model_output") or "").strip()
        ranked_lookup = {
            f'{entry["table"]}||{entry["column"]}': dict(entry)
            for entry in constrained_pool
        }
        validated_columns: list[dict[str, str]] = []
        seen_ranked: set[tuple[str, str]] = set()
        dropped_any = False
        for identifier in ranked_ids:
            if identifier not in ranked_lookup:
                dropped_any = True
                continue
            entry = ranked_lookup[identifier]
            pair = (entry["table"], entry["column"])
            if pair in seen_ranked:
                dropped_any = True
                continue
            validated_columns.append(entry)
            seen_ranked.add(pair)

        if not validated_columns:
            candidate = ColumnSelectionCandidate(
                selection_id=selection_id,
                question=question,
                tables=retrieval_tables,
                columns=list(constrained_pool),
                rationale=_RETRIEVAL_FALLBACK_RATIONALE,
                feedback_history=normalized_feedback_history,
                selection_source="retrieval_fallback",
                fallback_reason=_RETRIEVAL_FALLBACK_RATIONALE,
                raw_model_output=raw_model_output,
            )
            return _enforce_selection_constraints(candidate, context, constraints)

        tables: list[str] = []
        for entry in validated_columns:
            table = entry["table"]
            if table not in tables:
                tables.append(table)
        for table in retrieval_tables:
            if table not in tables:
                tables.append(table)

        candidate = ColumnSelectionCandidate(
            selection_id=selection_id,
            question=question,
            tables=tables,
            columns=validated_columns,
            rationale=str((ranked or {}).get("rationale") or "").strip() or _OPENAI_RANKED_DEFAULT_RATIONALE,
            feedback_history=normalized_feedback_history,
            selection_source="openai_ranked_partial" if dropped_any else "openai_ranked",
            raw_model_output=raw_model_output,
        )
        return _enforce_selection_constraints(candidate, context, constraints)
