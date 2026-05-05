from __future__ import annotations

import json
from typing import Any

from utils.llm_response import coerce_text_content

from . import classifier
from .models import DbRagContext, DbRagIntent, DbRagQaAnswer
from .schema import _lookup_schema_column, _schema_table_names
from ..config import resolve_db_rag_reply_classifier_model
from ..generation import (
    DB_RAG_CONTEXT_FALLBACK_ANSWER,
    DB_RAG_CONTEXT_FALLBACK_RATIONALE,
    default_selection_id,
    parse_json_object,
)


class DbRagIntentMixin:
    llm: Any

    def classify_pending_reply(
        self,
        *,
        pending_kind: str,
        pending_question: str,
        user_reply: str,
        recent_transcript: str,
    ) -> dict[str, Any]:
        del pending_kind
        return classifier.classify_pending_reply(
            pending_question=pending_question,
            user_reply=user_reply,
            recent_transcript=recent_transcript,
            resolve_model=resolve_db_rag_reply_classifier_model,
        )

    def classify_extraction_gate_message(
        self,
        *,
        pending_prompt: str,
        user_message: str,
        recent_transcript: str,
        active_intent: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return classifier.classify_extraction_gate_message(
            pending_prompt=pending_prompt,
            user_message=user_message,
            recent_transcript=recent_transcript,
            active_intent=active_intent,
            resolve_model=resolve_db_rag_reply_classifier_model,
        )

    def classify_sql_review_feedback(
        self,
        *,
        feedback_text: str,
        sql: str,
    ) -> dict[str, Any]:
        return classifier.classify_sql_review_feedback(
            feedback_text=feedback_text,
            sql=sql,
            resolve_model=resolve_db_rag_reply_classifier_model,
        )

    def answer_question(self, question: str) -> dict[str, Any]:
        from langchain_core.messages import HumanMessage, SystemMessage

        context = self.retrieve_context(question)
        response = self.llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are answering a question about the RePORT clinical database using only retrieved "
                        "database metadata and schema context. Answer directly from the retrieved context. "
                        "If the user's question would require row-level computation or cohort extraction, say "
                        "which tables/columns appear relevant and state that a read-only SQL extraction would be "
                        "needed for an exact numeric answer."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question:\n{question}\n\n"
                        f"Table context:\n{context.table_context or 'none'}\n\n"
                        f"Column context:\n{context.column_context or 'none'}"
                    )
                ),
            ]
        )
        return {
            "answer": coerce_text_content(getattr(response, "content", "")),
            "retrieval_summary": {
                "tables": context.table_names,
                "columns": context.column_names,
            },
        }

    def answer_from_context(self, question: str, context: DbRagContext) -> DbRagQaAnswer:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = self.llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are classifying whether a RePORT database question can be answered from retrieved "
                        "metadata and schema context alone.\n"
                        "Use the retrieved table and column descriptions to answer overview, schema, variable, and "
                        "other metadata questions directly.\n"
                        "Mark needs_sql true when the user asks for row-level records, subsets, counts, filters, "
                        "aggregations, or any other request that depends on actual data values rather than schema.\n"
                        "Return only a JSON object with exactly these keys: "
                        '{"answer": string, "needs_sql": boolean, "rationale": string}.\n'
                        "If the request can be answered from metadata/schema context, set needs_sql to false.\n"
                        "If the request needs SQL for an exact answer, set needs_sql to true and explain why in rationale."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question:\n{question}\n\n"
                        f"Table context:\n{context.table_context or 'none'}\n\n"
                        f"Column context:\n{context.column_context or 'none'}"
                    )
                ),
            ]
        )
        parsed = parse_json_object(coerce_text_content(getattr(response, "content", "")))
        needs_sql_value = parsed.get("needs_sql")
        if not parsed or not isinstance(needs_sql_value, bool):
            return DbRagQaAnswer(
                answer=DB_RAG_CONTEXT_FALLBACK_ANSWER,
                needs_sql=True,
                rationale=DB_RAG_CONTEXT_FALLBACK_RATIONALE,
                relevant_tables=context.table_names,
                relevant_columns=context.column_names,
            )

        answer_text = str(parsed.get("answer", "") or "").strip() or DB_RAG_CONTEXT_FALLBACK_ANSWER
        rationale = str(parsed.get("rationale", "") or "").strip() or DB_RAG_CONTEXT_FALLBACK_RATIONALE
        return DbRagQaAnswer(
            answer=answer_text,
            needs_sql=needs_sql_value,
            rationale=rationale,
            relevant_tables=context.table_names,
            relevant_columns=context.column_names,
        )

    def resolve_intent(
        self,
        question: str,
        context: DbRagContext,
        prior_intent: dict[str, Any] | None = None,
    ) -> DbRagIntent:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = self.llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are normalizing a RePORT DB-RAG request into a structured intent. "
                        "Return only JSON with keys intent_id, goal_text, mode, population, requested_fields, filters."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question:\n{question}\n\n"
                        f"Table context:\n{context.table_context or 'none'}\n\n"
                        f"Column context:\n{context.column_context or 'none'}\n\n"
                        f"Prior intent:\n{json.dumps(prior_intent or {}, indent=2, sort_keys=True)}"
                    )
                ),
            ]
        )
        parsed = parse_json_object(coerce_text_content(getattr(response, "content", ""))) or {}
        return DbRagIntent(
            intent_id=str(parsed.get("intent_id") or default_selection_id(question, context.table_names, [], [])),
            source_question=question,
            goal_text=str(parsed.get("goal_text") or question).strip(),
            mode=str(parsed.get("mode") or "metadata").strip(),
            population=str(parsed.get("population") or "").strip() or None,
            requested_fields=[str(value).strip() for value in list(parsed.get("requested_fields") or []) if str(value).strip()],
            filters=[str(value).strip() for value in list(parsed.get("filters") or []) if str(value).strip()],
        )

    def update_intent_from_feedback(
        self,
        intent: DbRagIntent | dict[str, Any],
        feedback_history: list[dict[str, str]],
    ) -> DbRagIntent:
        from langchain_core.messages import HumanMessage, SystemMessage

        base = intent if isinstance(intent, DbRagIntent) else DbRagIntent(**intent)
        response = self.llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are updating a structured DB-RAG intent from human review feedback. "
                        "Return JSON only with keys: "
                        '{"goal_text": string, "required_tables": [string], "required_columns": [string], '
                        '"excluded_tables": [string], "excluded_columns": [string]}. '
                        "Use exact schema table names and exact 'table.column' strings when possible."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Current intent:\n{json.dumps(base.__dict__, indent=2, sort_keys=True)}\n\n"
                        f"Feedback history:\n{json.dumps(list(feedback_history or []), indent=2, sort_keys=True)}"
                    )
                ),
            ]
        )
        parsed = parse_json_object(coerce_text_content(getattr(response, "content", ""))) or {}

        schema_tables = _schema_table_names()
        required_tables: list[str] = []
        required_columns: list[str] = []
        excluded_tables: list[str] = []
        excluded_columns: list[str] = []

        for table in [*list(base.required_tables), *list(parsed.get("required_tables") or [])]:
            table_name = str(table or "").strip()
            if table_name and table_name in schema_tables and table_name not in required_tables:
                required_tables.append(table_name)

        for item in [*list(base.required_columns), *list(parsed.get("required_columns") or [])]:
            value = str(item or "").strip()
            if "." not in value:
                continue
            table_name, column_name = value.split(".", 1)
            if _lookup_schema_column(table_name, column_name) is None:
                continue
            normalized = f"{table_name}.{column_name}"
            if normalized not in required_columns:
                required_columns.append(normalized)

        for table in [*list(base.excluded_tables), *list(parsed.get("excluded_tables") or [])]:
            table_name = str(table or "").strip()
            if table_name and table_name in schema_tables and table_name not in excluded_tables:
                excluded_tables.append(table_name)

        for item in [*list(base.excluded_columns), *list(parsed.get("excluded_columns") or [])]:
            value = str(item or "").strip()
            if "." not in value:
                continue
            table_name, column_name = value.split(".", 1)
            if _lookup_schema_column(table_name, column_name) is None:
                continue
            normalized = f"{table_name}.{column_name}"
            if normalized not in excluded_columns:
                excluded_columns.append(normalized)

        required_tables = [table for table in required_tables if table not in excluded_tables]
        required_columns = [column for column in required_columns if column not in excluded_columns]

        goal_text = str(parsed.get("goal_text") or base.goal_text).strip() or base.goal_text
        return DbRagIntent(
            intent_id=base.intent_id,
            source_question=base.source_question,
            goal_text=goal_text,
            mode=base.mode,
            population=base.population,
            requested_fields=list(base.requested_fields),
            filters=list(base.filters),
            required_tables=required_tables,
            required_columns=required_columns,
            excluded_tables=excluded_tables,
            excluded_columns=excluded_columns,
            feedback_history=list(feedback_history),
            status=base.status,
        )
