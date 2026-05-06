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
        def __init__(self, content: str = "") -> None:
            self.content = content

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
from db_rag.service.selection_service import DbRagSelectionMixin


class _SelectionService(DbRagSelectionMixin):
    llm = object()

    def __init__(self, ranked_output=None) -> None:
        self.ranked_output = ranked_output

    def ground_feedback_constraints(self, *args, **kwargs) -> FeedbackConstraintSet:  # type: ignore[override]
        return FeedbackConstraintSet(goal_text="")

    def _rank_columns_with_openai(self, **kwargs):  # type: ignore[override]
        del kwargs
        return self.ranked_output


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
    assert selection.raw_model_output == '{"columns":["Missing||BAD"]}'
