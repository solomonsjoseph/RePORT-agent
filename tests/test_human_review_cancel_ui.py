from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from UI import ui_after_error_review
from UI import ui_before_run_review
from UI import ui_final_review
from UI import ui_human_review_rag_db_column_selection
from UI import ui_human_review_rag_db_sql_execution


class _FakeColumn:
    def __init__(self, streamlit: _FakeStreamlit) -> None:
        self._streamlit = streamlit

    def button(self, *args, **kwargs):
        return self._streamlit.button(*args, **kwargs)


class _FakeStreamlit:
    def __init__(self, *, clicked_label: str = "❌ Cancel", text_area_value: str = "ignore this feedback") -> None:
        self.clicked_label = clicked_label
        self.text_area_value = text_area_value
        self.session_state = {}
        self.rerun_count = 0

    def subheader(self, value):
        del value

    def caption(self, value):
        del value

    def markdown(self, value):
        del value

    def write(self, value):
        del value

    def warning(self, value):
        del value

    def success(self, value):
        del value

    def info(self, value):
        del value

    def text(self, value):
        del value

    def code(self, value, language=None):
        del value, language

    def image(self, value):
        del value

    def text_area(self, *args, **kwargs):
        del args, kwargs
        return self.text_area_value

    def columns(self, count):
        return [_FakeColumn(self) for _ in range(count)]

    def button(self, label, *args, **kwargs):
        del args, kwargs
        return label == self.clicked_label

    def expander(self, *args, **kwargs):
        del args, kwargs
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        del exc_type, exc, tb
        return False

    def error(self, value):
        raise AssertionError(f"unexpected Streamlit error: {value}")

    def stop(self):
        raise AssertionError("unexpected Streamlit stop")

    def rerun(self):
        self.rerun_count += 1


def _assert_cancel_emitted(monkeypatch, module, render) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(module, "st", fake_st)
    queued: list[tuple[str, dict]] = []

    render(lambda interrupt_id, decision: queued.append((interrupt_id, decision)))

    assert queued == [("interrupt-1", {"action": "cancel"})]
    assert fake_st.session_state["dismissed_interrupt_id"] == "interrupt-1"
    assert fake_st.rerun_count == 1


def test_ui_before_run_review_cancel_emits_cancel_without_suggestion(monkeypatch) -> None:
    _assert_cancel_emitted(
        monkeypatch,
        ui_before_run_review,
        lambda queue_resume: ui_before_run_review.ui_before_run_review(
            app=None,
            config={},
            payload={"generated_code": "print(1)", "code_summary": "summary"},
            interrupt_id="interrupt-1",
            queue_resume=queue_resume,
        ),
    )


def test_ui_after_error_review_cancel_emits_cancel_without_feedback_requirement(monkeypatch) -> None:
    _assert_cancel_emitted(
        monkeypatch,
        ui_after_error_review,
        lambda queue_resume: ui_after_error_review.ui_after_error_review(
            app=None,
            config={},
            payload={
                "generated_code": "print(1)",
                "error": {"type": "ValueError", "message": "bad column"},
            },
            interrupt_id="interrupt-1",
            queue_resume=queue_resume,
        ),
    )


def test_ui_final_review_cancel_emits_cancel_without_suggestion(monkeypatch) -> None:
    _assert_cancel_emitted(
        monkeypatch,
        ui_final_review,
        lambda queue_resume: ui_final_review.ui_final_review(
            app=None,
            config={},
            payload={"generated_code": "print(1)", "output": "ok", "code_summary": "summary"},
            interrupt_id="interrupt-1",
            queue_resume=queue_resume,
        ),
    )


def test_ui_human_review_rag_db_column_selection_cancel_emits_cancel_without_feedback(monkeypatch) -> None:
    _assert_cancel_emitted(
        monkeypatch,
        ui_human_review_rag_db_column_selection,
        lambda queue_resume: ui_human_review_rag_db_column_selection.ui_human_review_rag_db_column_selection(
            app=None,
            config={},
            payload={
                "goal_text": "extract ages",
                "columns": [{"table": "Form 2A", "column": "IC_AGE"}],
            },
            interrupt_id="interrupt-1",
            queue_resume=queue_resume,
        ),
    )


def test_ui_human_review_rag_db_sql_execution_cancel_emits_cancel_without_feedback(monkeypatch) -> None:
    _assert_cancel_emitted(
        monkeypatch,
        ui_human_review_rag_db_sql_execution,
        lambda queue_resume: ui_human_review_rag_db_sql_execution.ui_human_review_rag_db_sql_execution(
            app=None,
            config={},
            payload={
                "columns": [{"table": "Form 2A", "column": "IC_AGE"}],
                "sql": 'select "IC_AGE" from "Form 2A"',
            },
            interrupt_id="interrupt-1",
            queue_resume=queue_resume,
        ),
    )
