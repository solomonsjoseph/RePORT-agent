from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from UI import ui_human_review_rag_db_column_selection as review_ui


class _FakeColumn:
    def button(self, *args, **kwargs):
        del args, kwargs
        return False


class _FakeStreamlit:
    def __init__(self) -> None:
        self.session_state = {}
        self.rendered: list[str] = []

    def subheader(self, value):
        self.rendered.append(str(value))

    def caption(self, value):
        self.rendered.append(str(value))

    def markdown(self, value):
        self.rendered.append(str(value))

    def write(self, value):
        self.rendered.append(str(value))

    def info(self, value):
        self.rendered.append(str(value))

    def text_area(self, *args, **kwargs):
        del args, kwargs
        return ""

    def columns(self, count):
        return [_FakeColumn() for _ in range(count)]


def test_retrieval_fallback_message_is_not_rendered_twice(monkeypatch) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(review_ui, "st", fake_st)
    fallback_message = (
        "Structured ranking output was unavailable. Showing retrieved candidate tables and columns directly for human review."
    )

    review_ui.ui_human_review_rag_db_column_selection(
        app=None,
        config={},
        payload={
            "rationale": fallback_message,
            "selection_source": "retrieval_fallback",
            "fallback_reason": fallback_message,
            "tables": ["Form 2A"],
            "columns": [{"table": "Form 2A", "column": "IC_AGE"}],
        },
        interrupt_id="interrupt-1",
        queue_resume=lambda *_args, **_kwargs: None,
    )

    assert fake_st.rendered.count(fallback_message) == 1
