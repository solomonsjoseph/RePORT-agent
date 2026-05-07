from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from UI import ui_human_review_rag_db_sql_execution as review_ui


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

    def code(self, value, language=None):
        del language
        self.rendered.append(str(value))

    def text_area(self, *args, **kwargs):
        del args, kwargs
        return ""

    def columns(self, count):
        return [_FakeColumn() for _ in range(count)]


def test_sql_execution_review_omits_approved_tables_section(monkeypatch) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(review_ui, "st", fake_st)

    review_ui.ui_human_review_rag_db_sql_execution(
        app=None,
        config={},
        payload={
            "tables": ["Form 2A"],
            "columns": [{"table": "Form 2A", "column": "IC_AGE"}],
            "sql": 'select "IC_AGE" from "Form 2A"',
        },
        interrupt_id="interrupt-1",
        queue_resume=lambda *_args, **_kwargs: None,
    )

    assert "**Approved tables**" not in fake_st.rendered
    assert "- Form 2A" not in fake_st.rendered
    assert "- `Form 2A.IC_AGE`" in fake_st.rendered
