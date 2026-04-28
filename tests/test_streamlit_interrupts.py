from __future__ import annotations

from types import SimpleNamespace

from utils.streamlit_interrupts import (
    blocking_review_notice,
    should_block_chat_submission,
    should_render_review_interrupt,
)


def test_should_block_chat_submission_for_active_before_run_review() -> None:
    interrupt_event = SimpleNamespace(
        id="int-1",
        value={"type": "before_run_review"},
    )

    assert should_block_chat_submission(
        interrupt_event,
        dismissed_interrupt_id="",
        review_state={"before_run_decision": None},
    ) is True


def test_should_not_block_chat_submission_for_dismissed_interrupt() -> None:
    interrupt_event = SimpleNamespace(
        id="int-1",
        value={"type": "before_run_review"},
    )

    assert should_block_chat_submission(
        interrupt_event,
        dismissed_interrupt_id="int-1",
        review_state={"before_run_decision": None},
    ) is False


def test_should_not_block_chat_submission_when_review_already_answered() -> None:
    interrupt_event = SimpleNamespace(
        id="int-1",
        value={"type": "final_review"},
    )

    assert should_block_chat_submission(
        interrupt_event,
        dismissed_interrupt_id="",
        review_state={"final_decision": "approve"},
    ) is False


def test_should_render_review_interrupt_for_db_column_selection_review() -> None:
    interrupt_event = SimpleNamespace(
        id="int-1",
        value={"type": "human_review_rag_db_column_selection"},
    )

    assert should_render_review_interrupt(
        interrupt_event,
        dismissed_interrupt_id="",
        review_state={"rag_db_column_review_decision": None},
    ) is True


def test_should_render_review_interrupt_for_db_sql_execution_review() -> None:
    interrupt_event = SimpleNamespace(
        id="int-1",
        value={"type": "human_review_rag_db_sql_execution"},
    )

    assert should_render_review_interrupt(
        interrupt_event,
        dismissed_interrupt_id="",
        review_state={"rag_db_sql_review_decision": None},
    ) is True


def test_blocking_review_notice_matches_review_type() -> None:
    interrupt_event = SimpleNamespace(
        id="int-1",
        value={"type": "after_error_review"},
    )

    assert blocking_review_notice(
        interrupt_event,
        dismissed_interrupt_id="",
        review_state={"after_error_decision": None},
    ) == "Finish the active review before sending a new message."
