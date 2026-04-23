from __future__ import annotations


def should_render_review_interrupt(
    interrupt_event,
    dismissed_interrupt_id: str,
    review_state: dict | None,
) -> bool:
    if not interrupt_event:
        return False
    if str(getattr(interrupt_event, "id", "")) == str(dismissed_interrupt_id or ""):
        return False

    payload = getattr(interrupt_event, "value", {}) or {}
    ui_type = payload.get("type")
    review_state = review_state or {}

    if ui_type == "before_run_review":
        return review_state.get("before_run_decision") is None
    if ui_type == "after_error_review":
        return review_state.get("after_error_decision") is None
    if ui_type == "human_review_rag_db_column_selection":
        return True
    if ui_type == "final_review":
        return review_state.get("final_decision") is None
    return False


def should_block_chat_submission(
    interrupt_event,
    dismissed_interrupt_id: str,
    review_state: dict | None,
) -> bool:
    return should_render_review_interrupt(
        interrupt_event,
        dismissed_interrupt_id=dismissed_interrupt_id,
        review_state=review_state,
    )


def blocking_review_notice(
    interrupt_event,
    dismissed_interrupt_id: str,
    review_state: dict | None,
) -> str | None:
    if not should_block_chat_submission(
        interrupt_event,
        dismissed_interrupt_id=dismissed_interrupt_id,
        review_state=review_state,
    ):
        return None
    return "Finish the active review before sending a new message."
