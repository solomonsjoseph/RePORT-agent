from utils.streamlit_rendering import (
    conversation_history_with_pending_user,
    normalize_submitted_question,
    should_render_conversation_controls,
)


class Message:
    def __init__(self, content: str):
        self.content = content


def test_conversation_controls_hidden_while_background_run_is_active() -> None:
    assert should_render_conversation_controls({"state": "running"}) is False


def test_conversation_controls_hidden_when_background_run_has_error() -> None:
    assert should_render_conversation_controls({"state": "error"}) is False


def test_conversation_controls_render_for_idle_background_run() -> None:
    assert should_render_conversation_controls({"state": "idle"}) is True


def test_normalize_submitted_question_trims_text() -> None:
    assert normalize_submitted_question("  hello  ", blocked=False) == "hello"


def test_normalize_submitted_question_ignores_empty_text() -> None:
    assert normalize_submitted_question("   ", blocked=False) is None


def test_normalize_submitted_question_ignores_blocked_submission() -> None:
    assert normalize_submitted_question("hello", blocked=True) is None


def test_conversation_history_with_pending_user_removes_welcome_before_append() -> None:
    pending = Message("question")

    history = conversation_history_with_pending_user(
        [Message("Hello! Ask me anything ...")],
        pending,
        welcome_message="Hello! Ask me anything ...",
    )

    assert history == [pending]


def test_conversation_history_with_pending_user_preserves_existing_messages() -> None:
    existing = Message("answer")
    pending = Message("next question")

    history = conversation_history_with_pending_user(
        [existing],
        pending,
        welcome_message="Hello! Ask me anything ...",
    )

    assert history == [existing, pending]
