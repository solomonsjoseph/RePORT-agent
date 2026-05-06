from utils.streamlit_rendering import (
    canonicalize_conversation_history,
    conversation_controls_state,
    conversation_history_with_pending_user,
    normalize_submitted_question,
)


class Message:
    def __init__(self, content: str, msg_type: str = "ai"):
        self.content = content
        self.type = msg_type


def test_conversation_controls_stay_rendered_while_background_run_is_active() -> None:
    controls = conversation_controls_state({"state": "running"}, review_blocked=False)

    assert controls.render is True
    assert controls.submission_blocked is True
    assert controls.run_in_progress is True


def test_conversation_controls_stay_rendered_when_background_run_has_error() -> None:
    controls = conversation_controls_state({"state": "error"}, review_blocked=False)

    assert controls.render is True
    assert controls.submission_blocked is True
    assert controls.run_in_progress is False


def test_conversation_controls_allow_idle_submission_without_review_block() -> None:
    controls = conversation_controls_state({"state": "idle"}, review_blocked=False)

    assert controls.render is True
    assert controls.submission_blocked is False
    assert controls.run_in_progress is False


def test_conversation_controls_block_idle_submission_during_review() -> None:
    controls = conversation_controls_state({"state": "idle"}, review_blocked=True)

    assert controls.render is True
    assert controls.submission_blocked is True
    assert controls.run_in_progress is False


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


def test_canonicalize_conversation_history_drops_welcome_after_human_turn() -> None:
    history = canonicalize_conversation_history(
        [
            Message("Hello! Ask me anything ...", "ai"),
            Message("question", "human"),
        ],
        welcome_message="Hello! Ask me anything ...",
    )

    assert [msg.content for msg in history] == ["question"]


def test_canonicalize_conversation_history_keeps_welcome_without_human_turn() -> None:
    history = canonicalize_conversation_history(
        [Message("Hello! Ask me anything ...", "ai")],
        welcome_message="Hello! Ask me anything ...",
    )

    assert [msg.content for msg in history] == ["Hello! Ask me anything ..."]
