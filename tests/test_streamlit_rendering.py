from utils.streamlit_rendering import (
    build_autoscroll_key,
    canonicalize_conversation_history,
    conversation_render_state,
    conversation_controls_state,
    conversation_history_with_pending_user,
    normalize_submitted_question,
    pending_user_text_caught_up,
    should_render_compact_running_view,
)


class Message:
    def __init__(self, content: str, msg_type: str = "ai"):
        self.content = content
        self.type = msg_type


def test_conversation_controls_hide_while_background_run_is_active() -> None:
    controls = conversation_controls_state({"state": "running"}, review_blocked=False)

    assert controls.render is False
    assert controls.submission_blocked is True
    assert controls.run_in_progress is True


def test_conversation_controls_hide_when_background_run_has_error() -> None:
    controls = conversation_controls_state({"state": "error"}, review_blocked=False)

    assert controls.render is False
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


def test_compact_running_view_enabled_only_for_background_polling() -> None:
    assert should_render_compact_running_view({"state": "running"}, has_review_interrupt=False) is True
    assert should_render_compact_running_view({"state": "running"}, has_review_interrupt=True) is False
    assert should_render_compact_running_view({"state": "idle"}, has_review_interrupt=False) is False


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


def test_conversation_history_with_pending_user_does_not_duplicate_matching_message() -> None:
    persisted = Message("next question", "human")
    pending = Message("next question", "human")

    history = conversation_history_with_pending_user(
        [persisted],
        pending,
        welcome_message="Hello! Ask me anything ...",
    )

    assert history == [persisted]


def test_pending_user_text_caught_up_requires_matching_human_message() -> None:
    assert pending_user_text_caught_up([Message("older question", "human")], "new question") is False
    assert pending_user_text_caught_up([Message("new question", "human")], "new question") is True


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


def test_build_autoscroll_key_changes_when_interrupt_state_changes() -> None:
    history = [Message("question", "human"), Message("answer", "ai")]

    base = build_autoscroll_key(history, ui_type=None, should_render_interrupt=False)
    changed = build_autoscroll_key(
        history,
        ui_type="human_review_rag_db_sql_execution",
        should_render_interrupt=True,
    )

    assert base != changed


def test_conversation_render_state_preserves_full_history_while_running() -> None:
    history = [
        Message("older question", "human"),
        Message("older answer", "ai"),
        Message("latest follow up", "human"),
    ]

    render_state = conversation_render_state(
        history,
        run_status={"state": "running"},
        has_review_interrupt=False,
    )

    assert [msg.content for msg in render_state.messages] == [
        "older question",
        "older answer",
        "latest follow up",
    ]
    assert render_state.status_level == "info"
    assert render_state.status_text == "⏳ Working in background..."
    assert render_state.hide_secondary_sections is True


def test_conversation_render_state_omits_success_status_for_qa_response() -> None:
    history = [
        Message("question", "human"),
        Message("answer", "ai"),
    ]

    render_state = conversation_render_state(
        history,
        run_status={"state": "idle"},
        has_review_interrupt=False,
    )

    assert [msg.content for msg in render_state.messages][-1] == "answer"
    assert render_state.status_level is None
    assert render_state.status_text is None
    assert render_state.hide_secondary_sections is False


def test_conversation_render_state_omits_success_status_for_analysis_completion() -> None:
    history = [
        Message("question", "human"),
        Message("answer with dataset", "ai"),
    ]

    render_state = conversation_render_state(
        history,
        run_status={"state": "idle"},
        has_review_interrupt=False,
    )

    assert [msg.content for msg in render_state.messages][-1] == "answer with dataset"
    assert render_state.status_level is None
    assert render_state.status_text is None


def test_conversation_render_state_keeps_background_status_with_latest_message() -> None:
    history = [
        Message("older question", "human"),
        Message("older answer", "ai"),
        Message("latest follow up", "human"),
    ]

    render_state = conversation_render_state(
        history,
        run_status={"state": "running"},
        has_review_interrupt=False,
    )

    assert [msg.content for msg in render_state.messages][-1] == "latest follow up"
    assert render_state.status_level == "info"
    assert render_state.status_text == "⏳ Working in background..."
