from __future__ import annotations

from UI.ui_after_error_review import _error_review_message


def test_error_review_message_for_policy_blocked() -> None:
    assert _error_review_message({"category": "policy_blocked"}) == (
        "Sandbox policy blocked this code. The app will not auto-retry."
    )


def test_error_review_message_for_fallback_case() -> None:
    assert _error_review_message({"category": "retryable_code"}) == (
        "The assistant hit repeated errors while executing the code."
    )
