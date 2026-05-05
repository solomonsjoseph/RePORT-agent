from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db_rag.service import classifier
from db_rag.service import intent_service


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = type("Msg", (), {"content": content})()


class _FakeCompletions:
    def __init__(self, owner: type["_FakeOpenAI"]) -> None:
        self._owner = owner

    def create(self, **kwargs):
        self._owner.last_create_kwargs = kwargs
        return type("Resp", (), {"choices": [_FakeChoice(self._owner.response_content)]})()


class _FakeChat:
    def __init__(self, owner: type["_FakeOpenAI"]) -> None:
        self.completions = _FakeCompletions(owner)


class _FakeOpenAI:
    response_content = ""
    last_init_kwargs: dict[str, object] | None = None
    last_create_kwargs: dict[str, object] | None = None

    def __init__(self, **kwargs) -> None:
        type(self).last_init_kwargs = kwargs
        self.chat = _FakeChat(type(self))


class _IntentService(intent_service.DbRagIntentMixin):
    llm = None


def _install_fake_openai(monkeypatch, *, content: str) -> type[_FakeOpenAI]:
    _FakeOpenAI.response_content = content
    _FakeOpenAI.last_init_kwargs = None
    _FakeOpenAI.last_create_kwargs = None
    monkeypatch.setattr(classifier, "_resolve_openai_client", lambda: _FakeOpenAI)
    return _FakeOpenAI


def test_classify_extraction_gate_message_returns_new_question(monkeypatch) -> None:
    monkeypatch.setenv("DB_RAG_REPLY_CLASSIFIER_API_KEY", "test-key")
    fake_openai = _install_fake_openai(
        monkeypatch,
        content='{"label":"new_question","confidence":0.93}',
    )

    result = classifier.classify_extraction_gate_message(
        pending_prompt="Would you like me to identify the tables and columns needed for a data extraction from this database question?",
        user_message="Which forms contain baseline smoking variables?",
        recent_transcript="Assistant: ...",
        active_intent={"intent_id": "i1", "goal_text": "age columns"},
        resolve_model=lambda: "gpt-4o-mini",
    )

    assert result == {"label": "new_question", "confidence": 0.93}
    assert fake_openai.last_init_kwargs == {"api_key": "test-key"}
    messages = fake_openai.last_create_kwargs["messages"]
    assert "Allowed labels: new_question, reply_to_pending_gate, unknown." in messages[0]["content"]
    assert '"goal_text": "age columns"' in messages[1]["content"]


def test_classify_extraction_gate_message_invalid_label_falls_back_to_unknown(monkeypatch) -> None:
    monkeypatch.setenv("DB_RAG_REPLY_CLASSIFIER_API_KEY", "test-key")
    _install_fake_openai(monkeypatch, content='{"label":"sideways","confidence":0.99}')

    result = classifier.classify_extraction_gate_message(
        pending_prompt="Would you like me to identify the tables and columns needed for a data extraction from this database question?",
        user_message="maybe",
        recent_transcript="Assistant: ...",
        active_intent=None,
        resolve_model=lambda: "gpt-4o-mini",
    )

    assert result == {"label": "unknown", "confidence": 0.0}


def test_classify_extraction_gate_message_returns_reply_to_pending_gate(monkeypatch) -> None:
    monkeypatch.setenv("DB_RAG_REPLY_CLASSIFIER_API_KEY", "test-key")
    _install_fake_openai(monkeypatch, content='{"label":"reply_to_pending_gate","confidence":0.82}')

    result = classifier.classify_extraction_gate_message(
        pending_prompt="Would you like me to identify the tables and columns needed for a data extraction from this database question?",
        user_message="Yes, please identify the relevant tables and columns.",
        recent_transcript="Assistant: ...",
        active_intent={"intent_id": "i1", "goal_text": "smoking variables"},
        resolve_model=lambda: "gpt-4o-mini",
    )

    assert result == {"label": "reply_to_pending_gate", "confidence": 0.82}


def test_classify_sql_review_feedback_selection_revision(monkeypatch) -> None:
    monkeypatch.setenv("DB_RAG_REPLY_CLASSIFIER_API_KEY", "test-key")
    fake_openai = _install_fake_openai(
        monkeypatch,
        content='{"label":"selection_revision","confidence":0.88}',
    )

    result = classifier.classify_sql_review_feedback(
        feedback_text="This query uses the wrong outcome table.",
        sql="select * from foo",
        resolve_model=lambda: "gpt-4o-mini",
    )

    assert result == {"label": "selection_revision", "confidence": 0.88}
    messages = fake_openai.last_create_kwargs["messages"]
    assert "Allowed labels: sql_only_revision, selection_revision, unknown." in messages[0]["content"]
    assert "SQL under review:\nselect * from foo" in messages[1]["content"]


def test_classify_sql_review_feedback_returns_sql_only_revision(monkeypatch) -> None:
    monkeypatch.setenv("DB_RAG_REPLY_CLASSIFIER_API_KEY", "test-key")
    _install_fake_openai(monkeypatch, content='{"label":"sql_only_revision","confidence":0.74}')

    result = classifier.classify_sql_review_feedback(
        feedback_text="Keep the same tables, but filter to enrollment visits only.",
        sql="select * from foo",
        resolve_model=lambda: "gpt-4o-mini",
    )

    assert result == {"label": "sql_only_revision", "confidence": 0.74}


def test_classify_sql_review_feedback_invalid_label_falls_back_to_unknown(monkeypatch) -> None:
    monkeypatch.setenv("DB_RAG_REPLY_CLASSIFIER_API_KEY", "test-key")
    _install_fake_openai(monkeypatch, content='{"label":"something_else","confidence":0.95}')

    result = classifier.classify_sql_review_feedback(
        feedback_text="Please revise this.",
        sql="select * from foo",
        resolve_model=lambda: "gpt-4o-mini",
    )

    assert result == {"label": "unknown", "confidence": 0.0}


def test_classify_extraction_gate_message_weak_confidence_falls_back_to_unknown(monkeypatch) -> None:
    monkeypatch.setenv("DB_RAG_REPLY_CLASSIFIER_API_KEY", "test-key")
    _install_fake_openai(monkeypatch, content='{"label":"reply_to_pending_gate","confidence":0.49}')

    result = classifier.classify_extraction_gate_message(
        pending_prompt="Would you like me to identify the tables and columns needed for a data extraction from this database question?",
        user_message="Yes, continue.",
        recent_transcript="Assistant: ...",
        active_intent={"intent_id": "i1"},
        resolve_model=lambda: "gpt-4o-mini",
    )

    assert result == {"label": "unknown", "confidence": 0.0}


def test_classify_extraction_gate_message_nonserializable_active_intent_falls_back_to_unknown(monkeypatch) -> None:
    monkeypatch.setenv("DB_RAG_REPLY_CLASSIFIER_API_KEY", "test-key")
    fake_openai = _install_fake_openai(
        monkeypatch,
        content='{"label":"reply_to_pending_gate","confidence":0.82}',
    )

    result = classifier.classify_extraction_gate_message(
        pending_prompt="Would you like me to identify the tables and columns needed for a data extraction from this database question?",
        user_message="Yes, continue.",
        recent_transcript="Assistant: ...",
        active_intent={"intent_id": object()},
        resolve_model=lambda: "gpt-4o-mini",
    )

    assert result == {"label": "unknown", "confidence": 0.0}
    assert fake_openai.last_create_kwargs is None


def test_classify_sql_review_feedback_weak_confidence_falls_back_to_unknown(monkeypatch) -> None:
    monkeypatch.setenv("DB_RAG_REPLY_CLASSIFIER_API_KEY", "test-key")
    _install_fake_openai(monkeypatch, content='{"label":"sql_only_revision","confidence":0.49}')

    result = classifier.classify_sql_review_feedback(
        feedback_text="Use the same selection but tighten the WHERE clause.",
        sql="select * from foo",
        resolve_model=lambda: "gpt-4o-mini",
    )

    assert result == {"label": "unknown", "confidence": 0.0}


def test_classify_sql_review_feedback_nonfinite_confidence_falls_back_to_unknown(monkeypatch) -> None:
    monkeypatch.setenv("DB_RAG_REPLY_CLASSIFIER_API_KEY", "test-key")
    _install_fake_openai(monkeypatch, content='{"label":"sql_only_revision","confidence":0.74}')
    monkeypatch.setattr(
        classifier,
        "parse_json_object",
        lambda _content: {"label": "sql_only_revision", "confidence": float("nan")},
    )

    result = classifier.classify_sql_review_feedback(
        feedback_text="Use the same selection but tighten the WHERE clause.",
        sql="select * from foo",
        resolve_model=lambda: "gpt-4o-mini",
    )

    assert result == {"label": "unknown", "confidence": 0.0}


def test_intent_mixin_exposes_extraction_gate_classifier() -> None:
    service = _IntentService()
    captured: dict[str, object] = {}

    def _fake_classifier(**kwargs):
        captured.update(kwargs)
        return {"label": "reply_to_pending_gate", "confidence": 0.61}

    result = None
    original = intent_service.classifier.classify_extraction_gate_message
    intent_service.classifier.classify_extraction_gate_message = _fake_classifier
    try:
        result = service.classify_extraction_gate_message(
            pending_prompt="Prompt",
            user_message="Yes, continue",
            recent_transcript="Assistant: Prompt",
            active_intent={"intent_id": "i1"},
        )
    finally:
        intent_service.classifier.classify_extraction_gate_message = original

    assert result == {"label": "reply_to_pending_gate", "confidence": 0.61}
    assert captured["resolve_model"] is intent_service.resolve_db_rag_reply_classifier_model


def test_intent_mixin_exposes_sql_review_feedback_classifier() -> None:
    service = _IntentService()
    captured: dict[str, object] = {}

    def _fake_classifier(**kwargs):
        captured.update(kwargs)
        return {"label": "sql_only_revision", "confidence": 0.72}

    result = None
    original = intent_service.classifier.classify_sql_review_feedback
    intent_service.classifier.classify_sql_review_feedback = _fake_classifier
    try:
        result = service.classify_sql_review_feedback(
            feedback_text="Use a WHERE clause for the study phase.",
            sql="select * from foo",
        )
    finally:
        intent_service.classifier.classify_sql_review_feedback = original

    assert result == {"label": "sql_only_revision", "confidence": 0.72}
    assert captured["resolve_model"] is intent_service.resolve_db_rag_reply_classifier_model
