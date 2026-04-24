from __future__ import annotations

import json
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest


def _install_langchain_message_stubs(monkeypatch) -> None:
    messages_mod = ModuleType("langchain_core.messages")

    class _Message:
        def __init__(self, content: str = "") -> None:
            self.content = content

    messages_mod.HumanMessage = _Message
    messages_mod.SystemMessage = _Message
    monkeypatch.setitem(sys.modules, "langchain_core.messages", messages_mod)


def test_openai_embedding_function_loads_dotenv_before_creating_client(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    import db_rag.service as service

    calls: list[str] = []

    def fake_load_dotenv() -> None:
        calls.append("load_dotenv")

    class _OpenAI:
        def __init__(self) -> None:
            assert calls == ["load_dotenv"]

    monkeypatch.setattr(service, "load_dotenv", fake_load_dotenv, raising=False)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_OpenAI))

    embedding_function = service.OpenAIEmbeddingFunction()

    assert embedding_function.model == "text-embedding-3-small"


def test_openai_embedding_function_exposes_chroma_name(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    import db_rag.service as service

    class _OpenAI:
        pass

    monkeypatch.setattr(service, "load_dotenv", lambda: None, raising=False)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_OpenAI))

    assert service.OpenAIEmbeddingFunction.name() == "openai"


def test_openai_embedding_function_exposes_chroma_embed_query(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    import db_rag.service as service

    calls: list[list[str]] = []

    class _Embeddings:
        def create(self, *, model, input):
            calls.append(input)
            return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 2.0, 3.0]) for _ in input])

    class _OpenAI:
        def __init__(self) -> None:
            self.embeddings = _Embeddings()

    monkeypatch.setattr(service, "load_dotenv", lambda: None, raising=False)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_OpenAI))

    embedding_function = service.OpenAIEmbeddingFunction()

    assert embedding_function.embed_query(input=["household contact"]) == [[1.0, 2.0, 3.0]]
    assert calls == [["household contact"]]


def test_resolve_embedding_model_requires_env_at_runtime(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    from db_rag import service

    monkeypatch.delenv("DB_RAG_EMBEDDING_MODEL", raising=False)

    db_rag_service = service.DbRagService(llm=object())
    readiness = db_rag_service.readiness()

    assert readiness["ready"] is False
    assert "DB_RAG_EMBEDDING_MODEL" in readiness["message"]
    assert ".env" in readiness["message"]


def test_openrouter_qwen_embedding_uses_openrouter_credentials(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    from db_rag import service

    captured: dict[str, str] = {}

    class _OpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.embeddings = SimpleNamespace(create=lambda **_: SimpleNamespace(data=[]))

    monkeypatch.setattr(service, "load_dotenv", lambda: None, raising=False)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_OpenAI))
    monkeypatch.setenv("DB_RAG_OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("DB_RAG_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    embedding_function = service.OpenAIEmbeddingFunction(model="Qwen/Qwen3-Embedding-4B")

    assert embedding_function.model == "Qwen/Qwen3-Embedding-4B"
    assert captured["api_key"] == "or-key"
    assert captured["base_url"] == "https://openrouter.ai/api/v1"


def test_db_rag_context_dataclasses_round_trip(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    from db_rag import service

    DbRagColumnHit = service.DbRagColumnHit
    DbRagContext = service.DbRagContext
    DbRagTableHit = service.DbRagTableHit

    context = DbRagContext(
        tables=[DbRagTableHit(table="Form 1A", text="Table summary")],
        columns=[DbRagColumnHit(table="Form 1A", column="AGE", text="Column summary")],
    )

    assert context.table_names == ["Form 1A"]
    assert context.column_names == ["AGE"]
    assert context.columns[0].as_prompt_line() == "Form 1A.AGE"


def test_retrieve_context_returns_typed_hits(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    from db_rag import service

    class _Collection:
        def __init__(self, result):
            self._result = result

        def query(self, **kwargs):
            return self._result

    table_collection = _Collection(
        {
            "documents": [["Table: Form 1A"]],
            "metadatas": [[{"table": "Form 1A"}]],
        }
    )
    column_collection = _Collection(
        {
            "documents": [["Column: AGE", "Column: SEX", "Column: OTHER"]],
            "metadatas": [[
                {"table": "Form 1A", "column": "AGE"},
                {"table": "Form 1A", "column": "SEX"},
                {"table": "Form 2A", "column": "OTHER"},
            ]],
        }
    )

    db_rag_service = service.DbRagService(llm=object())
    monkeypatch.setattr(
        db_rag_service,
        "_load_collections",
        lambda: (table_collection, column_collection),
    )

    context = db_rag_service.retrieve_context("age and sex")

    assert context.table_names == ["Form 1A"]
    assert context.column_names == ["AGE", "SEX"]
    assert "Table: Form 1A" in context.table_context
    assert "Column: AGE" in context.column_context
    assert "Column: OTHER" not in context.column_context


class _LLM:
    def __init__(self, content: str):
        self.content = content
        self.calls: list[list[object]] = []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self.content)


class _LLMRecorder:
    def __init__(self, content: str):
        self.content = content
        self.calls: list[list[object]] = []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(content=self.content)


def test_answer_from_context_returns_metadata_qa_without_sql(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    from db_rag import service

    context = service.DbRagContext(
        tables=[service.DbRagTableHit(table="Form 1A", text="Form 1A summary")],
        columns=[service.DbRagColumnHit(table="Form 1A", column="AGE", text="AGE summary")],
    )
    llm = _LLM(
        json.dumps(
            {
                "answer": "This database contains TB study forms.",
                "needs_sql": False,
                "rationale": "overview question answered from metadata",
            }
        )
    )
    db_rag_service = service.DbRagService(llm=llm)

    answer = db_rag_service.answer_from_context("Give me overview of the database", context)

    assert isinstance(answer, service.DbRagQaAnswer)
    assert answer.answer == "This database contains TB study forms."
    assert answer.needs_sql is False
    assert answer.relevant_tables == ["Form 1A"]
    assert answer.relevant_columns == ["AGE"]


def test_answer_from_context_marks_subset_request_as_sql_needed(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    from db_rag import service

    context = service.DbRagContext(
        tables=[service.DbRagTableHit(table="Form 1A", text="Form 1A summary")],
        columns=[service.DbRagColumnHit(table="Form 1A", column="AGE", text="AGE summary")],
    )
    llm = _LLM(
        json.dumps(
            {
                "answer": "A row-level subset is needed for this request.",
                "needs_sql": True,
                "rationale": "row-level subset requested",
            }
        )
    )
    db_rag_service = service.DbRagService(llm=llm)

    answer = db_rag_service.answer_from_context("Give me the subset of records", context)

    assert isinstance(answer, service.DbRagQaAnswer)
    assert answer.needs_sql is True
    assert answer.rationale == "row-level subset requested"


def test_answer_from_context_fails_closed_on_weakly_typed_structured_output(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    from db_rag import service

    context = service.DbRagContext(
        tables=[service.DbRagTableHit(table="Form 1A", text="Form 1A summary")],
        columns=[service.DbRagColumnHit(table="Form 1A", column="AGE", text="AGE summary")],
    )
    llm = _LLM('{"answer":"Subset requested.","needs_sql":"true","rationale":"row-level subset requested"}')
    db_rag_service = service.DbRagService(llm=llm)

    answer = db_rag_service.answer_from_context("Give me the subset of records", context)

    assert isinstance(answer, service.DbRagQaAnswer)
    assert answer.needs_sql is True
    assert answer.answer == "I could not answer from the retrieved DB-RAG context."
    assert answer.rationale == "Invalid structured response from the model."


def test_answer_from_context_sends_table_and_column_context_to_llm(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    from db_rag import service

    context = service.DbRagContext(
        tables=[service.DbRagTableHit(table="Form 1A", text="Form 1A summary")],
        columns=[service.DbRagColumnHit(table="Form 1A", column="AGE", text="AGE summary")],
        table_context="Form 1A summary text",
        column_context="AGE summary text",
    )
    llm = _LLMRecorder(
        json.dumps(
            {
                "answer": "This database contains TB study forms.",
                "needs_sql": False,
                "rationale": "overview question answered from metadata",
            }
        )
    )
    db_rag_service = service.DbRagService(llm=llm)

    answer = db_rag_service.answer_from_context("Give me overview of the database", context)

    assert isinstance(answer, service.DbRagQaAnswer)
    assert len(llm.calls) == 1
    message_text = "\n".join(getattr(message, "content", "") for message in llm.calls[0])
    assert "Table context:\nForm 1A summary text" in message_text
    assert "Column context:\nAGE summary text" in message_text


def test_prepare_column_selection_uses_feedback_history(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    from db_rag import service

    captured: list[list[object]] = []

    class _LLM:
        def invoke(self, messages):
            captured.append(messages)
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "selection_id": "sel-1",
                        "rationale": "subset request needs age and sex columns",
                        "tables": ["Form 1A"],
                        "columns": [
                            {
                                "table": "Form 1A",
                                "column": "AGE",
                                "description": "Age in years",
                            },
                            {
                                "table": "Form 1A",
                                "column": "SEX",
                                "description": "Sex at enrollment",
                            },
                        ],
                    }
                )
            )

    context = service.DbRagContext(
        tables=[service.DbRagTableHit(table="Form 1A", text="Form 1A summary")],
        columns=[service.DbRagColumnHit(table="Form 1A", column="AGE", text="AGE summary")],
    )
    db_rag_service = service.DbRagService(llm=_LLM())

    selection = db_rag_service.prepare_column_selection(
        "subset age and sex",
        context,
        feedback_history=[{"feedback": "Include gender explicitly."}],
    )

    assert selection.selection_id == "sel-1"
    assert selection.tables == ["Form 1A"]
    assert selection.columns[0]["column"] == "AGE"
    assert selection.feedback_history == [{"feedback": "Include gender explicitly."}]

    message_text = "\n".join(getattr(message, "content", "") for message in captured[0])
    assert "Include gender explicitly." in message_text


def test_prepare_column_selection_filters_invented_pairs(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    from db_rag import service

    class _LLM:
        def invoke(self, messages):
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "selection_id": "sel-2",
                        "rationale": "keep the exact retrieved pairs",
                        "tables": ["Form 1A", "Invented Form"],
                        "columns": [
                            {
                                "table": "Form 1A",
                                "column": "AGE",
                                "description": "Age in years",
                            },
                            {
                                "table": "Form 1A",
                                "column": "HEIGHT",
                                "description": "Not present in context",
                            },
                            {
                                "table": "Invented Form",
                                "column": "SEX",
                                "description": "Invented pair",
                            },
                        ],
                    }
                )
            )

    context = service.DbRagContext(
        tables=[service.DbRagTableHit(table="Form 1A", text="Form 1A summary")],
        columns=[
            service.DbRagColumnHit(table="Form 1A", column="AGE", text="AGE summary"),
            service.DbRagColumnHit(table="Form 1A", column="SEX", text="SEX summary"),
        ],
    )
    db_rag_service = service.DbRagService(llm=_LLM())

    selection = db_rag_service.prepare_column_selection("subset age and sex", context)

    assert selection.selection_id == "sel-2"
    assert selection.tables == ["Form 1A"]
    assert selection.columns == [
        {
            "table": "Form 1A",
            "column": "AGE",
            "description": "Age in years",
        }
    ]


def test_prepare_column_selection_derives_tables_when_omitted(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    from db_rag import service

    class _LLM:
        def invoke(self, messages):
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "selection_id": "sel-3",
                        "rationale": "tables omitted but the column is valid",
                        "columns": [
                            {
                                "table": "Form 1A",
                                "column": "AGE",
                                "description": "Age in years",
                            }
                        ],
                    }
                )
            )

    context = service.DbRagContext(
        tables=[service.DbRagTableHit(table="Form 1A", text="Form 1A summary")],
        columns=[service.DbRagColumnHit(table="Form 1A", column="AGE", text="AGE summary")],
    )
    db_rag_service = service.DbRagService(llm=_LLM())

    selection = db_rag_service.prepare_column_selection("subset age", context)

    assert selection.selection_id == "sel-3"
    assert selection.tables == ["Form 1A"]
    assert selection.columns == [
        {
            "table": "Form 1A",
            "column": "AGE",
            "description": "Age in years",
        }
    ]


@pytest.mark.parametrize(
    "content",
    [
        "not json at all",
        "{}",
    ],
)
def test_prepare_column_selection_fails_closed_on_invalid_output(monkeypatch, content) -> None:
    _install_langchain_message_stubs(monkeypatch)

    from db_rag import service

    class _LLM:
        def invoke(self, messages):
            return SimpleNamespace(content=content)

    context = service.DbRagContext(
        tables=[service.DbRagTableHit(table="Form 1A", text="Form 1A summary")],
        columns=[service.DbRagColumnHit(table="Form 1A", column="AGE", text="AGE summary")],
    )
    db_rag_service = service.DbRagService(llm=_LLM())

    selection = db_rag_service.prepare_column_selection(
        "subset age and sex",
        context,
        feedback_history=[{"feedback": "Include gender explicitly."}],
    )

    expected_selection_id = service._default_selection_id(
        "subset age and sex",
        context,
        [{"feedback": "Include gender explicitly."}],
    )
    assert selection.selection_id == expected_selection_id
    assert selection.question == "subset age and sex"
    assert selection.tables == []
    assert selection.columns == []
    assert selection.rationale == "Invalid structured response from the model."
    assert selection.feedback_history == [{"feedback": "Include gender explicitly."}]


def test_prepare_sql_candidate_uses_approved_columns_only(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    from db_rag import service

    captured: list[list[object]] = []

    class _LLM:
        def invoke(self, messages):
            captured.append(messages)
            return SimpleNamespace(content='SELECT "AGE", "SEX" FROM "Form 1A"')

    selection = service.ColumnSelectionCandidate(
        selection_id="sel-approved",
        question="subset age and sex",
        tables=["Form 1A"],
        columns=[
            {
                "table": "Form 1A",
                "column": "AGE",
                "description": "Age in years",
            },
            {
                "table": "Form 1A",
                "column": "SEX",
                "description": "Sex at enrollment",
            },
        ],
        rationale="approved selection",
        status="approved",
    )
    db_rag_service = service.DbRagService(llm=_LLM())

    candidate = db_rag_service.prepare_sql_candidate("subset age and sex", selection)

    assert candidate.sql == 'SELECT "AGE", "SEX" FROM "Form 1A"'
    assert candidate.columns == selection.columns
    message_text = "\n".join(getattr(message, "content", "") for message in captured[0])
    assert "AGE" in message_text
    assert "not approved" not in message_text


def test_prepare_sql_candidate_rejects_unapproved_selection(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    from db_rag import service

    selection = service.ColumnSelectionCandidate(
        selection_id="sel-awaiting-review",
        question="subset age and sex",
        tables=["Form 1A"],
        columns=[
            {
                "table": "Form 1A",
                "column": "AGE",
                "description": "Age in years",
            }
        ],
        rationale="awaiting approval",
        status="awaiting_review",
    )
    db_rag_service = service.DbRagService(llm=object())

    with pytest.raises(ValueError, match="approved"):
        db_rag_service.prepare_sql_candidate("subset age and sex", selection)
