from __future__ import annotations

import json
import sys
from types import ModuleType
from types import SimpleNamespace


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
