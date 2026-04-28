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

    import db_rag.vectorstore as vectorstore

    calls: list[str] = []

    def fake_load_dotenv() -> None:
        calls.append("load_dotenv")

    class _OpenAI:
        def __init__(self) -> None:
            assert calls == ["load_dotenv"]

    monkeypatch.setattr(vectorstore, "load_dotenv", fake_load_dotenv, raising=False)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_OpenAI))

    embedding_function = vectorstore.OpenAIEmbeddingFunction(model="OpenAI/text-embedding-3-small")

    assert embedding_function.model == "text-embedding-3-small"
    assert embedding_function.config_model == "OpenAI/text-embedding-3-small"


def test_openai_embedding_function_exposes_chroma_name(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    import db_rag.vectorstore as vectorstore

    class _OpenAI:
        pass

    monkeypatch.setattr(vectorstore, "load_dotenv", lambda: None, raising=False)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_OpenAI))

    assert vectorstore.OpenAIEmbeddingFunction.name() == "openai"


def test_openai_embedding_function_exposes_chroma_embed_query(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    import db_rag.vectorstore as vectorstore

    calls: list[list[str]] = []

    class _Embeddings:
        def create(self, *, model, input):
            calls.append(input)
            return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 2.0, 3.0]) for _ in input])

    class _OpenAI:
        def __init__(self) -> None:
            self.embeddings = _Embeddings()

    monkeypatch.setattr(vectorstore, "load_dotenv", lambda: None, raising=False)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_OpenAI))

    embedding_function = vectorstore.OpenAIEmbeddingFunction(model="OpenAI/text-embedding-3-small")

    assert embedding_function.embed_query(input=["household contact"]) == [[1.0, 2.0, 3.0]]
    assert calls == [["household contact"]]


def test_openai_embedding_function_normalizes_single_query_string(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    import db_rag.vectorstore as vectorstore

    calls: list[list[str]] = []

    class _Embeddings:
        def create(self, *, model, input):
            calls.append(input)
            return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 2.0, 3.0]) for _ in input])

    class _OpenAI:
        def __init__(self) -> None:
            self.embeddings = _Embeddings()

    monkeypatch.setattr(vectorstore, "load_dotenv", lambda: None, raising=False)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_OpenAI))

    embedding_function = vectorstore.OpenAIEmbeddingFunction(model="OpenAI/text-embedding-3-small")

    assert embedding_function.embed_query(input="household contact") == [[1.0, 2.0, 3.0]]
    assert calls == [["household contact"]]


def test_openai_embedding_function_retries_when_provider_returns_no_embedding_data(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    import db_rag.vectorstore as vectorstore

    calls: list[list[str]] = []
    responses = [
        SimpleNamespace(data=None),
        SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 2.0, 3.0])]),
    ]

    class _Embeddings:
        def create(self, *, model, input):
            calls.append(input)
            return responses.pop(0)

    class _OpenAI:
        def __init__(self) -> None:
            self.embeddings = _Embeddings()

    monkeypatch.setattr(vectorstore, "load_dotenv", lambda: None, raising=False)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_OpenAI))

    embedding_function = vectorstore.OpenAIEmbeddingFunction(model="OpenAI/text-embedding-3-small")

    assert embedding_function.embed_query(input="household contact") == [[1.0, 2.0, 3.0]]
    assert calls == [["household contact"], ["household contact"]]


def test_resolve_embedding_model_requires_env_at_runtime(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    from db_rag import service

    def _raise_missing():
        raise ValueError("DB_RAG_EMBEDDING_MODEL is not set. Set it in .env or export it in your shell.")

    monkeypatch.setattr(service, "resolve_db_rag_embedding_model", _raise_missing)

    db_rag_service = service.DbRagService(llm=object())
    readiness = db_rag_service.readiness()

    assert readiness["ready"] is False
    assert "DB_RAG_EMBEDDING_MODEL" in readiness["message"]
    assert ".env" in readiness["message"]


def test_supported_reranker_models_only_include_actual_rerankers(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    from db_rag import config

    assert config.SUPPORTED_DB_RAG_RERANKER_MODELS == (
        "cohere/rerank-v3.5",
        "cohere/rerank-4-fast",
        "cohere/rerank-4-pro",
    )
    assert not hasattr(config, "DEFAULT_DB_RAG_RERANKER_BY_EMBEDDING")


def test_resolve_db_rag_reranker_model_returns_none_when_unset(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    from db_rag import config

    monkeypatch.delenv("DB_RAG_RERANKER_MODEL", raising=False)

    assert config.resolve_db_rag_reranker_model() is None


def test_resolve_db_rag_reranker_model_validates_supported_values(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    from db_rag import config

    monkeypatch.setenv("DB_RAG_RERANKER_MODEL", "cohere/rerank-v3.5")
    assert config.resolve_db_rag_reranker_model() == "cohere/rerank-v3.5"

    monkeypatch.setenv("DB_RAG_RERANKER_MODEL", "bad/model")
    with pytest.raises(ValueError, match="Unsupported DB_RAG_RERANKER_MODEL"):
        config.resolve_db_rag_reranker_model()


def test_openrouter_reranker_uses_openrouter_credentials(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    import db_rag.vectorstore as vectorstore

    monkeypatch.setattr(vectorstore, "load_dotenv", lambda: None, raising=False)
    monkeypatch.setenv("DB_RAG_OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("DB_RAG_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    reranker = vectorstore.OpenAIReranker(model="cohere/rerank-v3.5")

    assert reranker.model == "cohere/rerank-v3.5"
    assert reranker.config_model == "cohere/rerank-v3.5"
    assert reranker.api_key == "or-key"
    assert reranker.base_url == "https://openrouter.ai/api/v1"


def test_openrouter_qwen_embedding_uses_openrouter_credentials(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    import db_rag.vectorstore as vectorstore

    captured: dict[str, str] = {}

    class _OpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.embeddings = SimpleNamespace(create=lambda **_: SimpleNamespace(data=[]))

    monkeypatch.setattr(vectorstore, "load_dotenv", lambda: None, raising=False)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_OpenAI))
    monkeypatch.setenv("DB_RAG_OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("DB_RAG_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    embedding_function = vectorstore.OpenAIEmbeddingFunction(model="Qwen/Qwen3-Embedding-4B")

    assert embedding_function.model == "qwen/qwen3-embedding-4b"
    assert embedding_function.config_model == "Qwen/Qwen3-Embedding-4B"
    assert captured["api_key"] == "or-key"
    assert captured["base_url"] == "https://openrouter.ai/api/v1"


def test_openrouter_qwen_8b_embedding_uses_openrouter_credentials(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    import db_rag.vectorstore as vectorstore

    captured: dict[str, str] = {}

    class _OpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.embeddings = SimpleNamespace(create=lambda **_: SimpleNamespace(data=[]))

    monkeypatch.setattr(vectorstore, "load_dotenv", lambda: None, raising=False)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_OpenAI))
    monkeypatch.setenv("DB_RAG_OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("DB_RAG_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    embedding_function = vectorstore.OpenAIEmbeddingFunction(model="Qwen/Qwen3-Embedding-8B")

    assert embedding_function.model == "qwen/qwen3-embedding-8b"
    assert embedding_function.config_model == "Qwen/Qwen3-Embedding-8B"
    assert captured["api_key"] == "or-key"
    assert captured["base_url"] == "https://openrouter.ai/api/v1"


def test_openrouter_qwen_embedding_requests_float_embeddings(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    import db_rag.vectorstore as vectorstore

    calls: list[dict[str, object]] = []

    class _Embeddings:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 2.0, 3.0])])

    class _OpenAI:
        def __init__(self, **kwargs) -> None:
            self.embeddings = _Embeddings()

    monkeypatch.setattr(vectorstore, "load_dotenv", lambda: None, raising=False)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=_OpenAI))
    monkeypatch.setenv("DB_RAG_OPENROUTER_API_KEY", "or-key")
    monkeypatch.setenv("DB_RAG_OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

    embedding_function = vectorstore.OpenAIEmbeddingFunction(model="Qwen/Qwen3-Embedding-4B")
    embedding_function(["semantic search query"])

    assert calls == [
        {
            "model": "qwen/qwen3-embedding-4b",
            "input": ["semantic search query"],
            "encoding_format": "float",
        }
    ]


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


def test_retrieve_context_falls_back_to_single_query_when_decomposition_returns_one_phrase(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    from db_rag import service

    calls: list[tuple[str, str]] = []

    class _Collection:
        def __init__(self, name: str, result: dict[str, list[list[object]]]) -> None:
            self.name = name
            self._result = result

        def query(self, **kwargs):
            calls.append((self.name, kwargs["query_texts"][0]))
            return self._result

    db_rag_service = service.DbRagService(llm=SimpleNamespace(invoke=lambda _: SimpleNamespace(content="age")))
    monkeypatch.setattr(
        db_rag_service,
        "_load_collections",
        lambda: (
            _Collection("tables", {"documents": [["Table: Form 1A"]], "metadatas": [[{"table": "Form 1A"}]]}),
            _Collection("columns", {"documents": [["Column: AGE"]], "metadatas": [[{"table": "Form 1A", "column": "AGE"}]]}),
        ),
    )

    context = db_rag_service.retrieve_context("age and sex")

    assert context.table_names == ["Form 1A"]
    assert calls == [("tables", "age and sex"), ("columns", "age and sex")]


def test_retrieve_context_debug_prints_single_query_fallback(monkeypatch, capsys) -> None:
    _install_langchain_message_stubs(monkeypatch)

    from db_rag import service

    class _Collection:
        def __init__(self, result):
            self._result = result

        def query(self, **kwargs):
            return self._result

    db_rag_service = service.DbRagService(llm=SimpleNamespace(invoke=lambda _: SimpleNamespace(content="age")))
    monkeypatch.setattr(
        db_rag_service,
        "_load_collections",
        lambda: (
            _Collection({"documents": [["Table: Form 1A"]], "metadatas": [[{"table": "Form 1A"}]]}),
            _Collection({"documents": [["Column: AGE"]], "metadatas": [[{"table": "Form 1A", "column": "AGE"}]]}),
        ),
    )

    context = db_rag_service.retrieve_context("age and sex", debug=True)

    output = capsys.readouterr().out
    assert context.table_names == ["Form 1A"]
    assert "Single-concept query, skipping decomposition" in output
    assert "Table retrieval:" in output
    assert "Column retrieval:" in output


def test_retrieve_context_merges_multiquery_hits(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    from db_rag import service

    table_queries = {
        "age": {"documents": [["Table: Form 1A"]], "metadatas": [[{"table": "Form 1A"}]]},
        "outcome": {"documents": [["Table: Final Outcome"]], "metadatas": [[{"table": "Final Outcome"}]]},
    }
    column_queries = {
        "age": {"documents": [["Column: AGE"]], "metadatas": [[{"table": "Form 1A", "column": "AGE"}]]},
        "outcome": {"documents": [["Column: OUTCOME"]], "metadatas": [[{"table": "Final Outcome", "column": "OUTCOME"}]]},
    }

    class _Collection:
        def __init__(self, result_map: dict[str, dict[str, list[list[object]]]]) -> None:
            self.result_map = result_map

        def query(self, **kwargs):
            return self.result_map[kwargs["query_texts"][0]]

    llm = SimpleNamespace(invoke=lambda _: SimpleNamespace(content="age\noutcome"))
    db_rag_service = service.DbRagService(llm=llm)
    monkeypatch.setattr(
        db_rag_service,
        "_load_collections",
        lambda: (_Collection(table_queries), _Collection(column_queries)),
    )

    context = db_rag_service.retrieve_context("age and final outcome")

    assert context.table_names == ["Form 1A", "Final Outcome"]
    assert context.column_names == ["AGE", "OUTCOME"]


def test_retrieve_context_debug_prints_multiconcept_merge(monkeypatch, capsys) -> None:
    _install_langchain_message_stubs(monkeypatch)

    from db_rag import service

    table_queries = {
        "age": {"documents": [["Table: Form 1A"]], "metadatas": [[{"table": "Form 1A"}]]},
        "outcome": {
            "documents": [["Table: Final Outcome Determination Form - Cohort A (Active Pulmonary TB)"]],
            "metadatas": [[{"table": "Final Outcome Determination Form - Cohort A (Active Pulmonary TB)"}]],
        },
        "Form 1A - Index Case Screening": {
            "documents": [["Table: Form 1A - Index Case Screening"]],
            "metadatas": [[{"table": "Form 1A - Index Case Screening"}]],
        },
    }
    column_queries = {
        "age": {"documents": [["Column: AGE"]], "metadatas": [[{"table": "Form 1A", "column": "AGE"}]]},
        "outcome": {
            "documents": [["Column: OUTCOME"]],
            "metadatas": [[{"table": "Final Outcome Determination Form - Cohort A (Active Pulmonary TB)", "column": "OUTCOME"}]],
        },
    }

    class _Collection:
        def __init__(self, result_map):
            self.result_map = result_map

        def query(self, **kwargs):
            return self.result_map[kwargs["query_texts"][0]]

    llm = SimpleNamespace(invoke=lambda _: SimpleNamespace(content="age\noutcome"))
    db_rag_service = service.DbRagService(llm=llm)
    monkeypatch.setattr(
        db_rag_service,
        "_load_collections",
        lambda: (_Collection(table_queries), _Collection(column_queries)),
    )

    context = db_rag_service.retrieve_context("age and final outcome", debug=True)

    output = capsys.readouterr().out
    assert "Query decomposition:" in output
    assert "Injected paired form: Form 1A - Index Case Screening" in output
    assert "Merged tables:" in output
    assert "Merged column candidates:" in output
    assert "Form 1A - Index Case Screening" in context.table_names


def test_retrieve_context_records_reranks_merged_columns(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    from db_rag import retrieval

    table_queries = {
        "gender": {"documents": [["Table: Form 1A"]], "metadatas": [[{"table": "Form 1A"}]]},
        "outcome": {"documents": [["Table: Final Outcome"]], "metadatas": [[{"table": "Final Outcome"}]]},
    }
    column_queries = {
        "gender": {
            "documents": [["Column: SEX", "Column: AGE"]],
            "metadatas": [[
                {"table": "Form 1A", "column": "SEX"},
                {"table": "Form 1A", "column": "AGE"},
            ]],
        },
        "outcome": {
            "documents": [["Column: OUTCOME", "Column: STATUS"]],
            "metadatas": [[
                {"table": "Final Outcome", "column": "OUTCOME"},
                {"table": "Final Outcome", "column": "STATUS"},
            ]],
        },
    }

    class _Collection:
        def __init__(self, result_map):
            self.result_map = result_map

        def query(self, **kwargs):
            return self.result_map[kwargs["query_texts"][0]]

    monkeypatch.setattr(retrieval, "decompose_query", lambda llm, question: ["gender", "outcome"])
    monkeypatch.setattr(
        retrieval,
        "rerank_columns",
        lambda query, column_hits, *, reranker_model, top_k, debug=False: list(reversed(column_hits)),
    )

    tables, columns = retrieval.retrieve_context_records(
        object(),
        _Collection(table_queries),
        _Collection(column_queries),
        "gender outcome question",
        reranker_model="cohere/rerank-v3.5",
    )

    assert [entry["table"] for entry in tables] == ["Form 1A", "Final Outcome"]
    assert [entry["column"] for entry in columns] == ["AGE", "SEX", "STATUS", "OUTCOME"]


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


def test_resolve_intent_returns_extraction_intent(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    from db_rag import service

    class _LLM:
        def invoke(self, _messages):
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "intent_id": "intent-1",
                        "goal_text": "subset age and sex among index cases",
                        "mode": "extraction",
                        "population": "index cases",
                        "requested_fields": ["age", "sex"],
                        "filters": [],
                    }
                )
            )

    db_rag_service = service.DbRagService(llm=_LLM())
    context = service.DbRagContext(
        tables=[service.DbRagTableHit(table="Form 2A - INDEX CASE: Clinical/Demographic Form", text="table summary")],
        columns=[
            service.DbRagColumnHit(
                table="Form 2A - INDEX CASE: Clinical/Demographic Form",
                column="IS_AGE",
                text="age summary",
            )
        ],
    )

    intent = db_rag_service.resolve_intent("subset age and sex among index cases", context)

    assert intent.intent_id == "intent-1"
    assert intent.mode == "extraction"
    assert intent.population == "index cases"
    assert intent.requested_fields == ["age", "sex"]


def test_validate_selection_against_intent_rejects_excluded_table(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    from db_rag import service

    db_rag_service = service.DbRagService(llm=object())
    intent = service.DbRagIntent(
        intent_id="intent-1",
        source_question="subset age",
        goal_text="subset age",
        mode="extraction",
        population=None,
        requested_fields=["age"],
        filters=[],
        required_tables=[],
        required_columns=[],
        excluded_tables=["Form 13 - TB Treatment Compliance Form"],
        excluded_columns=[],
        feedback_history=[],
        status="active",
    )

    candidate = service.ColumnSelectionCandidate(
        selection_id="sel-1",
        question="subset age",
        tables=["Form 13 - TB Treatment Compliance Form"],
        columns=[{"table": "Form 13 - TB Treatment Compliance Form", "column": "EXTRPERI", "description": ""}],
        rationale="bad selection",
    )

    ok, error_message = db_rag_service.validate_selection_against_intent(candidate, intent)

    assert ok is False
    assert "excluded table" in error_message.lower()


def test_retrieve_context_for_intent_applies_required_and_excluded_tables(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    from db_rag import service

    captured: dict[str, object] = {}

    def _fake_retrieve_context_records(_llm, _table_collection, _column_collection, question, **kwargs):
        captured["question"] = question
        captured["required_tables"] = kwargs.get("required_tables")
        captured["excluded_tables"] = kwargs.get("excluded_tables")
        return (
            [{"table": "Form 2A - INDEX CASE: Clinical/Demographic Form", "text": "table summary"}],
            [{"table": "Form 2A - INDEX CASE: Clinical/Demographic Form", "column": "IS_AGE", "text": "age summary"}],
        )

    monkeypatch.setattr(service, "retrieve_context_records", _fake_retrieve_context_records)

    class _DbRagService(service.DbRagService):
        def _load_collections(self):
            return object(), object()

    intent = service.DbRagIntent(
        intent_id="intent-1",
        source_question="subset age",
        goal_text="subset age among index cases",
        mode="extraction",
        population="index cases",
        requested_fields=["age"],
        filters=[],
        required_tables=["Form 2A - INDEX CASE: Clinical/Demographic Form"],
        required_columns=[],
        excluded_tables=["Form 13 - TB Treatment Compliance Form"],
        excluded_columns=[],
        feedback_history=[],
        status="active",
    )

    db_rag_service = _DbRagService(llm=object())
    context = db_rag_service.retrieve_context_for_intent(intent)

    assert captured["question"] == "subset age among index cases"
    assert captured["required_tables"] == ["Form 2A - INDEX CASE: Clinical/Demographic Form"]
    assert captured["excluded_tables"] == ["Form 13 - TB Treatment Compliance Form"]
    assert context.table_names == ["Form 2A - INDEX CASE: Clinical/Demographic Form"]


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


def test_prepare_column_selection_fails_closed_when_model_returns_tables_but_no_columns(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    from db_rag import service

    class _LLM:
        def invoke(self, messages):
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "selection_id": "sel-empty-columns",
                        "rationale": "picked a table but no exact columns",
                        "tables": ["Form 1A"],
                        "columns": [],
                    }
                )
            )

    context = service.DbRagContext(
        tables=[service.DbRagTableHit(table="Form 1A", text="Form 1A summary")],
        columns=[service.DbRagColumnHit(table="Form 1A", column="AGE", text="AGE summary")],
    )
    db_rag_service = service.DbRagService(llm=_LLM())

    selection = db_rag_service.prepare_column_selection("subset age", context)

    assert selection.tables == []
    assert selection.columns == []
    assert selection.rationale == "Invalid structured response from the model."


def test_prepare_column_selection_preserves_explicit_schema_column_mentions(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    from db_rag import service

    class _LLM:
        def invoke(self, messages):
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "selection_id": "sel-explicit",
                        "rationale": "model only returned one retrieved column",
                        "tables": ["Form 1A - Index Case Screening"],
                        "columns": [
                            {
                                "table": "Form 1A - Index Case Screening",
                                "column": "IS_AGE",
                                "description": "Age in years",
                            }
                        ],
                    }
                )
            )

    monkeypatch.setattr(
        service,
        "_schema_column_catalog",
        lambda: (
            {
                ("Form 1A - Index Case Screening", "IS_AGE"): {
                    "table": "Form 1A - Index Case Screening",
                    "column": "IS_AGE",
                    "description": "Age in years",
                },
                ("Form 2A - INDEX CASE: Clinical/Demographic Form", "IC_DMDX"): {
                    "table": "Form 2A - INDEX CASE: Clinical/Demographic Form",
                    "column": "IC_DMDX",
                    "description": "Diabetes diagnosis status",
                },
            },
            {
                "IS_AGE": [
                    {
                        "table": "Form 1A - Index Case Screening",
                        "column": "IS_AGE",
                        "description": "Age in years",
                    }
                ],
                "IC_DMDX": [
                    {
                        "table": "Form 2A - INDEX CASE: Clinical/Demographic Form",
                        "column": "IC_DMDX",
                        "description": "Diabetes diagnosis status",
                    }
                ],
            },
        ),
    )

    context = service.DbRagContext(
        tables=[service.DbRagTableHit(table="Form 1A - Index Case Screening", text="Form 1A summary")],
        columns=[service.DbRagColumnHit(table="Form 1A - Index Case Screening", column="IS_AGE", text="IS_AGE summary")],
    )
    db_rag_service = service.DbRagService(llm=_LLM())

    selection = db_rag_service.prepare_column_selection(
        "Subset Form 2A - INDEX CASE: Clinical/Demographic Form.IC_DMDX and IS_AGE among index cases",
        context,
    )

    assert ("Form 1A - Index Case Screening", "IS_AGE") in {
        (column["table"], column["column"]) for column in selection.columns
    }
    assert ("Form 2A - INDEX CASE: Clinical/Demographic Form", "IC_DMDX") in {
        (column["table"], column["column"]) for column in selection.columns
    }
    assert "Form 2A - INDEX CASE: Clinical/Demographic Form" in selection.tables


def test_prepare_column_selection_includes_previous_selection_in_prompt(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    from db_rag import service

    captured: list[list[object]] = []

    class _LLM:
        def invoke(self, messages):
            captured.append(messages)
            return SimpleNamespace(
                content=json.dumps(
                    {
                        "selection_id": "sel-prev",
                        "rationale": "preserved reviewed columns",
                        "tables": ["Form 1A"],
                        "columns": [
                            {"table": "Form 1A", "column": "AGE", "description": "Age in years"},
                        ],
                    }
                )
            )

    context = service.DbRagContext(
        tables=[service.DbRagTableHit(table="Form 1A", text="Form 1A summary")],
        columns=[service.DbRagColumnHit(table="Form 1A", column="AGE", text="AGE summary")],
    )
    db_rag_service = service.DbRagService(llm=_LLM())

    previous_selection = {
        "selection_id": "sel-old",
        "question": "subset age",
        "tables": ["Form 1A"],
        "columns": [{"table": "Form 1A", "column": "AGE", "description": "Age in years"}],
        "rationale": "previous reviewed candidate",
        "status": "needs_revision",
    }
    selection = db_rag_service.prepare_column_selection(
        "subset age",
        context,
        feedback_history=[{"feedback": "keep AGE"}],
        previous_selection=previous_selection,
    )

    assert selection.selection_id == "sel-prev"
    message_text = "\n".join(getattr(message, "content", "") for message in captured[0])
    assert "Previous selection candidate:" in message_text
    assert "sel-old" in message_text
    assert '"column": "AGE"' in message_text


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


def test_execute_sql_flow_debug_returns_sql_preparation_details(monkeypatch) -> None:
    _install_langchain_message_stubs(monkeypatch)

    from db_rag import service

    context = service.DbRagContext(
        tables=[service.DbRagTableHit(table="Form 1A", text="Form 1A summary")],
        columns=[service.DbRagColumnHit(table="Form 1A", column="AGE", text="AGE summary")],
    )

    db_rag_service = service.DbRagService(llm=object())
    monkeypatch.setattr(
        db_rag_service,
        "retrieve_context",
        lambda question, debug=False, reranker_model="none": context,
    )
    monkeypatch.setattr(
        db_rag_service,
        "prepare_sql_candidate",
        lambda question, approved_selection: service.PreparedSqlCandidate(
            question=question,
            sql='SELECT "AGE" FROM "Form 1A"',
            tables=["Form 1A"],
            columns=[{"table": "Form 1A", "column": "AGE", "description": "Age in years"}],
            selection_id="sel-1",
        ),
    )
    monkeypatch.setattr(
        db_rag_service,
        "execute_prepared_sql",
        lambda candidate: service.SqlExecutionResult(
            answer="Read-only SQL execution completed with 1 result row(s).",
            sql=candidate.sql,
            dataframe=None,
            source_tables=candidate.tables,
        ),
    )

    result = db_rag_service.execute_sql_flow("subset age", debug=True)

    assert result["debug"]["question"] == "subset age"
    assert result["debug"]["retrieved_tables"] == ["Form 1A"]
    assert result["debug"]["retrieved_columns"] == ["Form 1A.AGE"]
    assert result["debug"]["sql_tables"] == ["Form 1A"]
    assert result["debug"]["sql_columns"] == ["Form 1A.AGE"]
