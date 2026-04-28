from __future__ import annotations

import sys
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType, SimpleNamespace
from typing import TypeVar

import pandas as pd
from langchain_core.messages import AIMessage as _RealAIMessage
from langchain_core.messages import HumanMessage as _RealHumanMessage

for module_name in (
    "langchain_core",
    "langchain_core.messages",
    "langchain_core.prompts",
    "langgraph.graph.message",
):
    sys.modules.pop(module_name, None)


class _HumanMessage:
    def __new__(cls, content: str):
        return _RealHumanMessage(content=content)


class _AIMessage:
    def __new__(cls, content: str, additional_kwargs: dict | None = None):
        return _RealAIMessage(content=content, additional_kwargs=additional_kwargs or {})


class _MessagesPlaceholder:
    def __init__(self, variable_name: str, optional: bool = False) -> None:
        self.variable_name = variable_name
        self.optional = optional


class _PromptTemplate:
    @staticmethod
    def from_messages(messages):
        return _PromptTemplateInstance(messages)


class _PromptTemplateInstance:
    def __init__(self, messages):
        self.messages = messages

    def format_prompt(self, **_kwargs):
        return self

    def to_messages(self):
        return self.messages


class _PromptLikeStub:
    def __init__(self, *_args, **_kwargs):
        pass

    @classmethod
    def from_messages(cls, *_args, **_kwargs):
        return cls()

    @classmethod
    def from_examples(cls, *_args, **_kwargs):
        return cls()


class _GenericStub:
    def __class_getitem__(cls, _item):
        return cls


class _RunnableBaseStub:
    def __class_getitem__(cls, _item):
        return cls

    def get_name(self, suffix: str | None = None, *, name: str | None = None) -> str:
        base_name = name or getattr(self, "name", None) or self.__class__.__name__
        if suffix:
            if base_name and base_name[0].isupper():
                return base_name + suffix.title()
            return f"{base_name}_{suffix.lower()}"
        return base_name

    def get_graph(self, config=None):
        return SimpleNamespace(nodes=[], edges=[])

    def with_config(self, **_kwargs):
        return self


class _RunnableLambdaStub:
    def __class_getitem__(cls, _item):
        return cls


class _RunnableParallelStub:
    def __class_getitem__(cls, _item):
        return cls


class _RunnableSequenceStub:
    def __class_getitem__(cls, _item):
        return cls


class _ContextVarStub:
    def __init__(self):
        self._value = None

    def get(self, default=None):
        return self._value if self._value is not None else default

    def set(self, value):
        old = self._value
        self._value = value
        return old

    def reset(self, token):
        self._value = token


class _CallbackHandlerStub:
    pass


class _StreamingCallbackHandlerStub:
    pass


class _CallbackManagerStub:
    def __init__(self):
        self.handlers = []
        self.run_id = "stub-run-id"

    def on_chain_start(self, *args, **kwargs):
        return self

    def on_chain_end(self, *args, **kwargs):
        return self

    def on_chain_error(self, *args, **kwargs):
        return self

    def on_llm_start(self, *args, **kwargs):
        return self

    def on_llm_end(self, *args, **kwargs):
        return self

    def on_llm_error(self, *args, **kwargs):
        return self

    def on_chat_model_start(self, *args, **kwargs):
        return self

    def on_text(self, *args, **kwargs):
        return None

    def get_child(self, *args, **kwargs):
        return self

    def add_tags(self, *args, **kwargs):
        return None

    def add_metadata(self, *args, **kwargs):
        return None

    def copy(self):
        return self

    def merge(self, _other):
        return self

    def add_handler(self, *args, **kwargs):
        return None


def _pkg(name: str) -> ModuleType:
    module = ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _mod(name: str) -> ModuleType:
    return ModuleType(name)


langchain_core_mod = _pkg("langchain_core")

messages_mod = _mod("langchain_core.messages")
messages_mod.BaseMessage = object
messages_mod.HumanMessage = _HumanMessage
messages_mod.AIMessage = _AIMessage
messages_mod.AnyMessage = object

messages_tool_mod = _mod("langchain_core.messages.tool")
messages_tool_mod.ToolOutputMixin = object

prompts_mod = _mod("langchain_core.prompts")
prompts_mod.ChatPromptTemplate = _PromptTemplate
prompts_mod.MessagesPlaceholder = _MessagesPlaceholder
prompts_mod.__getattr__ = lambda _name: _PromptLikeStub

runnables_pkg = _pkg("langchain_core.runnables")
runnables_base_mod = _mod("langchain_core.runnables.base")
runnables_base_mod.Runnable = _RunnableBaseStub
runnables_base_mod.RunnableConfig = dict
runnables_base_mod.RunnableLambda = _RunnableLambdaStub
runnables_base_mod.RunnableParallel = _RunnableParallelStub
runnables_base_mod.RunnableSequence = _RunnableSequenceStub
runnables_base_mod.RunnableLike = _RunnableBaseStub
runnables_base_mod.Input = TypeVar("RunnableInput")
runnables_base_mod.Output = TypeVar("RunnableOutput")
runnables_config_mod = _mod("langchain_core.runnables.config")
runnables_config_mod.run_in_executor = lambda *args, **kwargs: None
runnables_config_mod.var_child_runnable_config = _ContextVarStub()
runnables_config_mod.CONFIG_KEYS = ()
runnables_config_mod.COPIABLE_KEYS = ()
runnables_config_mod.RunnableConfig = dict
runnables_config_mod.get_executor_for_config = lambda *args, **kwargs: ThreadPoolExecutor(max_workers=1)
runnables_config_mod.get_async_callback_manager_for_config = lambda *args, **kwargs: _CallbackManagerStub()
runnables_config_mod.get_callback_manager_for_config = lambda *args, **kwargs: _CallbackManagerStub()
runnables_config_mod.patch_config = (
    lambda config, **kwargs: {
        **(config.copy() if config is not None else {}),
        **(
            {"configurable": {**(config.get("configurable", {}) if config else {}), **kwargs["configurable"]}}
            if kwargs.get("configurable") is not None
            else {}
        ),
        **{k: v for k, v in kwargs.items() if k != "configurable" and v is not None},
    }
)
runnables_config_mod.ensure_config = lambda base, config=None: {**(base or {}), **(config or {})}
runnables_utils_mod = _mod("langchain_core.runnables.utils")
runnables_utils_mod.Input = TypeVar("Input")
runnables_utils_mod.Output = TypeVar("Output")
runnables_graph_mod = _mod("langchain_core.runnables.graph")
runnables_graph_mod.Graph = object
runnables_graph_mod.Node = object
runnables_pkg.base = runnables_base_mod
runnables_pkg.config = runnables_config_mod
runnables_pkg.utils = runnables_utils_mod
runnables_pkg.graph = runnables_graph_mod
runnables_pkg.Runnable = runnables_base_mod.Runnable
runnables_pkg.RunnableConfig = runnables_base_mod.RunnableConfig
runnables_pkg.RunnableLambda = runnables_base_mod.RunnableLambda
runnables_pkg.RunnableParallel = runnables_base_mod.RunnableParallel
runnables_pkg.RunnableSequence = runnables_base_mod.RunnableSequence
runnables_pkg.Input = runnables_base_mod.Input
runnables_pkg.Output = runnables_base_mod.Output
runnables_pkg.__getattr__ = lambda _name: _GenericStub

callbacks_pkg = _pkg("langchain_core.callbacks")
callbacks_mod = _mod("langchain_core.callbacks")
callbacks_mod.AsyncCallbackManager = _CallbackManagerStub
callbacks_mod.BaseCallbackManager = _CallbackManagerStub
callbacks_mod.CallbackManager = _CallbackManagerStub
callbacks_mod.Callbacks = _GenericStub
callbacks_mod.BaseCallbackHandler = _CallbackHandlerStub
callbacks_mod.AsyncParentRunManager = _CallbackManagerStub
callbacks_mod.ParentRunManager = _CallbackManagerStub
callbacks_manager_mod = _mod("langchain_core.callbacks.manager")
callbacks_manager_mod.AsyncCallbackManager = _CallbackManagerStub
callbacks_manager_mod.CallbackManager = _CallbackManagerStub
callbacks_manager_mod.AsyncParentRunManager = _CallbackManagerStub
callbacks_manager_mod.ParentRunManager = _CallbackManagerStub
callbacks_manager_mod.__getattr__ = lambda _name: _GenericStub
callbacks_pkg.AsyncCallbackManager = callbacks_mod.AsyncCallbackManager
callbacks_pkg.BaseCallbackManager = callbacks_mod.BaseCallbackManager
callbacks_pkg.CallbackManager = callbacks_mod.CallbackManager
callbacks_pkg.Callbacks = callbacks_mod.Callbacks
callbacks_pkg.BaseCallbackHandler = callbacks_mod.BaseCallbackHandler
callbacks_pkg.AsyncParentRunManager = callbacks_mod.AsyncParentRunManager
callbacks_pkg.ParentRunManager = callbacks_mod.ParentRunManager

embeddings_mod = _mod("langchain_core.embeddings")
embeddings_mod.Embeddings = object

load_pkg = _pkg("langchain_core.load")
load_load_mod = _mod("langchain_core.load.load")
load_load_mod.Reviver = object
load_pkg.load = load_load_mod

outputs_mod = _mod("langchain_core.outputs")
outputs_mod.ChatGeneration = object
outputs_mod.ChatGenerationChunk = object
outputs_mod.LLMResult = object

tools_pkg = _pkg("langchain_core.tools")
tools_mod = _mod("langchain_core.tools")
tools_mod.BaseTool = object
tools_mod.InjectedToolArg = object
tools_mod.tool = lambda *args, **kwargs: (args[0] if args else None)
tools_base_mod = _mod("langchain_core.tools.base")
tools_base_mod.BaseTool = object
tools_base_mod.InjectedToolArg = object

language_models_mod = _mod("langchain_core.language_models")
language_models_mod.BaseChatModel = object
language_models_mod.LLM = object

globals_mod = _mod("langchain_core.globals")
globals_mod.get_debug = lambda: False

tracers_pkg = _pkg("langchain_core.tracers")
tracers_langchain_mod = _mod("langchain_core.tracers.langchain")
tracers_langchain_mod.LangChainTracer = _GenericStub
tracers_streaming_mod = _mod("langchain_core.tracers._streaming")
tracers_streaming_mod._StreamingCallbackHandler = _StreamingCallbackHandlerStub

utils_pkg = _pkg("langchain_core.utils")
pydantic_mod = _mod("langchain_core.utils.pydantic")
pydantic_mod.is_basemodel_subclass = lambda *_args, **_kwargs: False

graph_message_mod = _mod("langgraph.graph.message")
graph_message_mod.MessageGraph = object
graph_message_mod.MessagesState = object
graph_message_mod.add_messages = lambda current, new: (current or []) + (new or [])

langchain_core_mod.messages = messages_mod
langchain_core_mod.prompts = prompts_mod
langchain_core_mod.runnables = runnables_pkg
langchain_core_mod.embeddings = embeddings_mod
langchain_core_mod.load = load_pkg
langchain_core_mod.callbacks = callbacks_mod
langchain_core_mod.outputs = outputs_mod
langchain_core_mod.tools = tools_mod
langchain_core_mod.language_models = language_models_mod
langchain_core_mod.globals = globals_mod
langchain_core_mod.tracers = tracers_pkg
langchain_core_mod.utils = utils_pkg

sys.modules["langchain_core"] = langchain_core_mod
sys.modules["langchain_core.messages"] = messages_mod
sys.modules["langchain_core.messages.tool"] = messages_tool_mod
sys.modules["langchain_core.prompts"] = prompts_mod
sys.modules["langchain_core.runnables"] = runnables_pkg
sys.modules["langchain_core.runnables.base"] = runnables_base_mod
sys.modules["langchain_core.runnables.config"] = runnables_config_mod
sys.modules["langchain_core.runnables.utils"] = runnables_utils_mod
sys.modules["langchain_core.runnables.graph"] = runnables_graph_mod
sys.modules["langchain_core.embeddings"] = embeddings_mod
sys.modules["langchain_core.load"] = load_pkg
sys.modules["langchain_core.load.load"] = load_load_mod
sys.modules["langchain_core.callbacks"] = callbacks_mod
sys.modules["langchain_core.callbacks.manager"] = callbacks_manager_mod
sys.modules["langchain_core.outputs"] = outputs_mod
sys.modules["langchain_core.tools"] = tools_mod
sys.modules["langchain_core.tools.base"] = tools_base_mod
sys.modules["langchain_core.language_models"] = language_models_mod
sys.modules["langchain_core.globals"] = globals_mod
sys.modules["langchain_core.tracers"] = tracers_pkg
sys.modules["langchain_core.tracers.langchain"] = tracers_langchain_mod
sys.modules["langchain_core.tracers._streaming"] = tracers_streaming_mod
sys.modules["langchain_core.utils"] = utils_pkg
sys.modules["langchain_core.utils.pydantic"] = pydantic_mod
sys.modules["langgraph.graph.message"] = graph_message_mod

from langgraph.types import Command

from db_rag.service import DbRagService
from graph.nodes.human_review_rag_db_column_selection import (
    human_review_rag_db_column_selection_node,
)
from graph.nodes.human_review_rag_db_sql_execution import (
    human_review_rag_db_sql_execution_node,
)
from graph.nodes.rag_db_qa import rag_db_qa_node


HumanMessage = _HumanMessage


class _SeqLLM:
    def __init__(self, contents: list[str]):
        self._contents = list(contents)
        self.calls: list[object] = []

    def invoke(self, messages):
        self.calls.append(messages)
        if not self._contents:
            return SimpleNamespace(content='{"action":"end","thought":"done"}')
        if len(self._contents) == 1:
            return SimpleNamespace(content=self._contents[0])
        return SimpleNamespace(content=self._contents.pop(0))


def _drain_until_interrupt_or_done(app, config):
    for _ in range(20):
        snapshot = app.get_state(config)
        if not snapshot or not snapshot.next or snapshot.interrupts:
            return snapshot
        app.invoke({}, config=config)
    raise AssertionError("graph did not reach an interrupt or terminal state in time")


def test_db_rag_full_review_workflow_reaches_sql_execution_and_persists_subset(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(DbRagService, "readiness", lambda self: {"ready": True, "message": ""})

    def _retrieve_context(self, question, *, reranker_model=None):
        assert "subset age and sex" in question.lower()
        return SimpleNamespace(
            tables=[SimpleNamespace(table="Form 1A", text="table summary")],
            columns=[
                SimpleNamespace(table="Form 1A", column="AGE", text="age summary"),
                SimpleNamespace(table="Form 1A", column="SEX", text="sex summary"),
            ],
            table_names=["Form 1A"],
            column_names=["AGE", "SEX"],
        )

    def _answer_from_context(self, question, context):
        assert "subset age and sex" in question.lower()
        assert context.table_names == ["Form 1A"]
        return SimpleNamespace(
            answer="This request needs row-level SQL to produce the exact subset.",
            needs_sql=True,
            rationale="subset request",
            relevant_tables=["Form 1A"],
            relevant_columns=["AGE", "SEX"],
        )

    def _prepare_column_selection(self, question, context, feedback_history=None, previous_selection=None):
        assert previous_selection is None
        assert feedback_history == []
        return SimpleNamespace(
            selection_id="sel-1",
            question=question,
            tables=["Form 1A"],
            columns=[
                {"table": "Form 1A", "column": "AGE", "description": "Age in years"},
                {"table": "Form 1A", "column": "SEX", "description": "Sex at enrollment"},
            ],
            rationale="Need demographic columns for the requested subset.",
            feedback_history=[],
            status="awaiting_review",
        )

    def _prepare_sql_candidate(self, question, approved_selection):
        assert approved_selection.status == "approved"
        assert approved_selection.selection_id == "sel-1"
        return SimpleNamespace(
            question=question,
            sql='SELECT "AGE", "SEX" FROM "Form 1A"',
            tables=["Form 1A"],
            columns=[
                {"table": "Form 1A", "column": "AGE", "description": "Age in years"},
                {"table": "Form 1A", "column": "SEX", "description": "Sex at enrollment"},
            ],
            selection_id="sel-1",
            status="prepared",
        )

    def _execute_prepared_sql(self, candidate):
        assert candidate.selection_id == "sel-1"
        return SimpleNamespace(
            answer="Read-only SQL execution completed with 1 result row(s).",
            sql=candidate.sql,
            dataframe=pd.DataFrame({"AGE": [42], "SEX": ["Male"]}),
            source_tables=["Form 1A"],
        )

    monkeypatch.setattr(DbRagService, "retrieve_context", _retrieve_context)
    monkeypatch.setattr(DbRagService, "answer_from_context", _answer_from_context)
    monkeypatch.setattr(DbRagService, "prepare_column_selection", _prepare_column_selection)
    monkeypatch.setattr(DbRagService, "prepare_sql_candidate", _prepare_sql_candidate)
    monkeypatch.setattr(DbRagService, "execute_prepared_sql", _execute_prepared_sql)
    import graph.nodes.human_review_rag_db_column_selection as column_review_mod
    import graph.nodes.human_review_rag_db_sql_execution as sql_review_mod

    initial_state = {
        "messages": [HumanMessage(content="Help me subset age and sex for the matching participants")],
        "output": {},
        "observations": [],
        "meta": {"thread_id": "thread-db-rag-e2e"},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {"before_run_decision": None, "after_error_decision": None, "final_decision": None},
            "rag_db_qa": {},
            "qa": {},
            "generate_code": {},
        },
        "artifacts": {"datasets": {}},
        "planner": {},
        "orchestrator": {},
        "node_data": {},
        "next_action": None,
        "last_action": None,
    }

    monkeypatch.setattr(column_review_mod, "interrupt", lambda _payload: {"action": "approve"})
    monkeypatch.setattr(sql_review_mod, "interrupt", lambda _payload: {"action": "approve"})

    state = rag_db_qa_node(
        initial_state,
        _SeqLLM(['{"action":"rag_db_qa","thought":"database question"}']),
        provider="openai",
        service=DbRagService(str(tmp_path / "db.sqlite")),
        reranker_model=None,
    )

    assert state["agents"]["rag_db_qa"]["pending_column_review"]["status"] == "awaiting_review"
    assert "SQL will be generated only after approval" in state["output"]["qa_response"]

    state = human_review_rag_db_column_selection_node(state)
    assert state["agents"]["rag_db_qa"]["pending_column_review"]["status"] == "approved"

    state = rag_db_qa_node(
        state,
        _SeqLLM(['{"action":"rag_db_qa","thought":"database question"}']),
        provider="openai",
        service=DbRagService(str(tmp_path / "db.sqlite")),
        reranker_model=None,
    )

    assert state["agents"]["rag_db_qa"]["pending_sql_candidate"]["status"] == "prepared"
    assert 'SELECT "AGE", "SEX" FROM "Form 1A"' in state["output"]["generated_sql"]

    final_state = human_review_rag_db_sql_execution_node(state, DbRagService(str(tmp_path / "db.sqlite")))

    assert "Read-only SQL execution completed with 1 result row" in final_state["output"]["qa_response"]
    assert final_state["output"]["generated_sql"] == 'SELECT "AGE", "SEX" FROM "Form 1A"'
    assert final_state["agents"]["rag_db_qa"]["status"] == "done"
    dataset_id = final_state["artifacts"]["active_dataset_id"]
    artifact = final_state["artifacts"]["datasets"][dataset_id]
    assert artifact["kind"] == "subset"
    assert artifact["provenance"]["source"] == "db_rag_sql"
    assert artifact["provenance"]["selected_columns"][0]["column"] == "AGE"
