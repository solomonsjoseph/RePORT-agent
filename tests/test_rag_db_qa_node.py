from __future__ import annotations

import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _ensure_langchain_core_stubs() -> None:
    langchain_core = sys.modules.get("langchain_core")
    if langchain_core is None:
        langchain_core = ModuleType("langchain_core")

    messages = sys.modules.get("langchain_core.messages")
    if messages is None:
        messages = getattr(langchain_core, "messages", None)
    if messages is None:
        messages = ModuleType("langchain_core.messages")

    prompts = sys.modules.get("langchain_core.prompts")
    if prompts is None:
        prompts = getattr(langchain_core, "prompts", None)
    if prompts is None:
        prompts = ModuleType("langchain_core.prompts")

    if not hasattr(messages, "BaseMessage"):
        class BaseMessage:
            type = "base"

            def __init__(self, content: str, additional_kwargs: dict | None = None) -> None:
                self.content = content
                self.additional_kwargs = dict(additional_kwargs or {})

        messages.BaseMessage = BaseMessage

    if not hasattr(messages, "AIMessage"):
        class AIMessage(messages.BaseMessage):
            type = "ai"

        messages.AIMessage = AIMessage

    if not hasattr(messages, "HumanMessage"):
        class HumanMessage(messages.BaseMessage):
            type = "human"

        messages.HumanMessage = HumanMessage

    if not hasattr(messages, "SystemMessage"):
        class SystemMessage(messages.BaseMessage):
            type = "system"

        messages.SystemMessage = SystemMessage

    if not hasattr(prompts, "ChatPromptTemplate"):
        prompts.ChatPromptTemplate = type(
            "ChatPromptTemplate",
            (),
            {"from_messages": staticmethod(lambda *_args, **_kwargs: None)},
        )
    if not hasattr(prompts, "MessagesPlaceholder"):
        prompts.MessagesPlaceholder = type(
            "MessagesPlaceholder",
            (),
            {"__init__": lambda self, *_args, **_kwargs: None},
        )

    langchain_core.messages = messages
    langchain_core.prompts = prompts
    sys.modules["langchain_core"] = langchain_core
    sys.modules["langchain_core.messages"] = messages
    sys.modules["langchain_core.prompts"] = prompts


_ensure_langchain_core_stubs()

from db_rag.service.errors import DbRagUnanswerableError
from graph.nodes.db_rag_qa.helpers import _render_db_rag_recent_turns
from graph.nodes.db_rag_qa.node import rag_db_qa_node

_EXTRACTION_PROMPT = (
    "Would you like me to identify the tables and columns needed for a data extraction "
    "from this database question?"
)


class _Service:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.extraction_gate_result: dict[str, object] | None = None
        self.pending_reply_result: dict[str, object] | None = None
        self.sql_error_message: str | None = None
        self.answer_needs_sql: bool = False
        self.resolved_goal_text: str | None = None
        self.population_scope_result: dict[str, object] | None = None

    def readiness(self):
        return {"ready": True}

    def retrieve_context(self, question: str, reranker_model: str | None = None):
        del reranker_model
        self.calls.append(("retrieve_context", question))
        return SimpleNamespace(
            table_names=["Form 2A"],
            column_names=["IC_AGE"],
            table_context="Form 2A",
            column_context="IC_AGE",
            tables=[SimpleNamespace(table="Form 2A", text="Index case demographics")],
            columns=[SimpleNamespace(table="Form 2A", column="IC_AGE", text="Age in years")],
        )

    def answer_from_context(self, question: str, context):
        del context
        self.calls.append(("answer_from_context", question))
        return SimpleNamespace(answer=f"Metadata answer for: {question}", needs_sql=self.answer_needs_sql)

    def resolve_intent(self, question: str, context, prior_intent=None):
        del context, prior_intent
        return SimpleNamespace(
            intent_id=f"intent:{question}",
            source_question=question,
            goal_text=self.resolved_goal_text or question,
            mode="metadata",
            population=None,
            filters=[],
            required_tables=[],
            required_columns=[],
            excluded_tables=[],
            excluded_columns=[],
            feedback_history=[],
            status="active",
        )

    def prepare_column_selection(
        self,
        question: str,
        context,
        feedback_history=None,
        previous_selection=None,
        intent_snapshot=None,
    ):
        del context, previous_selection, intent_snapshot
        self.calls.append(("prepare_column_selection", question))
        return SimpleNamespace(
            selection_id="sel-1",
            question=question,
            tables=["Form 2A"],
            columns=[{"table": "Form 2A", "column": "IC_AGE", "description": "Age in years"}],
            rationale="Needed for extraction",
            feedback_history=list(feedback_history or []),
            status="awaiting_review",
        )

    def prepare_sql_candidate(self, question: str, selection):
        self.calls.append(("prepare_sql_candidate", question))
        if self.sql_error_message:
            raise RuntimeError(self.sql_error_message)
        return SimpleNamespace(
            question=question,
            sql="select IC_AGE from \"Form 2A\"",
            tables=list(getattr(selection, "tables", [])),
            columns=list(getattr(selection, "columns", [])),
            selection_id=str(getattr(selection, "selection_id", "")),
            status="prepared",
        )

    def build_recoverable_error_clarification(self, *, workflow: str, error_payload, context):
        del error_payload, context
        self.calls.append(("build_recoverable_error_clarification", workflow))
        return {
            "clarification_question": "Please clarify how to revise the DB-RAG selection so SQL can be generated."
        }

    def classify_extraction_gate_message(
        self,
        *,
        pending_prompt: str,
        user_message: str,
        recent_transcript: str,
        active_intent: dict[str, object] | None,
    ):
        del pending_prompt, recent_transcript, active_intent
        self.calls.append(("classify_extraction_gate_message", user_message))
        return dict(self.extraction_gate_result or {"label": "unknown", "confidence": 0.0})

    def classify_pending_reply(
        self,
        *,
        pending_kind: str,
        pending_question: str,
        user_reply: str,
        recent_transcript: str,
    ):
        del pending_kind, pending_question, recent_transcript
        self.calls.append(("classify_pending_reply", user_reply))
        return dict(self.pending_reply_result or {"label": "unknown", "confidence": 0.0})

    def classify_population_scope(
        self,
        *,
        question: str,
        context,
        active_intent: dict[str, object] | None = None,
        intent_snapshot: dict[str, object] | None = None,
        referenced_artifacts: dict[str, object] | None = None,
    ):
        del context, active_intent, intent_snapshot, referenced_artifacts
        self.calls.append(("classify_population_scope", question))
        return dict(self.population_scope_result or {"label": "unknown", "confidence": 0.0})


class _ServiceWithoutPendingClassifiers(_Service):
    def __getattribute__(self, name: str):
        if name in {"classify_extraction_gate_message", "classify_pending_reply"}:
            raise AttributeError(name)
        return super().__getattribute__(name)


def _state(message: str) -> dict:
    human_message = type(
        "HumanMessage",
        (),
        {
            "type": "human",
            "__init__": lambda self, content: setattr(self, "content", content),
        },
    )
    return {
        "messages": [human_message(message)],
        "output": {},
        "observations": [],
        "agents": {"rag_db_qa": {}},
        "meta": {"last_user_message_hash": "u1"},
        "artifacts": {
            "conversation_events": [],
            "conversation_events_version": 1,
            "artifact_manifest_version": 1,
            "files": {},
        },
    }


def _expected_extraction_opt_in(*, intent_id: str, goal_text: str) -> dict[str, str]:
    prompt = f'For your request: "{goal_text}"\n\n{_EXTRACTION_PROMPT}'
    return {
        "status": "awaiting_reply",
        "prompt": prompt,
        "intent_id": intent_id,
        "goal_text": goal_text,
    }


def _memory_with_db_rag_user_intent(
    *,
    intent_id: str = "user-intent-1",
    active_intent_id: str = "intent:age",
    source_question: str = "Which forms contain age?",
    goal_text: str = "Which forms contain age?",
    status: str = "awaiting_extraction_opt_in",
) -> dict:
    return {
        "completed_tasks": {},
        "failed_tasks": {},
        "task_order": [],
        "last_task_id": None,
        "last_task_id_by_kind": {},
        "last_failed_task_id_by_kind": {},
        "last_reference_resolution": None,
        "pending_reference_clarification": None,
        "user_intents": {
            intent_id: {
                "intent_id": intent_id,
                "display_ordinal": 1,
                "kind": "db_rag_query",
                "agent": "rag_db_qa",
                "source_question": source_question,
                "goal_text": goal_text,
                "status": status,
                "source_message_hash": "u0",
                "active_intent_id": active_intent_id,
                "completed_task_id": None,
                "continued_from_intent_id": None,
                "created_at": "2026-05-08T00:00:00+00:00",
                "updated_at": "2026-05-08T00:00:00+00:00",
            }
        },
        "intent_order": [intent_id],
        "last_user_intent_id": intent_id,
        "last_user_intent_id_by_kind": {"db_rag_query": intent_id},
    }


def _state_with_pending_extraction_reply(message: str) -> dict:
    state = _state(message)
    state["memory"] = _memory_with_db_rag_user_intent()
    state["agents"]["rag_db_qa"] = {
        "active_intent": {
            "intent_id": "intent:age",
            "goal_text": "Which forms contain age?",
            "source_question": "Which forms contain age?",
            "mode": "metadata",
            "status": "active",
        },
        "pending_extraction_opt_in": _expected_extraction_opt_in(
            intent_id="intent:age",
            goal_text="Which forms contain age?",
        ),
    }
    return state


def _patch_successful_subset_persistence(monkeypatch, *, dataset_id: str = "dataset-art-1") -> None:
    from graph.nodes.db_rag_qa import helpers as rag_helpers

    def _fake_persist_dataset_artifact(
        *,
        runtime_root,
        thread_id,
        dataset_id,
        kind,
        dataframe,
        schema,
        provenance,
    ):
        del runtime_root, thread_id, dataframe, schema
        return {
            "id": dataset_id,
            "kind": kind,
            "created_at": "2026-05-01T00:00:00+00:00",
            "row_count": 2,
            "column_count": 1,
            "columns": ["IC_AGE"],
            "provenance": dict(provenance or {}),
        }

    monkeypatch.setattr(rag_helpers, "_build_subset_dataset_id", lambda: dataset_id)
    monkeypatch.setattr(rag_helpers, "persist_dataset_artifact", _fake_persist_dataset_artifact)


def _successful_sql_execution_state() -> tuple[dict, dict, SimpleNamespace]:
    from graph.state import MetaKeys

    state = _state("continue")
    state["meta"][MetaKeys.THREAD_ID] = "thread-1"
    state["artifacts"]["files"]["sel-art-1"] = {
        "artifact_id": "sel-art-1",
        "created_at": "2026-05-01T00:00:00+00:00",
        "kind": "db_rag_column_selection",
        "producer": "rag_db_qa",
        "mime": "application/json",
        "summary": "Approved selection",
        "content": {
            "selection_id": "sel-1",
            "tables": ["Form 2A"],
            "columns": [{"table": "Form 2A", "column": "IC_AGE", "description": "Age in years"}],
            "feedback_history": [{"action": "approve", "feedback": "looks right"}],
        },
    }
    rag_state = {
        "pending_column_review": {
            "selection_id": "sel-1",
            "tables": ["Form 2A"],
            "columns": [{"table": "Form 2A", "column": "IC_AGE", "description": "Age in years"}],
            "feedback_history": [{"action": "approve", "feedback": "looks right"}],
        },
        "pending_column_review_artifact_id": "sel-art-1",
        "approved_column_selection_artifact_id": "sel-art-1",
        "pending_sql_candidate_artifact_id": "sql-art-2",
        "pending_sql_candidate": {"status": "prepared"},
        "sql_review_approved_artifact_id": "sql-art-2",
    }
    candidate = SimpleNamespace(
        question="Generate the SQL to subset index cases with diabetes.",
        source_question="Generate the SQL to subset index cases with diabetes.",
        goal_text="Generate the SQL to subset index cases with diabetes.",
        sql='select IC_AGE from "Form 2A"',
        tables=["Form 2A"],
        columns=[{"table": "Form 2A", "column": "IC_AGE", "description": "Age in years"}],
        selection_id="sel-1",
        status="prepared",
    )
    return state, rag_state, candidate


def _execute_successful_sql(monkeypatch, state: dict, rag_state: dict, candidate: SimpleNamespace) -> dict:
    import pandas as pd

    from graph.nodes.db_rag_qa import helpers as rag_helpers

    _patch_successful_subset_persistence(monkeypatch)

    class _ExecutionService:
        def execute_prepared_sql(self, prepared_candidate):
            return SimpleNamespace(
                answer="Read-only SQL execution completed.",
                sql=prepared_candidate.sql,
                source_tables=list(prepared_candidate.tables),
                dataframe=pd.DataFrame({"IC_AGE": [34, 35]}),
            )

    return rag_helpers._execute_prepared_sql_candidate(state, rag_state, candidate, _ExecutionService())


def _state_with_completed_sql_task(message: str) -> tuple[dict, str]:
    from graph.memory import complete_task

    state = _state(message)
    state["artifacts"]["files"]["sel-art-1"] = {
        "artifact_id": "sel-art-1",
        "created_at": "2026-05-01T00:00:00+00:00",
        "kind": "db_rag_column_selection",
        "producer": "rag_db_qa",
        "mime": "application/json",
        "summary": "Approved DB-RAG column selection.",
        "content": {
            "selection_id": "sel-1",
            "source_question": "Subset index cases with diabetes.",
            "goal_text": "Subset index cases with diabetes.",
            "intent_snapshot": {"intent_id": "intent:original"},
            "retrieval_summary": {"tables": ["Form 2A"], "columns": ["IC_AGE"]},
            "tables": ["Form 2A"],
            "columns": [{"table": "Form 2A", "column": "IC_AGE", "description": "Age in years"}],
            "rationale": "Needed for extraction",
            "feedback_history": [],
            "status": "approved",
        },
    }
    state["artifacts"]["files"]["sql-art-1"] = {
        "artifact_id": "sql-art-1",
        "created_at": "2026-05-01T00:00:01+00:00",
        "kind": "db_rag_sql_candidate",
        "producer": "rag_db_qa",
        "mime": "application/json",
        "summary": "Prepared read-only SQL candidate for DB-RAG review.",
        "content": {
            "sql_candidate_id": "sql:sel-1",
            "selection_artifact_id": "sel-art-1",
            "source_question": "Subset index cases with diabetes.",
            "goal_text": "Subset index cases with diabetes.",
            "intent_snapshot": {"intent_id": "intent:original"},
            "tables": ["Form 2A"],
            "columns": [{"table": "Form 2A", "column": "IC_AGE", "description": "Age in years"}],
            "sql": 'select IC_AGE from "Form 2A"',
            "status": "prepared",
        },
    }
    state = complete_task(
        state,
        kind="db_rag_sql_extraction",
        source_question="Subset index cases with diabetes.",
        goal_text="Subset index cases with diabetes.",
        label="DB-RAG SQL extraction: diabetes index-case subset",
        summary="Reviewed SQL executed and saved dataset dataset-art-1.",
        artifact_refs={
            "selection_artifact_id": "sel-art-1",
            "sql_candidate_artifact_id": "sql-art-1",
            "dataset_artifact_id": "dataset-art-1",
        },
        provenance={"producer_node": "rag_db_qa", "selection_id": "sel-1"},
    )
    return state, state["memory"]["last_task_id"]


def _set_resolved_sql_meta(
    state: dict,
    task_id: str,
    *,
    relationship: str,
    intended_action: str,
) -> None:
    from graph.state import MetaKeys

    state["meta"][MetaKeys.RESOLVED_TASK_ID] = task_id
    state["meta"][MetaKeys.RESOLVED_TASK_KIND] = "db_rag_sql_extraction"
    state["meta"][MetaKeys.RESOLVED_TASK_RELATIONSHIP] = relationship
    state["meta"][MetaKeys.RESOLVED_TASK_INTENDED_ACTION] = intended_action
    state["meta"][MetaKeys.RESOLVED_TASK_USER_MESSAGE_HASH] = "u1"


def test_langchain_core_stub_bootstrap_reuses_existing_messages_module(monkeypatch) -> None:
    langchain_core = ModuleType("langchain_core")
    messages = ModuleType("langchain_core.messages")
    messages.BaseMessage = object

    monkeypatch.setitem(sys.modules, "langchain_core", langchain_core)
    monkeypatch.setitem(sys.modules, "langchain_core.messages", messages)
    monkeypatch.delitem(sys.modules, "langchain_core.prompts", raising=False)

    _ensure_langchain_core_stubs()

    assert sys.modules["langchain_core"] is langchain_core
    assert sys.modules["langchain_core.messages"] is messages
    assert langchain_core.messages is messages
    assert hasattr(messages, "AIMessage")
    assert "langchain_core.prompts" in sys.modules
    assert langchain_core.prompts is sys.modules["langchain_core.prompts"]


def test_langchain_core_stub_bootstrap_reuses_existing_prompts_module(monkeypatch) -> None:
    langchain_core = ModuleType("langchain_core")
    prompts = ModuleType("langchain_core.prompts")

    monkeypatch.setitem(sys.modules, "langchain_core", langchain_core)
    monkeypatch.setitem(sys.modules, "langchain_core.prompts", prompts)
    monkeypatch.delitem(sys.modules, "langchain_core.messages", raising=False)

    _ensure_langchain_core_stubs()

    assert sys.modules["langchain_core"] is langchain_core
    assert sys.modules["langchain_core.prompts"] is prompts
    assert langchain_core.prompts is prompts
    assert hasattr(prompts, "ChatPromptTemplate")
    assert hasattr(prompts, "MessagesPlaceholder")
    assert "langchain_core.messages" in sys.modules
    assert langchain_core.messages is sys.modules["langchain_core.messages"]


def test_resolved_sql_inspection_answers_from_sql_artifact() -> None:
    from graph.state import MetaKeys

    service = _Service()
    state, task_id = _state_with_completed_sql_task("What SQL did you use?")
    state["meta"][MetaKeys.ANALYSIS_DATASET_ID] = "dataset-art-1"
    _set_resolved_sql_meta(
        state,
        task_id,
        relationship="inspect_artifact",
        intended_action="show_sql",
    )

    updated = rag_db_qa_node(state, llm=None, provider="openai", service=service)

    assert 'select IC_AGE from "Form 2A"' in updated["output"]["qa_response"]
    assert "Subset index cases with diabetes." in updated["output"]["qa_response"]
    assert "Form 2A" in updated["output"]["qa_response"]
    assert "IC_AGE" in updated["output"]["qa_response"]
    assert not any(call[0] == "prepare_column_selection" for call in service.calls)
    assert updated["agents"]["rag_db_qa"]["pending_column_review_artifact_id"] is None
    assert MetaKeys.RESOLVED_TASK_ID not in updated["meta"]
    assert updated["meta"][MetaKeys.ANALYSIS_DATASET_ID] == "dataset-art-1"


def test_resolved_sql_inspection_creates_linked_qa_answer_task() -> None:
    service = _Service()
    state, parent_task_id = _state_with_completed_sql_task("What SQL did you use?")
    _set_resolved_sql_meta(
        state,
        parent_task_id,
        relationship="inspect_artifact",
        intended_action="show_sql",
    )

    updated = rag_db_qa_node(state, llm=None, provider="openai", service=service)

    memory = updated["memory"]
    child_task_id = memory["last_task_id"]
    assert child_task_id != parent_task_id
    child_task = memory["completed_tasks"][child_task_id]
    assert child_task["kind"] == "qa_answer"
    assert child_task["parent_task_id"] == parent_task_id
    assert child_task["relationship_to_parent"] == "inspect_artifact"
    assert child_task["source_question"] == "What SQL did you use?"
    assert child_task["artifact_refs"] == {"sql_candidate_artifact_id": "sql-art-1"}
    assert memory["task_order"] == [parent_task_id, child_task_id]


def test_resolved_sql_revision_opens_new_column_review() -> None:
    service = _Service()
    state, task_id = _state_with_completed_sql_task("Add gender too.")
    _set_resolved_sql_meta(
        state,
        task_id,
        relationship="revision",
        intended_action="add_fields",
    )

    updated = rag_db_qa_node(state, llm=None, provider="openai", service=service)

    rag_state = updated["agents"]["rag_db_qa"]
    assert rag_state["pending_column_review_artifact_id"]
    assert rag_state["thread_status"] == "awaiting_column_review"
    assert any(call[0] == "retrieve_context" and "Add gender too." in call[1] for call in service.calls)
    assert ("prepare_column_selection", "Add gender too.") in service.calls
    assert not any(call[0] == "answer_from_context" for call in service.calls)


def test_resolved_sql_revision_sets_active_task_parent() -> None:
    service = _Service()
    state, parent_task_id = _state_with_completed_sql_task("Add gender too.")
    _set_resolved_sql_meta(
        state,
        parent_task_id,
        relationship="revision",
        intended_action="add_fields",
    )

    updated = rag_db_qa_node(state, llm=None, provider="openai", service=service)

    assert updated["agents"]["rag_db_qa"]["active_task"] == {
        "parent_task_id": parent_task_id,
        "relationship_to_parent": "revision",
    }


def test_resolved_sql_revision_column_selection_artifact_uses_clean_source_and_goal() -> None:
    service = _Service()
    state, task_id = _state_with_completed_sql_task("Add gender too.")
    _set_resolved_sql_meta(
        state,
        task_id,
        relationship="revision",
        intended_action="add_fields",
    )

    updated = rag_db_qa_node(state, llm=None, provider="openai", service=service)

    artifact_id = updated["agents"]["rag_db_qa"]["pending_column_review_artifact_id"]
    artifact_content = updated["artifacts"]["files"][artifact_id]["content"]
    assert artifact_content["source_question"] == "Add gender too."
    assert artifact_content["goal_text"] == "Add gender too."
    artifact_text = json.dumps(artifact_content, sort_keys=True)
    assert "Prior SQL" not in artifact_text
    assert "Parent task summary" not in artifact_text
    assert "select IC_AGE" not in artifact_text


def test_resolved_sql_revision_consumes_resolved_task_meta() -> None:
    from graph.state import MetaKeys

    service = _Service()
    state, task_id = _state_with_completed_sql_task("Add gender too.")
    state["meta"][MetaKeys.ANALYSIS_DATASET_ID] = "dataset-art-1"
    _set_resolved_sql_meta(
        state,
        task_id,
        relationship="revision",
        intended_action="add_fields",
    )

    updated = rag_db_qa_node(state, llm=None, provider="openai", service=service)

    assert MetaKeys.RESOLVED_TASK_ID not in updated["meta"]
    assert MetaKeys.RESOLVED_TASK_KIND not in updated["meta"]
    assert MetaKeys.RESOLVED_TASK_RELATIONSHIP not in updated["meta"]
    assert MetaKeys.RESOLVED_TASK_INTENDED_ACTION not in updated["meta"]
    assert MetaKeys.RESOLVED_TASK_USER_MESSAGE_HASH not in updated["meta"]
    assert updated["meta"][MetaKeys.ANALYSIS_DATASET_ID] == "dataset-art-1"


def test_resolved_sql_inspection_and_revision_clear_private_consumed_marker() -> None:
    service = _Service()
    inspection_state, inspection_task_id = _state_with_completed_sql_task("What SQL did you use?")
    inspection_state["meta"]["resolved_task_meta_consumed"] = "u1"
    _set_resolved_sql_meta(
        inspection_state,
        inspection_task_id,
        relationship="inspect_artifact",
        intended_action="show_sql",
    )

    inspected = rag_db_qa_node(inspection_state, llm=None, provider="openai", service=service)

    revision_service = _Service()
    revision_state, revision_task_id = _state_with_completed_sql_task("Add gender too.")
    revision_state["meta"]["resolved_task_meta_consumed"] = "u1"
    _set_resolved_sql_meta(
        revision_state,
        revision_task_id,
        relationship="revision",
        intended_action="add_fields",
    )

    revised = rag_db_qa_node(revision_state, llm=None, provider="openai", service=revision_service)

    assert "resolved_task_meta_consumed" not in inspected["meta"]
    assert "resolved_task_meta_consumed" not in revised["meta"]


def test_resolved_sql_revision_preserves_unrelated_clarification_meta() -> None:
    from graph.state import MetaKeys

    service = _Service()
    state, task_id = _state_with_completed_sql_task("Add gender too.")
    state["meta"][MetaKeys.ANALYSIS_DATASET_ID] = "dataset-art-1"
    state["meta"][MetaKeys.AWAITING_USER_CLARIFICATION] = True
    state["meta"][MetaKeys.CLARIFICATION_KIND] = "dummy_kind"
    state["meta"][MetaKeys.CLARIFICATION_RETURN_NODE] = "qa"
    state["meta"][MetaKeys.PENDING_QUESTION] = "dummy pending"
    _set_resolved_sql_meta(
        state,
        task_id,
        relationship="revision",
        intended_action="add_fields",
    )

    updated = rag_db_qa_node(state, llm=None, provider="openai", service=service)

    assert updated["meta"][MetaKeys.ANALYSIS_DATASET_ID] == "dataset-art-1"
    assert updated["meta"][MetaKeys.AWAITING_USER_CLARIFICATION] is True
    assert updated["meta"][MetaKeys.CLARIFICATION_KIND] == "dummy_kind"
    assert updated["meta"][MetaKeys.CLARIFICATION_RETURN_NODE] == "qa"
    assert updated["meta"][MetaKeys.PENDING_QUESTION] == "dummy pending"
    assert MetaKeys.RESOLVED_TASK_ID not in updated["meta"]
    assert MetaKeys.RESOLVED_TASK_KIND not in updated["meta"]
    assert MetaKeys.RESOLVED_TASK_RELATIONSHIP not in updated["meta"]
    assert MetaKeys.RESOLVED_TASK_INTENDED_ACTION not in updated["meta"]
    assert MetaKeys.RESOLVED_TASK_USER_MESSAGE_HASH not in updated["meta"]


def test_resolved_sql_use_as_input_received_by_db_rag_is_terminal_not_fresh() -> None:
    service = _Service()
    state, task_id = _state_with_completed_sql_task("Analyze that subset.")
    _set_resolved_sql_meta(
        state,
        task_id,
        relationship="use_as_input",
        intended_action="analyze_dataset",
    )

    updated = rag_db_qa_node(state, llm=None, provider="openai", service=service)

    assert "resolved dataset handoff should be routed to code generation" in updated["output"]["qa_response"]
    assert updated["agents"]["rag_db_qa"]["thread_status"] == "error"
    assert not any(call[0] in {"retrieve_context", "answer_from_context"} for call in service.calls)


def test_resolved_sql_unsupported_relationship_is_terminal_not_fresh() -> None:
    service = _Service()
    state, task_id = _state_with_completed_sql_task("Compare that.")
    _set_resolved_sql_meta(
        state,
        task_id,
        relationship="compare",
        intended_action="compare",
    )

    updated = rag_db_qa_node(state, llm=None, provider="openai", service=service)

    assert "cannot handle the resolved DB-RAG SQL relationship" in updated["output"]["qa_response"]
    assert updated["agents"]["rag_db_qa"]["thread_status"] == "error"
    assert not any(call[0] in {"retrieve_context", "answer_from_context"} for call in service.calls)


def test_resolved_sql_missing_task_is_terminal_not_fresh() -> None:
    service = _Service()
    state = _state("What SQL did you use?")
    _set_resolved_sql_meta(
        state,
        "task_missing",
        relationship="inspect_artifact",
        intended_action="show_sql",
    )

    updated = rag_db_qa_node(state, llm=None, provider="openai", service=service)

    assert "could not find the referenced completed DB-RAG SQL extraction task" in updated["output"]["qa_response"]
    assert updated["agents"]["rag_db_qa"]["thread_status"] == "error"
    assert not any(call[0] in {"retrieve_context", "answer_from_context"} for call in service.calls)


def test_resolved_sql_inspection_missing_sql_artifact_is_terminal_not_fresh() -> None:
    service = _Service()
    state, task_id = _state_with_completed_sql_task("What SQL did you use?")
    del state["artifacts"]["files"]["sql-art-1"]
    _set_resolved_sql_meta(
        state,
        task_id,
        relationship="inspect_artifact",
        intended_action="show_sql",
    )

    updated = rag_db_qa_node(state, llm=None, provider="openai", service=service)

    assert "SQL candidate artifact is missing" in updated["output"]["qa_response"]
    assert updated["agents"]["rag_db_qa"]["thread_status"] == "error"
    assert not any(call[0] in {"retrieve_context", "answer_from_context"} for call in service.calls)


def test_resolved_sql_revision_missing_artifact_is_terminal_not_fresh() -> None:
    service = _Service()
    state, task_id = _state_with_completed_sql_task("Add gender too.")
    del state["artifacts"]["files"]["sel-art-1"]
    _set_resolved_sql_meta(
        state,
        task_id,
        relationship="revision",
        intended_action="add_fields",
    )

    updated = rag_db_qa_node(state, llm=None, provider="openai", service=service)

    assert "selection or SQL artifact is missing" in updated["output"]["qa_response"]
    assert updated["agents"]["rag_db_qa"]["thread_status"] == "error"
    assert not any(call[0] in {"retrieve_context", "answer_from_context"} for call in service.calls)


def test_resolved_sql_inspection_preserves_unrelated_meta_and_clarification_keys() -> None:
    from graph.state import MetaKeys

    service = _Service()
    state, task_id = _state_with_completed_sql_task("What SQL did you use?")
    state["meta"][MetaKeys.ANALYSIS_DATASET_ID] = "dataset-art-1"
    state["meta"][MetaKeys.CLARIFICATION_RETURN_NODE] = "qa"
    state["meta"][MetaKeys.CLARIFICATION_KIND] = "dummy_kind"
    state["meta"][MetaKeys.PENDING_QUESTION] = "dummy pending"
    _set_resolved_sql_meta(
        state,
        task_id,
        relationship="inspect_artifact",
        intended_action="show_sql",
    )

    updated = rag_db_qa_node(state, llm=None, provider="openai", service=service)

    assert updated["meta"][MetaKeys.ANALYSIS_DATASET_ID] == "dataset-art-1"
    assert updated["meta"][MetaKeys.CLARIFICATION_RETURN_NODE] == "qa"
    assert updated["meta"][MetaKeys.CLARIFICATION_KIND] == "dummy_kind"
    assert updated["meta"][MetaKeys.PENDING_QUESTION] == "dummy pending"
    assert MetaKeys.RESOLVED_TASK_ID not in updated["meta"]


def test_metadata_question_always_opens_extraction_prompt_after_answer() -> None:
    service = _Service()

    updated = rag_db_qa_node(_state("What tables contain age?"), llm=None, provider="openai", service=service)

    rag_state = updated["agents"]["rag_db_qa"]
    assert rag_state["active_intent"]["goal_text"] == "What tables contain age?"
    assert rag_state["pending_extraction_opt_in"] == _expected_extraction_opt_in(
        intent_id="intent:What tables contain age?",
        goal_text="What tables contain age?",
    )
    assert "Metadata answer for: What tables contain age?" in updated["output"]["qa_response"]
    assert _EXTRACTION_PROMPT in updated["output"]["qa_response"]


def test_ambiguous_population_metadata_question_warns_before_answer() -> None:
    service = _Service()
    service.population_scope_result = {
        "label": "ambiguous",
        "population": None,
        "confidence": 0.91,
        "reason": "The question asks for gender distribution without specifying index cases or household contacts.",
    }

    updated = rag_db_qa_node(_state("What is the gender distribution?"), llm=None, provider="openai", service=service)

    response = updated["output"]["qa_response"]
    assert response.startswith(
        "Warning: This database contains both index cases and household contacts."
    )
    assert "verify that the answer matches your intended population" in response
    assert "Metadata answer for: What is the gender distribution?" in response
    rag_state = updated["agents"]["rag_db_qa"]
    assert rag_state["population_scope"]["label"] == "ambiguous"
    assert rag_state["population_scope"]["confidence"] == 0.91
    assert ("classify_population_scope", "What is the gender distribution?") in service.calls


def test_specified_population_metadata_question_does_not_warn() -> None:
    service = _Service()
    service.population_scope_result = {
        "label": "specified",
        "population": "index_case",
        "confidence": 0.93,
        "reason": "The question specifies index cases.",
    }

    updated = rag_db_qa_node(
        _state("What is the gender distribution among index cases?"),
        llm=None,
        provider="openai",
        service=service,
    )

    assert not updated["output"]["qa_response"].startswith("Warning:")
    assert "Metadata answer for: What is the gender distribution among index cases?" in updated["output"]["qa_response"]
    assert updated["agents"]["rag_db_qa"]["population_scope"]["label"] == "specified"


def test_metadata_question_ignores_needs_sql_routing_flag() -> None:
    service = _Service()
    service.answer_needs_sql = True

    updated = rag_db_qa_node(_state("What tables contain age?"), llm=None, provider="openai", service=service)

    rag_state = updated["agents"]["rag_db_qa"]
    assert ("answer_from_context", "What tables contain age?") in service.calls
    assert rag_state["pending_column_review_artifact_id"] is None
    assert rag_state["pending_extraction_opt_in"] == _expected_extraction_opt_in(
        intent_id="intent:What tables contain age?",
        goal_text="What tables contain age?",
    )
    assert _EXTRACTION_PROMPT in updated["output"]["qa_response"]


def test_metadata_question_emits_clarification_event_and_no_selection_artifact() -> None:
    service = _Service()

    updated = rag_db_qa_node(_state("What tables contain age?"), llm=None, provider="openai", service=service)

    events = updated["artifacts"]["conversation_events"]
    assert events[-1]["type"] == "clarification"
    assert events[-1]["text"] == _expected_extraction_opt_in(
        intent_id="intent:What tables contain age?",
        goal_text="What tables contain age?",
    )["prompt"]
    assert not any(
        artifact["kind"] == "db_rag_column_selection"
        for artifact in updated["artifacts"]["files"].values()
    )


def test_metadata_question_stores_targeted_clarification_prompt_in_meta() -> None:
    service = _Service()

    updated = rag_db_qa_node(
        _state("query my database, what variables related to tb outcome, hiv status, and diabetes status"),
        llm=None,
        provider="openai",
        service=service,
    )

    expected_prompt = _expected_extraction_opt_in(
        intent_id=(
            "intent:query my database, what variables related to tb outcome, hiv status, and diabetes status"
        ),
        goal_text="query my database, what variables related to tb outcome, hiv status, and diabetes status",
    )["prompt"]
    assert expected_prompt in updated["output"]["qa_response"]
    assert updated["meta"]["pending_question"] == expected_prompt
    assert updated["agents"]["rag_db_qa"]["pending_extraction_opt_in"]["prompt"] == expected_prompt


def test_db_rag_recent_turns_prefers_semantic_events_over_raw_messages() -> None:
    state = _state("yes")
    state["messages"].append(type(state["messages"][0])("stale raw message should not drive routing"))
    state["artifacts"]["conversation_events"] = [
        {
            "event_id": "e1",
            "seq": 1,
            "created_at": "2026-05-01T00:00:00Z",
            "type": "user",
            "actor": "user",
            "actor_role": "user",
            "user_turn_hash": "u1",
            "text": "query my database, what variables related to tb outcome, hiv status, and diabetes status",
        },
        {
            "event_id": "e2",
            "seq": 2,
            "created_at": "2026-05-01T00:00:01Z",
            "type": "clarification",
            "actor": "rag_db_qa",
            "actor_role": "assistant",
            "user_turn_hash": "u1",
            "text": 'For your request: "query my database, what variables related to tb outcome, hiv status, and diabetes status"',
        },
        {
            "event_id": "e3",
            "seq": 3,
            "created_at": "2026-05-01T00:00:02Z",
            "type": "review_decision",
            "actor": "human_review_rag_db_column_selection",
            "actor_role": "review",
            "user_turn_hash": "u1",
            "review_kind": "rag_db_column_selection",
            "decision": "regenerate",
            "text": "Use SUBJID_PSEUDO or FID_PSEUDO as join key.",
        },
    ]

    transcript = _render_db_rag_recent_turns(state, "yes")

    assert "Use SUBJID_PSEUDO or FID_PSEUDO as join key." in transcript
    assert "stale raw message should not drive routing" not in transcript


def test_query_word_only_does_not_force_extraction_flow() -> None:
    service = _Service()

    updated = rag_db_qa_node(
        _state("Query my database: what variables are related to TB outcome, HIV status, and diabetes status?"),
        llm=None,
        provider="openai",
        service=service,
    )

    rag_state = updated["agents"]["rag_db_qa"]
    assert rag_state["pending_extraction_opt_in"] == _expected_extraction_opt_in(
        intent_id=(
            "intent:Query my database: what variables are related to TB outcome, HIV status, and diabetes status?"
        ),
        goal_text=(
            "Query my database: what variables are related to TB outcome, HIV status, and diabetes status?"
        ),
    )
    assert rag_state["pending_column_review_artifact_id"] is None
    assert not any(call[0] == "prepare_column_selection" for call in service.calls)
    assert any(call[0] == "answer_from_context" for call in service.calls)


def test_explicit_extraction_question_opens_column_review_without_opt_in() -> None:
    service = _Service()

    updated = rag_db_qa_node(
        _state("Generate the SQL to subset index cases with diabetes."),
        llm=None,
        provider="openai",
        service=service,
    )

    rag_state = updated["agents"]["rag_db_qa"]
    assert rag_state["pending_extraction_opt_in"] is None
    assert rag_state["pending_column_review_artifact_id"]
    assert ("prepare_column_selection", "Generate the SQL to subset index cases with diabetes.") in service.calls
    assert not any(call[0] == "answer_from_context" for call in service.calls)

    artifact = updated["artifacts"]["files"][rag_state["pending_column_review_artifact_id"]]
    assert artifact["kind"] == "db_rag_column_selection"
    assert artifact["content"]["goal_text"] == "Generate the SQL to subset index cases with diabetes."
    events = updated["artifacts"]["conversation_events"]
    assert events[-1]["type"] == "review_request"
    assert events[-1]["artifact_id"] == rag_state["pending_column_review_artifact_id"]


def test_ambiguous_population_column_review_warns_and_stores_scope() -> None:
    service = _Service()
    service.population_scope_result = {
        "label": "ambiguous",
        "population": None,
        "confidence": 0.89,
        "reason": "The extraction asks for participants without specifying index cases or household contacts.",
    }

    updated = rag_db_qa_node(
        _state("Generate the SQL to subset participants with diabetes."),
        llm=None,
        provider="openai",
        service=service,
    )

    response = updated["output"]["qa_response"]
    assert response.startswith(
        "Warning: This database contains both index cases and household contacts."
    )
    assert "verify that the selected columns match your intended population" in response
    assert "Please review the proposed DB-RAG column selection in the panel below." in response
    rag_state = updated["agents"]["rag_db_qa"]
    artifact = updated["artifacts"]["files"][rag_state["pending_column_review_artifact_id"]]
    assert artifact["content"]["population_scope"]["label"] == "ambiguous"
    assert artifact["content"]["population_scope"]["confidence"] == 0.89
    assert rag_state["pending_column_review"]["population_scope"]["label"] == "ambiguous"


def test_rag_db_node_records_stage_timings() -> None:
    service = _Service()

    updated = rag_db_qa_node(
        _state("Generate the SQL to subset index cases with diabetes."),
        llm=None,
        provider="openai",
        service=service,
    )

    timing = updated["meta"]["db_rag_timing"]
    stages = [record["stage"] for record in timing["stages"]]
    assert timing["node"] == "rag_db_qa"
    assert "rag_db_qa.total" in stages
    assert all(record["elapsed_ms"] >= 0 for record in timing["stages"])


def test_resolved_goal_text_is_preserved_separately_from_source_question() -> None:
    service = _Service()
    service.resolved_goal_text = "Identify index-case age, gender, diabetes status, and TB outcome variables."

    updated = rag_db_qa_node(
        _state("Which variables identify age, gender, diabetes status, and TB outcome among index case?"),
        llm=None,
        provider="openai",
        service=service,
    )

    rag_state = updated["agents"]["rag_db_qa"]
    assert rag_state["active_intent"]["source_question"] == (
        "Which variables identify age, gender, diabetes status, and TB outcome among index case?"
    )
    assert rag_state["active_intent"]["goal_text"] == (
        "Identify index-case age, gender, diabetes status, and TB outcome variables."
    )
    assert rag_state["pending_extraction_opt_in"]["goal_text"] == (
        "Identify index-case age, gender, diabetes status, and TB outcome variables."
    )


def test_fresh_db_rag_question_records_user_intent() -> None:
    service = _Service()

    updated = rag_db_qa_node(
        _state("Which forms contain age?"),
        llm=None,
        provider="openai",
        service=service,
    )

    active_intent = updated["agents"]["rag_db_qa"]["active_intent"]
    memory = updated["memory"]
    intent_id = memory["last_user_intent_id_by_kind"]["db_rag_query"]
    card = memory["user_intents"][intent_id]
    assert card["source_question"] == "Which forms contain age?"
    assert card["kind"] == "db_rag_query"
    assert card["status"] == "awaiting_extraction_opt_in"
    assert card["source_message_hash"] == "u1"
    assert card["active_intent_id"] == active_intent["intent_id"]


def test_explicit_extraction_question_records_awaiting_column_review_user_intent() -> None:
    service = _Service()

    updated = rag_db_qa_node(
        _state("Generate the SQL to subset index cases with diabetes."),
        llm=None,
        provider="openai",
        service=service,
    )

    memory = updated["memory"]
    intent_id = memory["last_user_intent_id_by_kind"]["db_rag_query"]
    card = memory["user_intents"][intent_id]
    assert card["source_question"] == "Generate the SQL to subset index cases with diabetes."
    assert card["status"] == "awaiting_column_review"
    assert card["active_intent_id"] == updated["agents"]["rag_db_qa"]["active_intent"]["intent_id"]


def test_fresh_db_rag_question_creates_new_user_intent_when_live_intent_exists() -> None:
    service = _Service()
    state = _state("Which forms contain gender?")
    state["memory"] = _memory_with_db_rag_user_intent(
        source_question="Which forms contain age?",
        goal_text="Which forms contain age?",
        status="awaiting_extraction_opt_in",
    )
    state["agents"]["rag_db_qa"] = {
        "active_intent": {
            "intent_id": "intent:age",
            "goal_text": "Which forms contain age?",
            "source_question": "Which forms contain age?",
            "mode": "metadata",
            "status": "active",
        }
    }

    updated = rag_db_qa_node(state, llm=None, provider="openai", service=service)

    memory = updated["memory"]
    assert len(memory["intent_order"]) == 2
    first_id, second_id = memory["intent_order"]
    first_card = memory["user_intents"][first_id]
    second_card = memory["user_intents"][second_id]
    assert first_card["source_question"] == "Which forms contain age?"
    assert second_card["source_question"] == "Which forms contain gender?"
    assert second_card["status"] == "awaiting_extraction_opt_in"
    assert memory["last_user_intent_id_by_kind"]["db_rag_query"] == second_id


def test_question_override_records_user_intent_source_question_and_is_consumed() -> None:
    from graph.state import MetaKeys

    service = _Service()
    state = _state("continue the previous query")
    state["meta"][MetaKeys.RAG_DB_QUESTION_OVERRIDE] = "Query my database for age"

    updated = rag_db_qa_node(
        state,
        llm=None,
        provider="openai",
        service=service,
        question_override="Query my database for age",
    )

    active_intent = updated["agents"]["rag_db_qa"]["active_intent"]
    memory = updated["memory"]
    intent_id = memory["last_user_intent_id_by_kind"]["db_rag_query"]
    card = memory["user_intents"][intent_id]
    assert card["source_question"] == "Query my database for age"
    assert card["source_question"] != "continue the previous query"
    assert card["kind"] == "db_rag_query"
    assert card["status"] in {"awaiting_extraction_opt_in", "awaiting_column_review"}
    assert card["active_intent_id"] == active_intent["intent_id"]
    assert MetaKeys.RAG_DB_QUESTION_OVERRIDE not in updated["meta"]


def test_question_override_bypasses_pending_extraction_opt_in_and_creates_new_intent() -> None:
    from graph.state import MetaKeys

    override = "Query my database for age"
    service = _Service()
    state = _state_with_pending_extraction_reply("yes")
    state["meta"][MetaKeys.RAG_DB_QUESTION_OVERRIDE] = override

    updated = rag_db_qa_node(
        state,
        llm=None,
        provider="openai",
        service=service,
        question_override=override,
    )

    rag_state = updated["agents"]["rag_db_qa"]
    memory = updated["memory"]
    assert len(memory["intent_order"]) == 2
    last_intent_id = memory["intent_order"][-1]
    card = memory["user_intents"][last_intent_id]
    assert card["source_question"] == override
    assert card["status"] == "awaiting_extraction_opt_in"
    assert card["active_intent_id"] == rag_state["active_intent"]["intent_id"]
    assert rag_state["pending_extraction_opt_in"]["goal_text"] == override
    assert ("answer_from_context", override) in service.calls
    assert not any(call[0] == "classify_extraction_gate_message" for call in service.calls)
    assert MetaKeys.RAG_DB_QUESTION_OVERRIDE not in updated["meta"]


def test_question_override_bypasses_pending_column_review_pointer_and_creates_new_intent() -> None:
    from graph.state import MetaKeys

    override = "Query my database for age"
    service = _Service()
    state = _state("continue")
    state["meta"][MetaKeys.RAG_DB_QUESTION_OVERRIDE] = override
    state["memory"] = _memory_with_db_rag_user_intent(
        active_intent_id="intent:old",
        source_question="Generate the SQL to subset index cases with diabetes.",
        goal_text="Generate the SQL to subset index cases with diabetes.",
        status="awaiting_column_review",
    )
    state["agents"]["rag_db_qa"] = {
        "active_intent": {
            "intent_id": "intent:old",
            "source_question": "Generate the SQL to subset index cases with diabetes.",
            "goal_text": "Generate the SQL to subset index cases with diabetes.",
            "mode": "extraction",
            "status": "active",
        },
        "pending_column_review_artifact_id": "selection-artifact",
        "pending_column_review": {"status": "awaiting_review"},
    }

    updated = rag_db_qa_node(
        state,
        llm=None,
        provider="openai",
        service=service,
        question_override=override,
    )

    rag_state = updated["agents"]["rag_db_qa"]
    memory = updated["memory"]
    assert len(memory["intent_order"]) == 2
    last_intent_id = memory["intent_order"][-1]
    card = memory["user_intents"][last_intent_id]
    assert card["source_question"] == override
    assert card["status"] == "awaiting_extraction_opt_in"
    assert card["active_intent_id"] == rag_state["active_intent"]["intent_id"]
    assert rag_state["pending_column_review_artifact_id"] is None
    assert rag_state["pending_extraction_opt_in"]["goal_text"] == override
    assert ("answer_from_context", override) in service.calls
    assert "pending column review state is incomplete" not in updated["output"]["qa_response"]
    assert MetaKeys.RAG_DB_QUESTION_OVERRIDE not in updated["meta"]


def test_active_intent_uses_required_columns_without_requested_fields() -> None:
    service = _Service()

    def _resolve_intent(question: str, context, prior_intent=None):
        del context, prior_intent
        return SimpleNamespace(
            intent_id="intent:diabetes",
            source_question=question,
            goal_text="Subset diabetes status.",
            mode="extraction",
            population=None,
            filters=[{"table": "Form 1A", "column": "IS_ENROLL", "operator": "=", "value": "Yes"}],
            required_tables=[],
            required_columns=[
                {"table": "Form 2A", "column": "IC_DMDX", "semantic": "diabetes_status"},
            ],
            excluded_tables=[],
            excluded_columns=[],
            feedback_history=[],
            status="active",
        )

    service.resolve_intent = _resolve_intent

    updated = rag_db_qa_node(
        _state("Subset diabetes status for enrolled index cases."),
        llm=None,
        provider="openai",
        service=service,
    )

    intent = updated["agents"]["rag_db_qa"]["active_intent"]
    assert "requested_fields" not in intent
    assert intent["required_columns"] == [
        {"table": "Form 2A", "column": "IC_DMDX", "semantic": "diabetes_status"},
    ]
    assert intent["filters"] == [
        {"table": "Form 1A", "column": "IS_ENROLL", "operator": "=", "value": "Yes"},
    ]


def test_explicit_extraction_artifact_preserves_distinct_goal_text_and_source_question() -> None:
    service = _Service()
    service.resolved_goal_text = "Subset index-case age, gender, diabetes status, and TB outcome."

    updated = rag_db_qa_node(
        _state("Query my database, Help me to subset age, gender, diabetes status, and TB outcome among index case"),
        llm=None,
        provider="openai",
        service=service,
    )

    artifact = updated["artifacts"]["files"][updated["agents"]["rag_db_qa"]["pending_column_review_artifact_id"]]
    assert ("prepare_column_selection", "Subset index-case age, gender, diabetes status, and TB outcome.") in service.calls
    assert artifact["content"]["source_question"] == (
        "Query my database, Help me to subset age, gender, diabetes status, and TB outcome among index case"
    )
    assert artifact["content"]["goal_text"] == (
        "Subset index-case age, gender, diabetes status, and TB outcome."
    )


def test_fresh_question_replaces_pending_extraction_gate() -> None:
    service = _Service()
    service.extraction_gate_result = {"label": "new_question", "confidence": 0.9}
    state = _state("Which forms contain smoking variables?")
    state["agents"]["rag_db_qa"] = {
        "active_intent": {
            "intent_id": "old",
            "goal_text": "Which forms contain age?",
            "source_question": "Which forms contain age?",
            "mode": "metadata",
            "status": "active",
        },
        "pending_extraction_opt_in": _expected_extraction_opt_in(
            intent_id="old",
            goal_text="Which forms contain age?",
        ),
    }

    updated = rag_db_qa_node(state, llm=None, provider="openai", service=service)

    rag_state = updated["agents"]["rag_db_qa"]
    assert rag_state["active_intent"]["goal_text"] == "Which forms contain smoking variables?"
    assert rag_state["pending_extraction_opt_in"]["intent_id"] != "old"
    assert rag_state["pending_extraction_opt_in"]["goal_text"] == "Which forms contain smoking variables?"


def test_approved_selection_pointer_resumes_sql_preparation_without_pending_review_pointer() -> None:
    service = _Service()
    state = _state("Continue")
    state["artifacts"]["files"]["selection-artifact"] = {
        "artifact_id": "selection-artifact",
        "created_at": "2026-05-01T00:00:00+00:00",
        "kind": "db_rag_column_selection",
        "producer": "rag_db_qa",
        "mime": "application/json",
        "summary": "Approved DB-RAG column selection.",
        "content": {
            "selection_id": "sel-1",
            "source_question": "Generate the SQL to subset index cases with diabetes.",
            "goal_text": "Generate the SQL to subset index cases with diabetes.",
            "intent_snapshot": {"intent_id": "intent:sql"},
            "retrieval_summary": {"tables": ["Form 2A"], "columns": ["IC_AGE"]},
            "tables": ["Form 2A"],
            "columns": [{"table": "Form 2A", "column": "IC_AGE", "description": "Age in years"}],
            "rationale": "Needed for extraction",
            "feedback_history": [],
            "status": "approved",
        },
    }
    state["agents"]["rag_db_qa"] = {
        "active_intent": {
            "intent_id": "intent:sql",
            "source_question": "Generate the SQL to subset index cases with diabetes.",
            "goal_text": "Generate the SQL to subset index cases with diabetes.",
            "mode": "extraction",
            "status": "active",
        },
        "approved_column_selection_artifact_id": "selection-artifact",
        "pending_column_review_artifact_id": None,
        "pending_sql_candidate_artifact_id": None,
    }

    updated = rag_db_qa_node(state, llm=None, provider="openai", service=service)

    rag_state = updated["agents"]["rag_db_qa"]
    assert ("prepare_sql_candidate", "Generate the SQL to subset index cases with diabetes.") in service.calls
    assert rag_state["approved_column_selection_artifact_id"] == "selection-artifact"
    assert rag_state["pending_sql_candidate_artifact_id"]
    assert "read-only SQL candidate" in updated["output"]["qa_response"]
    assert "review the SQL details in the panel below before execution" in updated["output"]["qa_response"]
    assert "Selected tables:" not in updated["output"]["qa_response"]
    assert "Selected columns:" not in updated["output"]["qa_response"]
    assert "Proposed SQL:" not in updated["output"]["qa_response"]
    assert "select IC_AGE from \"Form 2A\"" not in updated["output"]["qa_response"]
    events = updated["artifacts"]["conversation_events"]
    assert events[-2]["type"] == "sql"
    assert events[-2]["artifact_id"] == rag_state["pending_sql_candidate_artifact_id"]
    assert events[-1]["type"] == "review_request"
    assert events[-1]["artifact_id"] == rag_state["pending_sql_candidate_artifact_id"]


def test_missing_join_keys_reopens_column_review_instead_of_erroring() -> None:
    service = _Service()

    def fail_with_join_key_preflight(question: str, selection):
        del selection
        service.calls.append(("prepare_sql_candidate", question))
        raise DbRagUnanswerableError(
            "Multi-table extraction requires an approved join key column for every selected source table "
            "before SQL generation. Missing approved subject/family ID join keys for: Form 2A, Form 5. "
            "Include approved SUBJID or FID columns in the column selection, or use a single-table extraction."
        )

    service.prepare_sql_candidate = fail_with_join_key_preflight
    state = _state("Continue")
    state["artifacts"]["files"]["selection-artifact"] = {
        "artifact_id": "selection-artifact",
        "created_at": "2026-05-01T00:00:00+00:00",
        "kind": "db_rag_column_selection",
        "producer": "rag_db_qa",
        "mime": "application/json",
        "summary": "Approved DB-RAG column selection.",
        "content": {
            "selection_id": "sel-1",
            "source_question": "Extract observations across forms.",
            "goal_text": "Extract observations across forms.",
            "intent_snapshot": {"intent_id": "intent:sql"},
            "retrieval_summary": {"tables": ["Form 2A", "Form 5"], "columns": ["IC_DMDX", "CBC_HGAPCT"]},
            "tables": ["Form 2A", "Form 5"],
            "columns": [
                {"table": "Form 2A", "column": "IC_DMDX", "description": "Diabetes status"},
                {"table": "Form 5", "column": "CBC_HGAPCT", "description": "HbA1C"},
            ],
            "rationale": "Needed for extraction",
            "feedback_history": [{"action": "approve", "feedback": "looks right"}],
            "status": "approved",
        },
    }
    state["agents"]["rag_db_qa"] = {
        "active_intent": {
            "intent_id": "intent:sql",
            "source_question": "Extract observations across forms.",
            "goal_text": "Extract observations across forms.",
            "mode": "extraction",
            "status": "active",
        },
        "approved_column_selection_artifact_id": "selection-artifact",
        "pending_column_review_artifact_id": None,
        "pending_sql_candidate_artifact_id": None,
    }

    updated = rag_db_qa_node(state, llm=None, provider="openai", service=service)

    rag_state = updated["agents"]["rag_db_qa"]
    reopened = updated["artifacts"]["files"]["selection-artifact"]["content"]
    assert rag_state["thread_status"] == "awaiting_column_review"
    assert rag_state["pending_column_review_artifact_id"] == "selection-artifact"
    assert rag_state["approved_column_selection_artifact_id"] is None
    assert rag_state["pending_sql_candidate_artifact_id"] is None
    assert updated["output"].get("error") is None
    assert "join-key clarification" in updated["output"]["qa_response"]
    assert "SUBJID or FID" in updated["output"]["qa_response"]
    assert "Please review the proposed column selection in the panel below." in updated["output"]["qa_response"]
    assert reopened["status"] == "needs_revision"
    assert reopened["feedback_history"][-1]["action"] == "sql_preflight_missing_join_keys"
    assert "Include approved SUBJID or FID columns" in reopened["feedback_history"][-1]["feedback"]


def test_pending_column_review_artifact_reprompts_instead_of_erroring() -> None:
    service = _Service()
    state = _state("Continue")
    state["artifacts"]["files"]["selection-artifact"] = {
        "artifact_id": "selection-artifact",
        "created_at": "2026-05-01T00:00:00+00:00",
        "kind": "db_rag_column_selection",
        "producer": "rag_db_qa",
        "mime": "application/json",
        "summary": "Proposed DB-RAG column selection awaiting human review.",
        "content": {
            "selection_id": "sel-1",
            "source_question": "Generate the SQL to subset index cases with diabetes.",
            "goal_text": "Generate the SQL to subset index cases with diabetes.",
            "intent_snapshot": {"intent_id": "intent:sql"},
            "retrieval_summary": {"tables": ["Form 2A"], "columns": ["IC_AGE"]},
            "tables": ["Form 2A"],
            "columns": [{"table": "Form 2A", "column": "IC_AGE", "description": "Age in years"}],
            "rationale": "Needed for extraction",
            "feedback_history": [],
            "status": "awaiting_review",
        },
    }
    state["agents"]["rag_db_qa"] = {
        "active_intent": {
            "intent_id": "intent:sql",
            "source_question": "Generate the SQL to subset index cases with diabetes.",
            "goal_text": "Generate the SQL to subset index cases with diabetes.",
            "mode": "extraction",
            "status": "active",
        },
        "pending_column_review_artifact_id": "selection-artifact",
        "approved_column_selection_artifact_id": None,
        "pending_sql_candidate_artifact_id": None,
    }

    updated = rag_db_qa_node(state, llm=None, provider="openai", service=service)

    rag_state = updated["agents"]["rag_db_qa"]
    assert rag_state["thread_status"] == "awaiting_column_review"
    assert rag_state["pending_column_review_artifact_id"] == "selection-artifact"
    assert "Please review the proposed DB-RAG column selection in the panel below." in updated["output"]["qa_response"]
    assert "Selected tables:" not in updated["output"]["qa_response"]
    assert "Selected columns:" not in updated["output"]["qa_response"]
    assert "Form 2A.IC_AGE: Age in years" not in updated["output"]["qa_response"]
    assert updated["output"].get("error") is None


def test_pending_extraction_reply_uses_service_classifiers_for_substantive_followup() -> None:
    service = _Service()
    service.extraction_gate_result = {"label": "reply_to_pending_gate", "confidence": 0.9}
    service.pending_reply_result = {"label": "substantive_followup", "confidence": 0.9}
    state = _state_with_pending_extraction_reply("Use household contacts instead.")

    updated = rag_db_qa_node(state, llm=None, provider="openai", service=service)

    rag_state = updated["agents"]["rag_db_qa"]
    card = updated["memory"]["user_intents"]["user-intent-1"]
    assert service.calls[:2] == [
        ("classify_extraction_gate_message", "Use household contacts instead."),
        ("classify_pending_reply", "Use household contacts instead."),
    ]
    assert ("prepare_column_selection", "Use household contacts instead.") in service.calls
    assert rag_state["pending_extraction_opt_in"] is None
    assert rag_state["pending_column_review_artifact_id"]
    assert rag_state["active_intent"]["goal_text"] == "Use household contacts instead."
    assert card["source_question"] == "Which forms contain age?"
    assert card["goal_text"] == "Use household contacts instead."
    assert card["status"] == "awaiting_column_review"


def test_pending_extraction_yes_reply_updates_user_intent_to_awaiting_column_review() -> None:
    service = _Service()
    service.extraction_gate_result = {"label": "reply_to_pending_gate", "confidence": 0.9}
    service.pending_reply_result = {"label": "yes", "confidence": 0.9}
    state = _state_with_pending_extraction_reply("yes")

    updated = rag_db_qa_node(state, llm=None, provider="openai", service=service)

    card = updated["memory"]["user_intents"]["user-intent-1"]
    assert updated["agents"]["rag_db_qa"]["pending_column_review_artifact_id"]
    assert card["source_question"] == "Which forms contain age?"
    assert card["status"] == "awaiting_column_review"


def test_pending_extraction_no_reply_updates_user_intent_to_declined() -> None:
    service = _Service()
    service.extraction_gate_result = {"label": "reply_to_pending_gate", "confidence": 0.9}
    service.pending_reply_result = {"label": "no", "confidence": 0.9}
    state = _state_with_pending_extraction_reply("no")

    updated = rag_db_qa_node(state, llm=None, provider="openai", service=service)

    card = updated["memory"]["user_intents"]["user-intent-1"]
    assert updated["agents"]["rag_db_qa"]["pending_extraction_opt_in"] is None
    assert card["source_question"] == "Which forms contain age?"
    assert card["status"] == "declined"


def test_pending_extraction_boundary_classifier_can_route_fresh_question_without_pending_reply_classification() -> None:
    service = _Service()
    service.extraction_gate_result = {"label": "new_question", "confidence": 0.9}
    state = _state("Which forms contain smoking variables?")
    state["agents"]["rag_db_qa"] = {
        "active_intent": {
            "intent_id": "intent:age",
            "goal_text": "Which forms contain age?",
            "source_question": "Which forms contain age?",
            "mode": "metadata",
            "status": "active",
        },
        "pending_extraction_opt_in": _expected_extraction_opt_in(
            intent_id="intent:age",
            goal_text="Which forms contain age?",
        ),
    }

    updated = rag_db_qa_node(state, llm=None, provider="openai", service=service)

    rag_state = updated["agents"]["rag_db_qa"]
    assert ("classify_extraction_gate_message", "Which forms contain smoking variables?") in service.calls
    assert not any(call[0] == "classify_pending_reply" for call in service.calls)
    assert rag_state["active_intent"]["goal_text"] == "Which forms contain smoking variables?"
    assert rag_state["pending_extraction_opt_in"]["goal_text"] == "Which forms contain smoking variables?"


def test_pending_extraction_boundary_unknown_reprompts_without_pending_reply_classification() -> None:
    service = _Service()
    service.extraction_gate_result = {"label": "unknown", "confidence": 0.1}
    service.pending_reply_result = {"label": "yes", "confidence": 0.9}
    state = _state("maybe")
    state["agents"]["rag_db_qa"] = {
        "active_intent": {
            "intent_id": "intent:age",
            "goal_text": "Which forms contain age?",
            "source_question": "Which forms contain age?",
            "mode": "metadata",
            "status": "active",
        },
        "pending_extraction_opt_in": _expected_extraction_opt_in(
            intent_id="intent:age",
            goal_text="Which forms contain age?",
        ),
    }

    updated = rag_db_qa_node(state, llm=None, provider="openai", service=service)

    rag_state = updated["agents"]["rag_db_qa"]
    assert service.calls == [("classify_extraction_gate_message", "maybe")]
    assert not any(call[0] == "prepare_column_selection" for call in service.calls)
    assert rag_state["pending_extraction_opt_in"] == _expected_extraction_opt_in(
        intent_id="intent:age",
        goal_text="Which forms contain age?",
    )
    assert rag_state["thread_status"] == "awaiting_extraction_opt_in"
    assert "Which forms contain age?" in updated["output"]["qa_response"]
    assert "Please reply with 'yes'" in updated["output"]["qa_response"]


def test_pending_extraction_without_service_classifiers_reprompts_conservatively() -> None:
    service = _ServiceWithoutPendingClassifiers()
    state = _state("yes")
    state["agents"]["rag_db_qa"] = {
        "active_intent": {
            "intent_id": "intent:age",
            "goal_text": "Which forms contain age?",
            "source_question": "Which forms contain age?",
            "mode": "metadata",
            "status": "active",
        },
        "pending_extraction_opt_in": _expected_extraction_opt_in(
            intent_id="intent:age",
            goal_text="Which forms contain age?",
        ),
    }

    updated = rag_db_qa_node(state, llm=None, provider="openai", service=service)

    rag_state = updated["agents"]["rag_db_qa"]
    assert not any(call[0] == "prepare_column_selection" for call in service.calls)
    assert rag_state["pending_extraction_opt_in"] == _expected_extraction_opt_in(
        intent_id="intent:age",
        goal_text="Which forms contain age?",
    )
    assert rag_state["thread_status"] == "awaiting_extraction_opt_in"
    assert "Which forms contain age?" in updated["output"]["qa_response"]
    assert "Please reply with 'yes'" in updated["output"]["qa_response"]


def test_active_routing_ignores_legacy_pending_sql_candidate_without_pointer() -> None:
    service = _Service()
    state = _state("What tables contain age?")
    state["agents"]["rag_db_qa"] = {
        "active_intent": {
            "intent_id": "intent:sql",
            "source_question": "Generate the SQL to subset index cases with diabetes.",
            "goal_text": "Generate the SQL to subset index cases with diabetes.",
            "mode": "extraction",
            "status": "active",
        },
        "pending_sql_candidate": {
            "question": "Generate the SQL to subset index cases with diabetes.",
            "sql": "select IC_AGE from \"Form 2A\"",
            "tables": ["Form 2A"],
            "columns": [{"table": "Form 2A", "column": "IC_AGE", "description": "Age in years"}],
            "selection_id": "sel-1",
            "status": "prepared",
        },
    }

    updated = rag_db_qa_node(state, llm=None, provider="openai", service=service)

    assert ("answer_from_context", "What tables contain age?") in service.calls
    assert not any(call[0] == "prepare_sql_candidate" for call in service.calls)
    assert "Metadata answer for: What tables contain age?" in updated["output"]["qa_response"]


def test_pending_sql_pointer_resume_repopulates_legacy_candidate_mirror() -> None:
    service = _Service()
    state = _state("continue")
    state["artifacts"]["files"]["sql-art-1"] = {
        "artifact_id": "sql-art-1",
        "created_at": "2026-05-01T00:00:00+00:00",
        "kind": "db_rag_sql_candidate",
        "producer": "rag_db_qa",
        "mime": "application/json",
        "summary": "Prepared read-only SQL candidate for DB-RAG review.",
        "content": {
            "question": "Generate the SQL to subset index cases with diabetes.",
            "sql": "select IC_AGE from \"Form 2A\"",
            "tables": ["Form 2A"],
            "columns": [{"table": "Form 2A", "column": "IC_AGE", "description": "Age in years"}],
            "selection_id": "sel-1",
            "status": "prepared",
        },
    }
    state["agents"]["rag_db_qa"] = {
        "pending_sql_candidate_artifact_id": "sql-art-1",
        "pending_sql_candidate": None,
    }

    updated = rag_db_qa_node(state, llm=None, provider="openai", service=service)

    rag_state = updated["agents"]["rag_db_qa"]
    assert rag_state["pending_sql_candidate_artifact_id"] == "sql-art-1"
    assert rag_state["pending_sql_candidate"]["sql"] == "select IC_AGE from \"Form 2A\""
    assert rag_state["thread_status"] == "awaiting_sql_review"


def test_active_routing_ignores_legacy_approved_selection_without_pointer() -> None:
    service = _Service()
    state = _state("What tables contain age?")
    state["agents"]["rag_db_qa"] = {
        "active_intent": {
            "intent_id": "intent:sql",
            "source_question": "Generate the SQL to subset index cases with diabetes.",
            "goal_text": "Generate the SQL to subset index cases with diabetes.",
            "mode": "extraction",
            "status": "active",
        },
        "pending_column_review": {
            "question": "Generate the SQL to subset index cases with diabetes.",
            "goal_text": "Generate the SQL to subset index cases with diabetes.",
            "selection_id": "sel-1",
            "tables": ["Form 2A"],
            "columns": [{"table": "Form 2A", "column": "IC_AGE", "description": "Age in years"}],
            "rationale": "Needed for extraction",
            "feedback_history": [],
            "status": "approved",
        },
    }

    updated = rag_db_qa_node(state, llm=None, provider="openai", service=service)

    assert ("answer_from_context", "What tables contain age?") in service.calls
    assert not any(call[0] == "prepare_sql_candidate" for call in service.calls)
    assert "Metadata answer for: What tables contain age?" in updated["output"]["qa_response"]


def test_sql_preparation_failure_asks_recoverable_clarification_instead_of_erroring() -> None:
    from graph.state import MetaKeys

    service = _Service()
    service.sql_error_message = "bad sql generation"
    state = _state("continue")
    state["agents"]["rag_db_qa"] = {
        "active_intent": {
            "intent_id": "intent:sql",
            "source_question": "Generate the SQL to subset index cases with diabetes.",
            "goal_text": "Generate the SQL to subset index cases with diabetes.",
            "mode": "extraction",
            "status": "active",
        },
        "approved_column_selection_artifact_id": "sel-art-1",
    }
    state["artifacts"]["files"]["sel-art-1"] = {
        "artifact_id": "sel-art-1",
        "created_at": "2026-05-01T00:00:00+00:00",
        "kind": "db_rag_column_selection",
        "producer": "rag_db_qa",
        "mime": "application/json",
        "summary": "Approved selection",
        "content": {
            "selection_id": "sel-1",
            "source_question": "Generate the SQL to subset index cases with diabetes.",
            "goal_text": "Generate the SQL to subset index cases with diabetes.",
            "intent_snapshot": {"intent_id": "intent:sql"},
            "retrieval_summary": {"tables": ["Form 2A"], "columns": ["IC_AGE"]},
            "tables": ["Form 2A"],
            "columns": [{"table": "Form 2A", "column": "IC_AGE", "description": "Age in years"}],
            "rationale": "Needed for extraction",
            "feedback_history": [],
            "status": "approved",
        },
    }

    updated = rag_db_qa_node(state, llm=None, provider="openai", service=service)

    rag_state = updated["agents"]["rag_db_qa"]
    assert ("build_recoverable_error_clarification", "db_rag_sql_preparation") in service.calls
    assert rag_state["thread_status"] == "awaiting_recoverable_error_clarification"
    assert rag_state["approved_column_selection_artifact_id"] == "sel-art-1"
    assert rag_state["pending_sql_candidate_artifact_id"] is None
    assert rag_state["pending_recoverable_error"]["status"] == "awaiting_reply"
    assert rag_state["pending_recoverable_error"]["selection_artifact_id"] == "sel-art-1"
    assert updated["output"].get("error") is None
    assert updated["meta"][MetaKeys.AWAITING_USER_CLARIFICATION] is True
    assert updated["meta"][MetaKeys.CLARIFICATION_KIND] == "db_rag_recoverable_error"
    assert updated["meta"][MetaKeys.CLARIFICATION_RETURN_NODE] == "rag_db_qa"
    assert "Please clarify how to revise" in updated["output"]["qa_response"]


def test_recoverable_sql_preparation_reply_reopens_column_review_with_feedback() -> None:
    service = _Service()
    state = _state("Use SUBJID_PSEUDO as the join key.")
    state["agents"]["rag_db_qa"] = {
        "active_intent": {
            "intent_id": "intent:sql",
            "source_question": "Generate the SQL to subset index cases with diabetes.",
            "goal_text": "Generate the SQL to subset index cases with diabetes.",
            "mode": "extraction",
            "status": "active",
        },
        "approved_column_selection_artifact_id": "sel-art-1",
        "pending_recoverable_error": {
            "status": "awaiting_reply",
            "stage": "sql_preparation",
            "selection_artifact_id": "sel-art-1",
            "error_payload": {
                "category": "db_rag_sql",
                "type": "RuntimeError",
                "message": "bad sql generation",
                "selection_artifact_id": "sel-art-1",
            },
            "prompt": "Please clarify how to revise the DB-RAG selection so SQL can be generated.",
        },
    }
    state["artifacts"]["files"]["sel-art-1"] = {
        "artifact_id": "sel-art-1",
        "created_at": "2026-05-01T00:00:00+00:00",
        "kind": "db_rag_column_selection",
        "producer": "rag_db_qa",
        "mime": "application/json",
        "summary": "Approved selection",
        "content": {
            "selection_id": "sel-1",
            "source_question": "Generate the SQL to subset index cases with diabetes.",
            "goal_text": "Generate the SQL to subset index cases with diabetes.",
            "intent_snapshot": {"intent_id": "intent:sql"},
            "retrieval_summary": {"tables": ["Form 2A"], "columns": ["IC_AGE"]},
            "tables": ["Form 2A"],
            "columns": [{"table": "Form 2A", "column": "IC_AGE", "description": "Age in years"}],
            "rationale": "Needed for extraction",
            "feedback_history": [],
            "status": "approved",
        },
    }

    updated = rag_db_qa_node(state, llm=None, provider="openai", service=service)

    rag_state = updated["agents"]["rag_db_qa"]
    assert rag_state.get("pending_recoverable_error") is None
    assert rag_state["thread_status"] == "awaiting_column_review"
    assert rag_state["pending_column_review_artifact_id"]
    assert not rag_state.get("approved_column_selection_artifact_id")
    assert ("prepare_column_selection", "Generate the SQL to subset index cases with diabetes.") in service.calls
    reopened = rag_state["pending_column_review"]
    assert reopened["feedback_history"][-1]["action"] == "sql_preparation_recovery_clarification"
    assert "Use SUBJID_PSEUDO as the join key." in reopened["feedback_history"][-1]["feedback"]


def test_missing_pending_sql_artifact_clears_pointer_and_legacy_mirror() -> None:
    service = _Service()
    state = _state("continue")
    state["agents"]["rag_db_qa"] = {
        "approved_column_selection_artifact_id": "sel-art-1",
        "pending_sql_candidate_artifact_id": "missing-sql-artifact",
        "pending_sql_candidate": {
            "question": "Generate the SQL to subset index cases with diabetes.",
            "sql": "select IC_AGE from \"Form 2A\"",
            "tables": ["Form 2A"],
            "columns": [{"table": "Form 2A", "column": "IC_AGE", "description": "Age in years"}],
            "selection_id": "sel-1",
            "status": "prepared",
        },
    }

    updated = rag_db_qa_node(state, llm=None, provider="openai", service=service)

    rag_state = updated["agents"]["rag_db_qa"]
    assert rag_state["thread_status"] == "error"
    assert rag_state["approved_column_selection_artifact_id"] is None
    assert rag_state["pending_sql_candidate_artifact_id"] is None
    assert rag_state["pending_sql_candidate"] is None


def test_execution_artifact_records_selection_and_sql_artifact_ids(monkeypatch) -> None:
    import pandas as pd

    from graph.nodes.db_rag_qa import helpers as rag_helpers
    from graph.state import MetaKeys

    def _fake_persist_dataset_artifact(
        *,
        runtime_root,
        thread_id,
        dataset_id,
        kind,
        dataframe,
        schema,
        provenance,
    ):
        del runtime_root, thread_id, dataframe, schema
        return {
            "id": dataset_id,
            "kind": kind,
            "created_at": "2026-05-01T00:00:00+00:00",
            "row_count": 2,
            "column_count": 1,
            "columns": ["IC_AGE"],
            "provenance": dict(provenance or {}),
        }

    monkeypatch.setattr(rag_helpers, "persist_dataset_artifact", _fake_persist_dataset_artifact)

    state = _state("continue")
    state["meta"][MetaKeys.THREAD_ID] = "thread-1"
    state["artifacts"]["files"]["sel-art-1"] = {
        "artifact_id": "sel-art-1",
        "created_at": "2026-05-01T00:00:00+00:00",
        "kind": "db_rag_column_selection",
        "producer": "rag_db_qa",
        "mime": "application/json",
        "summary": "Approved selection",
        "content": {
            "selection_id": "sel-1",
            "tables": ["Form 2A"],
            "columns": [{"table": "Form 2A", "column": "IC_AGE", "description": "Age in years"}],
            "feedback_history": [{"action": "approve", "feedback": "looks right"}],
        },
    }
    rag_state = {
        "pending_column_review": {
            "selection_id": "sel-1",
            "tables": ["WRONG_TABLE"],
            "columns": [{"table": "WRONG_TABLE", "column": "WRONG_COL", "description": "Wrong"}],
            "feedback_history": [{"action": "approve", "feedback": "wrong mirror"}],
        },
        "pending_column_review_artifact_id": "sel-art-1",
        "approved_column_selection_artifact_id": "sel-art-1",
        "pending_sql_candidate_artifact_id": "sql-art-2",
        "sql_review_approved_artifact_id": "sql-art-2",
    }
    candidate = SimpleNamespace(
        question="Generate the SQL to subset index cases with diabetes.",
        source_question="Generate the SQL to subset index cases with diabetes.",
        goal_text="Generate the SQL to subset index cases with diabetes.",
        sql='select IC_AGE from "Form 2A"',
        tables=["Form 2A"],
        columns=[{"table": "Form 2A", "column": "IC_AGE", "description": "Age in years"}],
        selection_id="wrong-sel",
        status="prepared",
    )

    class _ExecutionService:
        def execute_prepared_sql(self, prepared_candidate):
            return SimpleNamespace(
                answer="Read-only SQL execution completed.",
                sql=prepared_candidate.sql,
                source_tables=list(prepared_candidate.tables),
                dataframe=pd.DataFrame({"IC_AGE": [34, 35]}),
            )

    updated = rag_helpers._execute_prepared_sql_candidate(state, rag_state, candidate, _ExecutionService())

    dataset = updated["artifacts"]["datasets"][updated["artifacts"]["active_dataset_id"]]
    assert dataset["provenance"]["selection_artifact_id"] == "sel-art-1"
    assert dataset["provenance"]["sql_candidate_artifact_id"] == "sql-art-2"
    assert dataset["provenance"]["selection_id"] == "sel-1"
    assert dataset["provenance"]["source_question"] == "Generate the SQL to subset index cases with diabetes."
    assert dataset["provenance"]["goal_text"] == "Generate the SQL to subset index cases with diabetes."
    assert dataset["provenance"]["selected_tables"] == ["Form 2A"]
    assert dataset["provenance"]["selected_columns"] == [
        {"table": "Form 2A", "column": "IC_AGE", "description": "Age in years"}
    ]
    assert dataset["provenance"]["feedback_history"] == [{"action": "approve", "feedback": "looks right"}]


def test_sql_execution_completion_writes_db_rag_sql_task(monkeypatch) -> None:
    state, rag_state, candidate = _successful_sql_execution_state()

    updated = _execute_successful_sql(monkeypatch, state, rag_state, candidate)

    memory = updated["memory"]
    task_id = memory["task_order"][0]
    task = memory["completed_tasks"][task_id]
    assert task["kind"] == "db_rag_sql_extraction"
    assert task["source_question"] == "Generate the SQL to subset index cases with diabetes."
    assert task["goal_text"] == "Generate the SQL to subset index cases with diabetes."
    assert task["artifact_refs"] == {
        "selection_artifact_id": "sel-art-1",
        "sql_candidate_artifact_id": "sql-art-2",
        "dataset_artifact_id": "dataset-art-1",
    }
    assert task["summary"] == "Reviewed SQL executed and saved dataset dataset-art-1."
    assert task["provenance"] == {
        "producer_node": "rag_db_qa",
        "selection_id": "sel-1",
    }
    assert memory["last_task_id"] == task_id
    assert memory["last_task_id_by_kind"]["db_rag_sql_extraction"] == task_id

    memory_text = json.dumps(task, sort_keys=True)
    assert 'select IC_AGE from "Form 2A"' not in memory_text
    assert "feedback_history" not in memory_text
    assert "dataframe" not in memory_text
    assert "row_count" not in memory_text
    assert "column_count" not in memory_text
    assert "columns" not in memory_text
    assert "tables" not in memory_text


def test_successful_sql_execution_keeps_reviewed_sql_and_selection_in_display_history(monkeypatch) -> None:
    from utils.display_history import build_display_history

    state, rag_state, candidate = _successful_sql_execution_state()

    updated = _execute_successful_sql(monkeypatch, state, rag_state, candidate)

    response = updated["output"]["qa_response"]
    assert "Reviewed tables:" in response
    assert "- Form 2A" in response
    assert "Reviewed columns:" in response
    assert "- Form 2A.IC_AGE: Age in years" in response
    assert "SQL used:" in response
    assert 'select IC_AGE from "Form 2A"' in response

    history_text = "\n\n".join(str(message.content) for message in build_display_history(updated))
    assert "Reviewed tables:" in history_text
    assert "- Form 2A.IC_AGE: Age in years" in history_text
    assert 'select IC_AGE from "Form 2A"' in history_text


def test_sql_execution_completion_uses_artifact_question_text_when_candidate_question_empty(monkeypatch) -> None:
    state, rag_state, candidate = _successful_sql_execution_state()
    state["artifacts"]["files"]["sel-art-1"]["content"]["source_question"] = "Original user extraction request"
    state["artifacts"]["files"]["sel-art-1"]["content"]["goal_text"] = "Reviewed extraction goal"
    state["artifacts"]["files"]["sql-art-2"] = {
        "artifact_id": "sql-art-2",
        "created_at": "2026-05-01T00:00:01+00:00",
        "kind": "db_rag_sql_candidate",
        "producer": "rag_db_qa",
        "mime": "application/json",
        "summary": "SQL candidate",
        "content": {
            "sql_candidate_id": "sql-2",
            "selection_artifact_id": "sel-art-1",
            "source_question": "Artifact source question",
            "goal_text": "Artifact goal text",
            "tables": ["Form 2A"],
            "columns": [{"table": "Form 2A", "column": "IC_AGE", "description": "Age in years"}],
            "sql": 'select IC_AGE from "Form 2A"',
            "status": "prepared",
        },
    }
    candidate.question = ""
    candidate.source_question = ""
    candidate.goal_text = ""

    updated = _execute_successful_sql(monkeypatch, state, rag_state, candidate)

    task_id = updated["memory"]["last_task_id"]
    task = updated["memory"]["completed_tasks"][task_id]
    assert task["source_question"] == "Artifact source question"
    assert task["goal_text"] == "Artifact goal text"
    assert task["label"] == "DB-RAG SQL extraction: Artifact source question"


def test_sql_execution_completion_keeps_live_pointers_cleared(monkeypatch) -> None:
    state, rag_state, candidate = _successful_sql_execution_state()

    updated = _execute_successful_sql(monkeypatch, state, rag_state, candidate)

    updated_rag_state = updated["agents"]["rag_db_qa"]
    assert updated_rag_state["thread_status"] == "completed"
    assert updated_rag_state["active_thread"] is False
    assert updated_rag_state["pending_sql_candidate_artifact_id"] is None
    assert updated_rag_state.get("pending_sql_candidate") is None
    assert updated_rag_state["pending_column_review_artifact_id"] is None
    assert updated_rag_state["approved_column_selection_artifact_id"] is None
    assert updated_rag_state.get("pending_column_review") is None


def test_revised_sql_execution_task_links_parent_task(monkeypatch) -> None:
    from graph.memory import complete_task

    state, rag_state, candidate = _successful_sql_execution_state()
    state = complete_task(
        state,
        kind="db_rag_sql_extraction",
        source_question="Original extraction",
        goal_text="Original extraction",
        label="DB-RAG SQL extraction: Original extraction",
        summary="Original task.",
    )
    parent_task_id = state["memory"]["last_task_id"]
    rag_state["active_task"] = {
        "task_id": "task_generated_placeholder",
        "parent_task_id": parent_task_id,
        "relationship_to_parent": "revision",
    }

    updated = _execute_successful_sql(monkeypatch, state, rag_state, candidate)

    memory = updated["memory"]
    task_id = memory["last_task_id"]
    task = memory["completed_tasks"][task_id]
    assert task_id != "task_generated_placeholder"
    assert task["parent_task_id"] == parent_task_id
    assert task["relationship_to_parent"] == "revision"
    assert memory["task_order"] == [parent_task_id, task_id]
    assert memory["last_task_id_by_kind"]["db_rag_sql_extraction"] == task_id
    assert updated["agents"]["rag_db_qa"].get("active_task") is None


def test_build_subset_schema_enriches_from_reviewed_schema_catalog(monkeypatch) -> None:
    import pandas as pd

    from graph.nodes.db_rag_qa import helpers as rag_helpers

    def _fake_lookup(table: str, column: str):
        if table == "Form 2A" and column == "IC_AGE":
            return {
                "description": "Age in years at enrollment",
                "values": None,
                "depends_on": "IC_ENROLL=1",
                "condition": "index case only",
                "section_context": "Demographics",
            }
        return None

    monkeypatch.setattr(rag_helpers, "_lookup_schema_variable_metadata", _fake_lookup)

    df = pd.DataFrame({"IC_AGE": [34, 35], "UNKNOWN_COL": [1, 0]})
    schema = rag_helpers._build_subset_schema(
        df,
        [
            {"table": "Form 2A", "column": "IC_AGE", "description": "Age in years"},
            {"table": "Form X", "column": "UNKNOWN_COL", "description": "Fallback description"},
        ],
    )

    assert schema["IC_AGE"] == {
        "description": "Age in years at enrollment",
        "depends_on": "IC_ENROLL=1",
        "condition": "index case only",
        "section_context": "Demographics",
        "dataType": "int64",
    }
    assert schema["UNKNOWN_COL"] == {
        "description": "Fallback description",
        "dataType": "int64",
    }


def test_build_subset_schema_enriches_aliased_output_columns_from_selected_columns(monkeypatch) -> None:
    import pandas as pd

    from graph.nodes.db_rag_qa import helpers as rag_helpers

    def _fake_lookup(table: str, column: str):
        if table == "Form 2A" and column == "IC_AGE":
            return {
                "description": "Age in years at enrollment",
                "values": None,
                "depends_on": "IC_ENROLL=1",
                "condition": "index case only",
                "section_context": "Demographics",
            }
        if table == "Form 2A" and column == "IC_GENDER":
            return {
                "description": "Gender at enrollment",
                "values": {"1": "Male", "2": "Female"},
                "depends_on": None,
                "condition": None,
                "section_context": "Demographics",
            }
        return None

    monkeypatch.setattr(rag_helpers, "_lookup_schema_variable_metadata", _fake_lookup)

    df = pd.DataFrame({"age": [34, 35], "gender": ["F", "M"]})
    schema = rag_helpers._build_subset_schema(
        df,
        [
            {"table": "Form 2A", "column": "IC_AGE", "description": "Age in years"},
            {"table": "Form 2A", "column": "IC_GENDER", "description": "Gender"},
        ],
        sql='SELECT IC_AGE AS age, IC_GENDER AS gender FROM "Form 2A"',
    )

    assert schema["age"] == {
        "description": "Age in years at enrollment",
        "depends_on": "IC_ENROLL=1",
        "condition": "index case only",
        "section_context": "Demographics",
        "dataType": "int64",
    }
    assert schema["gender"] == {
        "description": "Gender at enrollment",
        "values": {"1": "Male", "2": "Female"},
        "section_context": "Demographics",
        "dataType": "object",
    }


def test_build_subset_schema_enriches_renamed_columns_from_approved_columns_without_sql(monkeypatch) -> None:
    import pandas as pd

    from graph.nodes.db_rag_qa import helpers as rag_helpers

    def _fake_lookup(table: str, column: str):
        if table == "Form 2A" and column == "IC_AGE":
            return {"description": "Age in years at enrollment"}
        if table == "Form 2A" and column == "IC_GENDER":
            return {"description": "Gender at enrollment"}
        return None

    monkeypatch.setattr(rag_helpers, "_lookup_schema_variable_metadata", _fake_lookup)

    df = pd.DataFrame({"age": [34, 35], "gender": ["F", "M"]})
    schema = rag_helpers._build_subset_schema(
        df,
        [
            {"table": "Form 2A", "column": "IC_AGE", "description": "Age in years"},
            {"table": "Form 2A", "column": "IC_GENDER", "description": "Gender"},
        ],
        sql="",
    )

    assert schema["age"]["description"] == "Age in years at enrollment"
    assert schema["gender"]["description"] == "Gender at enrollment"
    assert schema["age"]["dataType"] == "int64"
    assert schema["gender"]["dataType"] == "object"


def test_execution_does_not_auto_repair_unreviewed_sql(monkeypatch) -> None:
    from graph.nodes.db_rag_qa import helpers as rag_helpers
    from graph.state import MetaKeys

    state = _state("continue")
    state["meta"][MetaKeys.THREAD_ID] = "thread-1"
    state["artifacts"]["files"]["sel-art-1"] = {
        "artifact_id": "sel-art-1",
        "created_at": "2026-05-01T00:00:00+00:00",
        "kind": "db_rag_column_selection",
        "producer": "rag_db_qa",
        "mime": "application/json",
        "summary": "Approved selection",
        "content": {
            "selection_id": "sel-1",
            "tables": ["Form 2A"],
            "columns": [{"table": "Form 2A", "column": "IC_AGE", "description": "Age in years"}],
            "feedback_history": [],
        },
    }
    rag_state = {
        "pending_column_review": {
            "selection_id": "sel-1",
            "tables": ["Form 2A"],
            "columns": [{"table": "Form 2A", "column": "IC_AGE", "description": "Age in years"}],
            "feedback_history": [],
        },
        "pending_column_review_artifact_id": "sel-art-1",
        "approved_column_selection_artifact_id": "sel-art-1",
        "pending_sql_candidate_artifact_id": "sql-art-2",
    }
    candidate = SimpleNamespace(
        question="Generate the SQL to subset index cases with diabetes.",
        source_question="Generate the SQL to subset index cases with diabetes.",
        goal_text="Generate the SQL to subset index cases with diabetes.",
        sql='select IC_AGE from "Form 2A"',
        tables=["Form 2A"],
        columns=[{"table": "Form 2A", "column": "IC_AGE", "description": "Age in years"}],
        selection_id="sel-1",
        status="prepared",
    )

    execution_calls: list[object] = []
    repair_calls: list[tuple[object, str]] = []

    class _ExecutionService:
        def execute_prepared_sql(self, prepared_candidate):
            execution_calls.append(prepared_candidate)
            raise RuntimeError(f"bad sql: {prepared_candidate.sql}")

        def repair_prepared_sql_candidate(self, prepared_candidate, error_text):
            repair_calls.append((prepared_candidate, error_text))
            return prepared_candidate

    updated = rag_helpers._execute_prepared_sql_candidate(state, rag_state, candidate, _ExecutionService())

    assert execution_calls == []
    assert repair_calls == []
    assert updated["agents"]["rag_db_qa"]["thread_status"] == "error"
    assert updated["output"]["error"]["category"] == "db_rag_sql"
    assert updated["output"]["error"]["type"] == "MissingSqlReviewApproval"


def test_execution_error_reopens_column_review_from_canonical_selection_artifact() -> None:
    from graph.nodes.db_rag_qa import helpers as rag_helpers
    from graph.state import MetaKeys

    state = _state("continue")
    state["meta"][MetaKeys.THREAD_ID] = "thread-1"
    state["artifacts"]["files"]["sel-art-1"] = {
        "artifact_id": "sel-art-1",
        "created_at": "2026-05-01T00:00:00+00:00",
        "kind": "db_rag_column_selection",
        "producer": "rag_db_qa",
        "mime": "application/json",
        "summary": "Approved selection",
        "content": {
            "selection_id": "sel-1",
            "tables": ["Form 2A"],
            "columns": [{"table": "Form 2A", "column": "IC_AGE", "description": "Age in years"}],
            "feedback_history": [{"action": "approve", "feedback": "looks right"}],
        },
    }
    rag_state = {
        "pending_column_review": {
            "selection_id": "wrong-sel",
            "tables": ["WRONG_TABLE"],
            "columns": [{"table": "WRONG_TABLE", "column": "WRONG_COL", "description": "Wrong"}],
            "feedback_history": [{"action": "approve", "feedback": "wrong mirror"}],
        },
        "pending_column_review_artifact_id": "sel-art-1",
        "approved_column_selection_artifact_id": "sel-art-1",
        "pending_sql_candidate_artifact_id": "sql-art-2",
        "sql_review_approved_artifact_id": "sql-art-2",
    }
    candidate = SimpleNamespace(
        question="Generate the SQL to subset index cases with diabetes.",
        source_question="Generate the SQL to subset index cases with diabetes.",
        goal_text="Generate the SQL to subset index cases with diabetes.",
        sql='select IC_AGE from "Form 2A"',
        tables=["Form 2A"],
        columns=[{"table": "Form 2A", "column": "IC_AGE", "description": "Age in years"}],
        selection_id="wrong-sel",
        status="prepared",
    )

    class _ExecutionService:
        def execute_prepared_sql(self, prepared_candidate):
            raise RuntimeError(f"bad sql: {prepared_candidate.sql}")

    updated = rag_helpers._execute_prepared_sql_candidate(state, rag_state, candidate, _ExecutionService())

    reopened = updated["agents"]["rag_db_qa"]["pending_column_review"]
    assert updated["agents"]["rag_db_qa"]["thread_status"] == "error"
    assert reopened["selection_id"] == "sel-1"
    assert reopened["tables"] == ["Form 2A"]
    assert reopened["columns"] == [{"table": "Form 2A", "column": "IC_AGE", "description": "Age in years"}]
    assert reopened["feedback_history"][-2:] == [
        {"action": "approve", "feedback": "looks right"},
        {"action": "sql_execution_error", "feedback": 'bad sql: select IC_AGE from "Form 2A"'},
    ]
