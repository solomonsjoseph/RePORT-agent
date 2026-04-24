from __future__ import annotations

import importlib
import sys
from types import ModuleType
from types import SimpleNamespace

import pandas as pd


class _AIMessage:
    type = "ai"

    def __init__(self, content: str):
        self.content = content


class _HumanMessage:
    type = "human"

    def __init__(self, content: str):
        self.content = content


class _MessagesPlaceholder:
    def __init__(self, variable_name: str, optional: bool = False) -> None:
        self.variable_name = variable_name
        self.optional = optional


class _PromptTemplate:
    @staticmethod
    def from_messages(messages):
        return messages


def _install_stubs() -> None:
    messages_mod = ModuleType("langchain_core.messages")
    messages_mod.BaseMessage = object
    messages_mod.AIMessage = _AIMessage
    messages_mod.HumanMessage = _HumanMessage

    prompts_mod = ModuleType("langchain_core.prompts")
    prompts_mod.ChatPromptTemplate = _PromptTemplate
    prompts_mod.MessagesPlaceholder = _MessagesPlaceholder

    langchain_core_mod = ModuleType("langchain_core")
    langchain_core_mod.messages = messages_mod
    langchain_core_mod.prompts = prompts_mod

    graph_message_mod = ModuleType("langgraph.graph.message")
    graph_message_mod.add_messages = lambda current, new: (current or []) + (new or [])

    sys.modules["langchain_core"] = langchain_core_mod
    sys.modules["langchain_core.messages"] = messages_mod
    sys.modules["langchain_core.prompts"] = prompts_mod
    sys.modules["langgraph.graph.message"] = graph_message_mod


def _fresh_rag_module():
    _install_stubs()
    for mod in (
        "graph.nodes.rag_db_qa",
        "graph.nodes.tool_routing",
        "graph.nodes.state_helpers",
        "graph.state",
    ):
        sys.modules.pop(mod, None)
    return importlib.import_module("graph.nodes.rag_db_qa")


def test_rag_db_qa_requires_openai_or_anthropic_provider() -> None:
    rag = _fresh_rag_module()

    state = {
        "messages": [_HumanMessage("How many participants are in cohort A?")],
        "output": {},
        "observations": [],
        "meta": {},
        "agents": {"rag_db_qa": {}},
        "artifacts": {},
    }

    updated = rag.rag_db_qa_node(state, llm=object(), provider="gemini", service=None)

    assert "OpenAI or Anthropic" in updated["output"]["qa_response"]
    assert updated["agents"]["rag_db_qa"]["status"] == "done"


def test_rag_db_qa_returns_guided_init_message_when_assets_are_missing() -> None:
    rag = _fresh_rag_module()

    class _Service:
        def readiness(self):
            return {
                "ready": False,
                "message": "DB-RAG assets are not initialized. Copy source data into local_data/db_rag_source/ and run python -m db_rag.build_index --rebuild.",
            }

    state = {
        "messages": [_HumanMessage("How many participants are in cohort A?")],
        "output": {},
        "observations": [],
        "meta": {},
        "agents": {"rag_db_qa": {}},
        "artifacts": {},
    }

    updated = rag.rag_db_qa_node(state, llm=object(), provider="openai", service=_Service())

    assert "python -m db_rag.build_index --rebuild" in updated["output"]["qa_response"]
    assert updated["agents"]["rag_db_qa"]["status"] == "done"


def test_rag_db_metadata_qa_answers_without_pending_sql_state() -> None:
    rag = _fresh_rag_module()

    class _Service:
        def readiness(self):
            return {"ready": True, "message": ""}

        def retrieve_context(self, question):
            assert "cohort A" in question
            return SimpleNamespace(
                tables=[SimpleNamespace(table="Form 1A", text="table summary")],
                columns=[SimpleNamespace(table="Form 1A", column="SEX", text="column summary")],
                table_names=["Form 1A"],
                column_names=["SEX"],
            )

        def answer_from_context(self, question, context):
            assert "cohort A" in question
            assert context.table_names == ["Form 1A"]
            return SimpleNamespace(
                answer="The relevant database concepts are cohort membership and sex in the screening and enrollment forms.",
                needs_sql=False,
                rationale="metadata answer",
                relevant_tables=["Form 1A"],
                relevant_columns=["SEX"],
            )

    state = {
        "messages": [_HumanMessage("How many male participants are in cohort A?")],
        "output": {},
        "observations": [],
        "meta": {},
        "agents": {"rag_db_qa": {}},
        "artifacts": {},
    }

    updated = rag.rag_db_qa_node(state, llm=object(), provider="openai", service=_Service())

    assert "cohort membership and sex" in updated["output"]["qa_response"]
    assert "pending_sql_candidate" not in updated["agents"]["rag_db_qa"]
    assert "pending_column_review" not in updated["agents"]["rag_db_qa"]
    assert "awaiting_user_clarification" not in updated["meta"]


def test_rag_db_qa_treats_bare_yes_as_fresh_message_without_executing_sql() -> None:
    rag = _fresh_rag_module()
    calls: list[str] = []

    class _Service:
        def readiness(self):
            return {"ready": True, "message": ""}

        def retrieve_context(self, question):
            calls.append("retrieve_context")
            assert question == "yes"
            return SimpleNamespace(
                tables=[SimpleNamespace(table="Form 1A", text="table summary")],
                columns=[SimpleNamespace(table="Form 1A", column="SEX", text="column summary")],
                table_names=["Form 1A"],
                column_names=["SEX"],
            )

        def answer_from_context(self, question, context):
            calls.append("answer_from_context")
            assert question == "yes"
            assert context.table_names == ["Form 1A"]
            return SimpleNamespace(
                answer="Fresh metadata answer from the new message.",
                needs_sql=False,
                rationale="metadata answer",
                relevant_tables=["Form 1A"],
                relevant_columns=["SEX"],
            )

        def execute_prepared_sql(self, candidate):
            raise AssertionError("execute_prepared_sql should not run for bare yes")

    state = {
        "messages": [_HumanMessage("yes")],
        "output": {},
        "observations": [],
        "meta": {},
        "agents": {
            "rag_db_qa": {
                "pending_sql_candidate": {
                    "question": "How many male participants are in cohort A?",
                    "sql": 'SELECT "AGE", "SEX" FROM "Form 1A"',
                    "tables": ["Form 1A"],
                    "columns": [
                        {"table": "Form 1A", "column": "AGE", "description": "Age in years"},
                        {"table": "Form 1A", "column": "SEX", "description": "Sex at enrollment"},
                    ],
                    "selection_id": "sel-bare-yes",
                    "status": "prepared",
                }
            }
        },
        "artifacts": {},
    }

    updated = rag.rag_db_qa_node(state, llm=object(), provider="openai", service=_Service())

    assert updated["output"]["qa_response"] == "Fresh metadata answer from the new message."
    assert "pending_sql_candidate" not in updated["agents"]["rag_db_qa"]
    assert "pending_column_review" not in updated["agents"]["rag_db_qa"]
    assert calls == ["retrieve_context", "answer_from_context"]


def test_rag_db_qa_decline_with_pending_sql_candidate_skips_execution_and_retrieval() -> None:
    rag = _fresh_rag_module()

    class _Service:
        def readiness(self):
            return {"ready": True, "message": ""}

        def retrieve_context(self, question):
            raise AssertionError("retrieve_context should not run for an explicit decline")

        def execute_prepared_sql(self, candidate):
            raise AssertionError("execute_prepared_sql should not run for an explicit decline")

    state = {
        "messages": [
            _HumanMessage("How many male participants are in cohort A?"),
            _AIMessage('SELECT "AGE", "SEX" FROM "Form 1A"\n\nDo you want me to run the read-only SQL?'),
            _HumanMessage("no"),
        ],
        "output": {},
        "observations": [],
        "meta": {},
        "agents": {
            "rag_db_qa": {
                "pending_column_review": {
                    "selection_id": "sel-approved",
                    "question": "How many male participants are in cohort A?",
                    "tables": ["Form 1A"],
                    "columns": [
                        {"table": "Form 1A", "column": "AGE", "description": "Age in years"},
                        {"table": "Form 1A", "column": "SEX", "description": "Sex at enrollment"},
                    ],
                    "rationale": "approved columns",
                    "feedback_history": [],
                    "status": "approved",
                },
                "pending_sql_candidate": {
                    "question": "How many male participants are in cohort A?",
                    "sql": 'SELECT "AGE", "SEX" FROM "Form 1A"',
                    "tables": ["Form 1A"],
                    "columns": [
                        {"table": "Form 1A", "column": "AGE", "description": "Age in years"},
                        {"table": "Form 1A", "column": "SEX", "description": "Sex at enrollment"},
                    ],
                    "selection_id": "sel-decline",
                    "status": "prepared",
                },
                "last_database_question": "How many male participants are in cohort A?",
            }
        },
        "artifacts": {},
    }

    updated = rag.rag_db_qa_node(state, llm=object(), provider="openai", service=_Service())

    assert updated["output"]["qa_response"] == "Okay. I will not run the read-only SQL."
    assert updated["agents"]["rag_db_qa"]["pending_column_review"]["status"] == "approved"
    assert "pending_sql_candidate" not in updated["agents"]["rag_db_qa"]
    assert updated["agents"]["rag_db_qa"]["last_database_question"] == "How many male participants are in cohort A?"


def test_rag_db_qa_combines_stale_qa_followup_context_when_taking_over() -> None:
    rag = _fresh_rag_module()
    captured: dict[str, str] = {}

    class _Service:
        def readiness(self):
            return {"ready": True, "message": ""}

        def retrieve_context(self, question):
            captured["question"] = question
            return SimpleNamespace(
                tables=[SimpleNamespace(table="Form 1A", text="table summary")],
                columns=[SimpleNamespace(table="Form 1A", column="AGE", text="column summary")],
                table_names=["Form 1A"],
                column_names=["AGE"],
            )

        def answer_from_context(self, question, context):
            captured["question"] = question
            assert context.table_names == ["Form 1A"]
            return SimpleNamespace(
                answer="The relevant database concepts are available in the local metadata store.",
                needs_sql=False,
                rationale="metadata answer",
                relevant_tables=["Form 1A"],
                relevant_columns=["AGE"],
            )

    state = {
        "messages": [
            _HumanMessage("Find the matching records"),
            _AIMessage("Which source should I use?"),
            _HumanMessage("Use the local metadata store"),
        ],
        "output": {"qa_response": "Which source should I use?"},
        "observations": [],
        "meta": {
            "awaiting_user_clarification": True,
            "clarification_return_node": "qa",
            "clarification_kind": "qa_followup",
            "pending_question": "Find the matching records",
        },
        "agents": {"rag_db_qa": {}},
        "artifacts": {"datasets": {}},
    }

    updated = rag.rag_db_qa_node(state, llm=object(), provider="openai", service=_Service())

    assert captured["question"] == "Find the matching records\n\nUser clarification: Use the local metadata store"
    assert "local metadata store" in updated["output"]["qa_response"]
    assert "pending_column_review" not in updated["agents"]["rag_db_qa"]
    assert "pending_sql_candidate" not in updated["agents"]["rag_db_qa"]


def test_rag_db_sql_needed_request_creates_column_review_without_sql() -> None:
    rag = _fresh_rag_module()
    calls: list[str] = []

    class _Service:
        def readiness(self):
            return {"ready": True, "message": ""}

        def retrieve_context(self, question):
            calls.append("retrieve_context")
            assert "subset age and sex" in question
            return SimpleNamespace(
                tables=[SimpleNamespace(table="Form 1A", text="table summary")],
                columns=[
                    SimpleNamespace(table="Form 1A", column="AGE", text="age summary"),
                    SimpleNamespace(table="Form 1A", column="SEX", text="sex summary"),
                ],
                table_names=["Form 1A"],
                column_names=["AGE", "SEX"],
            )

        def answer_from_context(self, question, context):
            calls.append("answer_from_context")
            assert context.column_names == ["AGE", "SEX"]
            return SimpleNamespace(
                answer="This request needs row-level SQL to produce the exact subset.",
                needs_sql=True,
                rationale="subset request",
                relevant_tables=["Form 1A"],
                relevant_columns=["AGE", "SEX"],
            )

        def prepare_column_selection(self, question, context, feedback_history=None):
            calls.append("prepare_column_selection")
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

        def prepare_sql_candidate(self, question, approved_selection):
            raise AssertionError("prepare_sql_candidate should not run before review approval")

    state = {
        "messages": [_HumanMessage("Help me subset age and sex for the matching participants")],
        "output": {},
        "observations": [],
        "meta": {},
        "agents": {"rag_db_qa": {}},
        "artifacts": {"datasets": {}},
    }

    updated = rag.rag_db_qa_node(state, llm=object(), provider="openai", service=_Service())

    review = updated["agents"]["rag_db_qa"]["pending_column_review"]
    assert review["status"] == "awaiting_review"
    assert review["selection_id"] == "sel-1"
    assert review["tables"] == ["Form 1A"]
    assert [column["column"] for column in review["columns"]] == ["AGE", "SEX"]
    assert "pending_sql_candidate" not in updated["agents"]["rag_db_qa"]
    assert "generated only after approval" in updated["output"]["qa_response"]
    assert calls == ["retrieve_context", "answer_from_context", "prepare_column_selection"]


def test_rag_db_approved_column_review_generates_sql_candidate() -> None:
    rag = _fresh_rag_module()

    class _Service:
        def readiness(self):
            return {"ready": True, "message": ""}

        def prepare_sql_candidate(self, question, approved_selection):
            assert question == "Help me subset age and sex for the matching participants"
            assert approved_selection.status == "approved"
            assert approved_selection.selection_id == "sel-approved"
            return SimpleNamespace(
                question=question,
                sql='SELECT "AGE", "SEX" FROM "Form 1A"',
                tables=["Form 1A"],
                columns=[
                    {"table": "Form 1A", "column": "AGE", "description": "Age in years"},
                    {"table": "Form 1A", "column": "SEX", "description": "Sex at enrollment"},
                ],
                selection_id="sel-approved",
                status="prepared",
            )

    state = {
        "messages": [
            _HumanMessage("Help me subset age and sex for the matching participants"),
        ],
        "output": {},
        "observations": [],
        "meta": {},
        "agents": {
            "rag_db_qa": {
                "pending_column_review": {
                    "selection_id": "sel-approved",
                    "question": "Help me subset age and sex for the matching participants",
                    "tables": ["Form 1A"],
                    "columns": [
                        {"table": "Form 1A", "column": "AGE", "description": "Age in years"},
                        {"table": "Form 1A", "column": "SEX", "description": "Sex at enrollment"},
                    ],
                    "rationale": "approved columns",
                    "feedback_history": [],
                    "status": "approved",
                }
            }
        },
        "artifacts": {"datasets": {}},
    }

    updated = rag.rag_db_qa_node(state, llm=object(), provider="openai", service=_Service())

    candidate = updated["agents"]["rag_db_qa"]["pending_sql_candidate"]
    assert candidate["status"] == "prepared"
    assert candidate["selection_id"] == "sel-approved"
    assert candidate["sql"] == 'SELECT "AGE", "SEX" FROM "Form 1A"'
    assert "run the read-only SQL" in updated["output"]["qa_response"]
    assert 'SELECT "AGE", "SEX" FROM "Form 1A"' in updated["output"]["qa_response"]


def test_rag_db_executes_prepared_sql_after_explicit_confirmation() -> None:
    rag = _fresh_rag_module()

    class _Service:
        def readiness(self):
            return {"ready": True, "message": ""}

        def execute_prepared_sql(self, candidate):
            assert candidate.selection_id == "sel-prepared"
            assert candidate.sql == 'SELECT "AGE", "SEX" FROM "Form 1A"'
            return SimpleNamespace(
                answer="Read-only SQL execution completed with 1 result row(s).",
                sql=candidate.sql,
                dataframe=pd.DataFrame({"AGE": [42], "SEX": ["Male"]}),
                source_tables=["Form 1A"],
            )

    state = {
        "messages": [
            _HumanMessage("Help me subset age and sex for the matching participants"),
            _AIMessage('SELECT "AGE", "SEX" FROM "Form 1A"\n\nDo you want me to run the read-only SQL?'),
            _HumanMessage("run the prepared sql"),
        ],
        "output": {},
        "observations": [],
        "meta": {"thread_id": "thread-execute"},
        "agents": {
            "rag_db_qa": {
                "pending_sql_candidate": {
                    "question": "Help me subset age and sex for the matching participants",
                    "sql": 'SELECT "AGE", "SEX" FROM "Form 1A"',
                    "tables": ["Form 1A"],
                    "columns": [
                        {"table": "Form 1A", "column": "AGE", "description": "Age in years"},
                        {"table": "Form 1A", "column": "SEX", "description": "Sex at enrollment"},
                    ],
                    "selection_id": "sel-prepared",
                    "status": "prepared",
                },
                "last_database_question": "Help me subset age and sex for the matching participants",
            }
        },
        "artifacts": {"datasets": {}},
    }

    updated = rag.rag_db_qa_node(state, llm=object(), provider="openai", service=_Service())

    assert "Read-only SQL execution completed with 1 result row" in updated["output"]["qa_response"]
    assert updated["agents"]["rag_db_qa"]["status"] == "done"
    assert "pending_sql_candidate" not in updated["agents"]["rag_db_qa"]


def test_rag_db_execution_persists_subset_artifact_with_db_rag_provenance(
    tmp_path,
    monkeypatch,
) -> None:
    rag = _fresh_rag_module()
    monkeypatch.setattr(rag, "DEFAULT_RUNTIME_ROOT", tmp_path)

    class _Service:
        def readiness(self):
            return {"ready": True, "message": ""}

        def execute_prepared_sql(self, candidate):
            assert candidate.selection_id == "sel-approved"
            return SimpleNamespace(
                answer="Read-only SQL execution completed with 2 result row(s).",
                sql=candidate.sql,
                dataframe=pd.DataFrame({"AGE": [42, 37]}),
                source_tables=["Form 1A"],
            )

    feedback_history = [
        {"role": "user", "content": "Keep only AGE for the first pass."},
        {"role": "assistant", "content": "Updated the approved column selection."},
    ]
    state = {
        "messages": [
            _HumanMessage("Help me subset age for the matching participants"),
            _AIMessage('SELECT "AGE" FROM "Form 1A"\n\nDo you want me to run the read-only SQL?'),
            _HumanMessage("run the prepared sql"),
        ],
        "output": {},
        "observations": [],
        "meta": {"thread_id": "thread-persist"},
        "agents": {
            "rag_db_qa": {
                "pending_column_review": {
                    "selection_id": "sel-approved",
                    "question": "Help me subset age for the matching participants",
                    "tables": ["Form 1A"],
                    "columns": [
                        {"table": "Form 1A", "column": "AGE", "description": "Age in years"},
                    ],
                    "rationale": "approved columns",
                    "feedback_history": feedback_history,
                    "status": "approved",
                },
                "pending_sql_candidate": {
                    "question": "Help me subset age for the matching participants",
                    "sql": 'SELECT "AGE" FROM "Form 1A"',
                    "tables": ["Form 1A"],
                    "columns": [
                        {"table": "Form 1A", "column": "AGE", "description": "Age in years"},
                    ],
                    "selection_id": "sel-approved",
                    "status": "prepared",
                },
                "last_database_question": "Help me subset age for the matching participants",
            }
        },
        "artifacts": {"datasets": {}},
    }

    updated = rag.rag_db_qa_node(state, llm=object(), provider="openai", service=_Service())

    dataset_id = updated["artifacts"]["active_dataset_id"]
    artifact = updated["artifacts"]["datasets"][dataset_id]

    assert artifact["kind"] == "subset"
    assert artifact["provenance"]["source"] == "db_rag_sql"
    assert artifact["provenance"]["feedback_history"] == feedback_history
    assert artifact["provenance"]["selected_columns"][0]["column"] == "AGE"
    assert "pending_sql_candidate" not in updated["agents"]["rag_db_qa"]
    assert "pending_column_review" not in updated["agents"]["rag_db_qa"]
    assert dataset_id in updated["output"]["qa_response"]


def test_rag_db_treats_contentful_followup_as_new_question_when_sql_candidate_exists() -> None:
    rag = _fresh_rag_module()
    captured: dict[str, str] = {}

    class _Service:
        def readiness(self):
            return {"ready": True, "message": ""}

        def retrieve_context(self, question):
            captured["question"] = question
            return SimpleNamespace(
                tables=[SimpleNamespace(table="Form 1A", text="table summary")],
                columns=[
                    SimpleNamespace(table="Form 1A", column="AGE", text="age summary"),
                    SimpleNamespace(table="Form 1A", column="SEX", text="sex summary"),
                ],
                table_names=["Form 1A"],
                column_names=["AGE", "SEX"],
            )

        def answer_from_context(self, question, context):
            assert context.column_names == ["AGE", "SEX"]
            return SimpleNamespace(
                answer="This new request needs SQL, so I prepared a fresh column review.",
                needs_sql=True,
                rationale="new subset request",
                relevant_tables=["Form 1A"],
                relevant_columns=["AGE", "SEX"],
            )

        def prepare_column_selection(self, question, context, feedback_history=None):
            return SimpleNamespace(
                selection_id="sel-fresh",
                question=question,
                tables=["Form 1A"],
                columns=[
                    {"table": "Form 1A", "column": "AGE", "description": "Age in years"},
                    {"table": "Form 1A", "column": "SEX", "description": "Sex at enrollment"},
                ],
                rationale="fresh review candidate",
                feedback_history=[],
                status="awaiting_review",
            )

        def execute_prepared_sql(self, candidate):
            raise AssertionError("execute_prepared_sql should not run for a contentful follow-up")

    state = {
        "messages": [
            _HumanMessage("Give me overview of the database in the system"),
            _AIMessage('SELECT "COUNT" FROM "Form 1A"\n\nDo you want me to run the read-only SQL?'),
            _HumanMessage("Help me subset age and sex for the matching participants"),
        ],
        "output": {},
        "observations": [],
        "meta": {},
        "agents": {
            "rag_db_qa": {
                "pending_sql_candidate": {
                    "question": "Give me overview of the database in the system",
                    "sql": 'SELECT COUNT(*) FROM "Form 1A"',
                    "tables": ["Form 1A"],
                    "columns": [],
                    "selection_id": "sel-old",
                    "status": "prepared",
                },
                "last_database_question": "Give me overview of the database in the system",
            }
        },
        "artifacts": {"datasets": {}},
    }

    updated = rag.rag_db_qa_node(state, llm=object(), provider="openai", service=_Service())

    assert captured["question"] == "Help me subset age and sex for the matching participants"
    assert updated["agents"]["rag_db_qa"]["pending_column_review"]["selection_id"] == "sel-fresh"
    assert updated["agents"]["rag_db_qa"]["last_database_question"] == captured["question"]


def test_rag_db_qa_surfaces_sql_execution_error() -> None:
    rag = _fresh_rag_module()

    class _Service:
        def readiness(self):
            return {"ready": True, "message": ""}

        def execute_prepared_sql(self, candidate):
            assert candidate.selection_id == "sel-error"
            raise RuntimeError('Parser Error: syntax error at or near ")"')

    state = {
        "messages": [
            _HumanMessage("What does this database include?"),
            _AIMessage('SELECT "AGE" FROM "Form 1A"\n\nDo you want me to run the read-only SQL?'),
            _HumanMessage("execute the query"),
        ],
        "output": {},
        "observations": [],
        "meta": {},
        "agents": {
            "rag_db_qa": {
                "pending_sql_candidate": {
                    "question": "What does this database include?",
                    "sql": 'SELECT "AGE" FROM "Form 1A"',
                    "tables": ["Form 1A"],
                    "columns": [{"table": "Form 1A", "column": "AGE", "description": "Age in years"}],
                    "selection_id": "sel-error",
                    "status": "prepared",
                },
                "last_database_question": "What does this database include?",
            }
        },
        "artifacts": {"datasets": {}},
    }

    updated = rag.rag_db_qa_node(state, llm=object(), provider="openai", service=_Service())

    assert "DB-RAG SQL execution failed" in updated["output"]["qa_response"]
    assert updated["output"]["error"] == {
        "category": "db_rag_sql",
        "type": "RuntimeError",
        "message": 'Parser Error: syntax error at or near ")"',
    }
    assert updated["agents"]["rag_db_qa"]["status"] == "error"
    assert updated["agents"]["rag_db_qa"]["pending_sql_candidate"]["selection_id"] == "sel-error"
