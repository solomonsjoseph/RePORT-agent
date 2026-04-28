from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pandas as pd


class _HumanMessage:
    type = "human"

    def __init__(self, content: str):
        self.content = content


class _AIMessage:
    type = "ai"

    def __init__(self, content: str):
        self.content = content


def _install_stubs(action: str, suggestion: str | None = None) -> dict:
    captured: dict = {}

    langgraph_types = ModuleType("langgraph.types")

    def _interrupt(payload):
        captured["payload"] = payload
        response = {"action": action}
        if suggestion is not None:
            response["suggestion"] = suggestion
        return response

    langgraph_types.interrupt = _interrupt

    graph_message_mod = ModuleType("langgraph.graph.message")
    graph_message_mod.add_messages = lambda current, new: (current or []) + (new or [])

    messages_mod = ModuleType("langchain_core.messages")
    messages_mod.BaseMessage = object
    messages_mod.AIMessage = _AIMessage
    messages_mod.HumanMessage = _HumanMessage

    prompts_mod = ModuleType("langchain_core.prompts")
    prompts_mod.ChatPromptTemplate = object
    prompts_mod.MessagesPlaceholder = object

    sys.modules["langgraph.types"] = langgraph_types
    sys.modules["langgraph.graph.message"] = graph_message_mod
    sys.modules["langchain_core.messages"] = messages_mod
    sys.modules["langchain_core.prompts"] = prompts_mod
    return captured


def _fresh_module(action: str, suggestion: str | None = None):
    captured = _install_stubs(action=action, suggestion=suggestion)
    for mod in (
        "graph.nodes.human_review_rag_db_sql_execution",
        "graph.nodes.rag_db_qa",
        "graph.nodes.state_helpers",
        "graph.state",
    ):
        sys.modules.pop(mod, None)
    module = importlib.import_module("graph.nodes.human_review_rag_db_sql_execution")
    rag_module = importlib.import_module("graph.nodes.rag_db_qa")
    return module, rag_module, captured


def test_human_review_rag_db_sql_execution_approve_executes_prepared_sql(tmp_path) -> None:
    mod, rag, captured = _fresh_module(action="approve")
    rag.DEFAULT_RUNTIME_ROOT = tmp_path

    execution_result = ModuleType("execution_result")
    execution_result.answer = "Read-only SQL execution completed with 1 result row(s)."
    execution_result.sql = 'SELECT "AGE" FROM "Form 1A"'
    execution_result.dataframe = pd.DataFrame({"AGE": [42]})
    execution_result.source_tables = ["Form 1A"]

    class _ExecService:
        def execute_prepared_sql(self, candidate):
            assert candidate.selection_id == "sel-1"
            return execution_result

    state = {
        "messages": [],
        "output": {},
        "observations": [],
        "meta": {"thread_id": "thread-sql-review"},
        "agents": {
            "rag_db_qa": {
                "pending_column_review": {
                    "selection_id": "sel-1",
                    "question": "Subset age",
                    "tables": ["Form 1A"],
                    "columns": [{"table": "Form 1A", "column": "AGE", "description": "Age in years"}],
                    "rationale": "Age is needed.",
                    "feedback_history": [],
                    "status": "approved",
                },
                "pending_sql_candidate": {
                    "question": "Subset age",
                    "sql": 'SELECT "AGE" FROM "Form 1A"',
                    "tables": ["Form 1A"],
                    "columns": [{"table": "Form 1A", "column": "AGE", "description": "Age in years"}],
                    "selection_id": "sel-1",
                    "status": "prepared",
                },
                "last_database_question": "Subset age",
            }
        },
        "artifacts": {"datasets": {}},
    }

    updated = mod.human_review_rag_db_sql_execution_node(state, _ExecService())

    assert captured["payload"]["type"] == "human_review_rag_db_sql_execution"
    assert captured["payload"]["sql"] == 'SELECT "AGE" FROM "Form 1A"'
    assert "Read-only SQL execution completed with 1 result row" in updated["output"]["qa_response"]
    assert updated["output"]["generated_sql"] == 'SELECT "AGE" FROM "Form 1A"'
    assert "pending_sql_candidate" not in updated["agents"]["rag_db_qa"]
    assert "pending_column_review" not in updated["agents"]["rag_db_qa"]
    assert updated["agents"]["rag_db_qa"]["status"] == "done"


def test_human_review_rag_db_sql_execution_regenerate_requests_revision() -> None:
    mod, _, captured = _fresh_module(action="regenerate", suggestion="Use the cohort A final outcome table instead.")

    state = {
        "messages": [],
        "output": {},
        "observations": [],
        "meta": {},
        "agents": {
            "rag_db_qa": {
                "pending_column_review": {
                    "selection_id": "sel-2",
                    "question": "Subset age and outcome",
                    "tables": ["Form 1A"],
                    "columns": [{"table": "Form 1A", "column": "AGE", "description": "Age in years"}],
                    "rationale": "Age is needed.",
                    "feedback_history": [],
                    "status": "approved",
                },
                "pending_sql_candidate": {
                    "question": "Subset age and outcome",
                    "sql": 'SELECT "AGE" FROM "Form 1A"',
                    "tables": ["Form 1A"],
                    "columns": [{"table": "Form 1A", "column": "AGE", "description": "Age in years"}],
                    "selection_id": "sel-2",
                    "status": "prepared",
                },
                "last_database_question": "Subset age and outcome",
            }
        },
        "artifacts": {"datasets": {}},
    }

    updated = mod.human_review_rag_db_sql_execution_node(state, object())

    assert captured["payload"]["type"] == "human_review_rag_db_sql_execution"
    assert updated["agents"]["rag_db_qa"]["pending_column_review"]["status"] == "needs_revision"
    assert updated["agents"]["rag_db_qa"]["thread_status"] == "awaiting_column_review"
    assert "pending_sql_candidate" not in updated["agents"]["rag_db_qa"]
    assert updated["messages"] == []
    history = updated["agents"]["rag_db_qa"]["pending_column_review"]["feedback_history"]
    assert history[-1]["action"] == "regenerate"
    assert "Human requested SQL regeneration." in history[-1]["feedback"]
    assert "Use the cohort A final outcome table instead." in history[-1]["feedback"]


def test_human_review_rag_db_sql_execution_regenerate_preserves_feedback_history_for_intent_update() -> None:
    mod, _, _ = _fresh_module(action="regenerate", suggestion="Use the cohort A final outcome table instead.")

    state = {
        "messages": [],
        "output": {},
        "observations": [],
        "meta": {},
        "agents": {
            "rag_db_qa": {
                "pending_column_review": {
                    "selection_id": "sel-3",
                    "goal_text": "subset age and outcome among index cases",
                    "question": "Subset age and outcome",
                    "tables": ["Form 1A"],
                    "columns": [{"table": "Form 1A", "column": "AGE", "description": "Age in years"}],
                    "rationale": "Age is needed.",
                    "feedback_history": [{"timestamp": "2026-04-28T12:00:00+00:00", "feedback": "Keep age."}],
                    "status": "approved",
                },
                "pending_sql_candidate": {
                    "question": "Subset age and outcome",
                    "sql": 'SELECT "AGE" FROM "Form 1A"',
                    "tables": ["Form 1A"],
                    "columns": [{"table": "Form 1A", "column": "AGE", "description": "Age in years"}],
                    "selection_id": "sel-3",
                    "status": "prepared",
                },
                "last_database_question": "Subset age and outcome",
            }
        },
        "artifacts": {"datasets": {}},
    }

    updated = mod.human_review_rag_db_sql_execution_node(state, service=object())

    history = updated["agents"]["rag_db_qa"]["pending_column_review"]["feedback_history"]
    assert len(history) == 2
    assert history[0]["feedback"] == "Keep age."
    assert "Use the cohort A final outcome table instead." in history[-1]["feedback"]
    assert updated["agents"]["rag_db_qa"]["pending_column_review"]["goal_text"] == "subset age and outcome among index cases"


def test_human_review_rag_db_sql_execution_approve_clears_pending_state_on_execution_error() -> None:
    mod, _, captured = _fresh_module(action="approve")

    class BinderException(Exception):
        pass

    class _ExecService:
        def execute_prepared_sql(self, candidate):
            assert candidate.selection_id == "sel-error"
            raise BinderException('Referenced column "SEX" not found in FROM clause')

    state = {
        "messages": [],
        "output": {
            "generated_sql": 'SELECT "AGE", "SEX" FROM "Form 1A"',
            "prepared_sql_candidate": {
                "question": "Subset age and sex",
                "sql": 'SELECT "AGE", "SEX" FROM "Form 1A"',
                "tables": ["Form 1A"],
                "columns": [
                    {"table": "Form 1A", "column": "AGE", "description": "Age in years"},
                    {"table": "Form 1A", "column": "SEX", "description": "Sex at enrollment"},
                ],
                "selection_id": "sel-error",
                "status": "prepared",
            },
        },
        "observations": [],
        "meta": {},
        "agents": {
            "rag_db_qa": {
                "pending_column_review": {
                    "selection_id": "sel-error",
                    "question": "Subset age and sex",
                    "tables": ["Form 1A"],
                    "columns": [
                        {"table": "Form 1A", "column": "AGE", "description": "Age in years"},
                        {"table": "Form 1A", "column": "SEX", "description": "Sex at enrollment"},
                    ],
                    "rationale": "Need demographic columns.",
                    "feedback_history": [],
                    "status": "approved",
                },
                "pending_sql_candidate": {
                    "question": "Subset age and sex",
                    "sql": 'SELECT "AGE", "SEX" FROM "Form 1A"',
                    "tables": ["Form 1A"],
                    "columns": [
                        {"table": "Form 1A", "column": "AGE", "description": "Age in years"},
                        {"table": "Form 1A", "column": "SEX", "description": "Sex at enrollment"},
                    ],
                    "selection_id": "sel-error",
                    "status": "prepared",
                },
                "last_database_question": "Subset age and sex",
            }
        },
        "artifacts": {"datasets": {}},
    }

    updated = mod.human_review_rag_db_sql_execution_node(state, _ExecService())

    assert captured["payload"]["type"] == "human_review_rag_db_sql_execution"
    assert updated["output"]["error"] == {
        "category": "db_rag_sql",
        "type": "BinderException",
        "message": 'Referenced column "SEX" not found in FROM clause',
    }
    assert "workflow stopped" in updated["output"]["qa_response"]
    assert "pending_sql_candidate" not in updated["agents"]["rag_db_qa"]
    assert "pending_column_review" not in updated["agents"]["rag_db_qa"]
    assert updated["agents"]["rag_db_qa"]["status"] == "error"
    assert updated["agents"]["rag_db_qa"]["active_thread"] is False
