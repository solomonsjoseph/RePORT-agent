from __future__ import annotations

import importlib
import sys
from types import ModuleType


def _install_stubs(action: str, feedback: str | None = None) -> dict:
    captured: dict = {}

    langgraph_types = ModuleType("langgraph.types")

    def _interrupt(payload):
        captured["payload"] = payload
        response = {"action": action}
        if feedback is not None:
            response["feedback"] = feedback
        return response

    langgraph_types.interrupt = _interrupt

    graph_message_mod = ModuleType("langgraph.graph.message")
    graph_message_mod.add_messages = lambda current, new: (current or []) + (new or [])

    messages_mod = ModuleType("langchain_core.messages")
    messages_mod.BaseMessage = object

    sys.modules["langgraph.types"] = langgraph_types
    sys.modules["langgraph.graph.message"] = graph_message_mod
    sys.modules["langchain_core.messages"] = messages_mod
    return captured


def _fresh_module(action: str, feedback: str | None = None):
    captured = _install_stubs(action=action, feedback=feedback)
    for mod in (
        "graph.nodes.human_review_rag_db_column_selection",
        "graph.nodes.state_helpers",
        "graph.state",
    ):
        sys.modules.pop(mod, None)
    module = importlib.import_module("graph.nodes.human_review_rag_db_column_selection")
    return module, captured


def test_human_review_rag_db_column_selection_approve_marks_selection_approved() -> None:
    mod, captured = _fresh_module(action="approve")

    state = {
        "agents": {
            "rag_db_qa": {
                "pending_column_review": {
                    "selection_id": "sel-1",
                    "question": "Which tables and columns should be used?",
                    "tables": ["form_a"],
                    "columns": [{"table": "form_a", "column": "sex", "description": "Sex"}],
                    "rationale": "Sex is required.",
                    "feedback_history": [],
                    "status": "awaiting_review",
                }
            }
        }
    }

    updated = mod.human_review_rag_db_column_selection_node(state)

    assert captured["payload"]["type"] == "human_review_rag_db_column_selection"
    assert captured["payload"]["selection_id"] == "sel-1"
    assert updated["agents"]["rag_db_qa"]["pending_column_review"]["status"] == "approved"
    assert updated["agents"]["rag_db_qa"]["pending_column_review"]["selection_id"] == "sel-1"


def test_human_review_rag_db_column_selection_revision_appends_feedback() -> None:
    mod, _ = _fresh_module(action="revise", feedback="Include the age column too.")

    state = {
        "agents": {
            "rag_db_qa": {
                "pending_column_review": {
                    "selection_id": "sel-2",
                    "question": "Which columns should be used?",
                    "tables": ["form_b"],
                    "columns": [{"table": "form_b", "column": "age", "description": "Age"}],
                    "rationale": "Age appears relevant.",
                    "feedback_history": [{"timestamp": "2026-04-23T15:00:00+00:00", "feedback": "Initial pass."}],
                    "status": "awaiting_review",
                }
            }
        }
    }

    updated = mod.human_review_rag_db_column_selection_node(state)

    review = updated["agents"]["rag_db_qa"]["pending_column_review"]
    assert review["status"] == "needs_revision"
    assert len(review["feedback_history"]) == 2
    assert review["feedback_history"][-1]["feedback"] == "Include the age column too."
    assert "timestamp" in review["feedback_history"][-1]


def test_human_review_rag_db_column_selection_cancel_clears_pending_sql_candidate() -> None:
    mod, _ = _fresh_module(action="cancel")

    state = {
        "agents": {
            "rag_db_qa": {
                "pending_column_review": {
                    "selection_id": "sel-3",
                    "question": "Which tables and columns should be used?",
                    "tables": ["form_c"],
                    "columns": [],
                    "rationale": "Need a reset.",
                    "feedback_history": [],
                    "status": "awaiting_review",
                },
                "pending_sql_candidate": {"sql": "SELECT 1"},
            }
        }
    }

    updated = mod.human_review_rag_db_column_selection_node(state)

    review = updated["agents"]["rag_db_qa"]["pending_column_review"]
    assert review["status"] == "cancelled"
    assert "pending_sql_candidate" not in updated["agents"]["rag_db_qa"]
