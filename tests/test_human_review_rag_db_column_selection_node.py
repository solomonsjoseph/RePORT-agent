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
                    "goal_text": "subset sex among index cases",
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
    assert captured["payload"]["goal_text"] == "subset sex among index cases"
    assert captured["payload"]["selection_id"] == "sel-1"
    assert updated["agents"]["rag_db_qa"]["pending_column_review"]["status"] == "approved"
    assert updated["agents"]["rag_db_qa"]["pending_column_review"]["selection_id"] == "sel-1"
    events = updated["artifacts"]["conversation_events"]
    assert events[-1]["type"] == "review_decision"
    assert events[-1]["review_kind"] == "rag_db_column_selection"
    assert events[-1]["decision"] == "approve"


def test_human_review_rag_db_column_selection_regeneration_appends_feedback() -> None:
    mod, _ = _fresh_module(action="regenerate", feedback="Include the age column too.")

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
    assert "Human requested regeneration." in review["feedback_history"][-1]["feedback"]
    assert "Include the age column too." in review["feedback_history"][-1]["feedback"]
    assert "timestamp" in review["feedback_history"][-1]
    assert updated["artifacts"]["conversation_events"][-1]["decision"] == "regenerate"


def test_human_review_rag_db_column_selection_unknown_action_keeps_explanation_with_feedback() -> None:
    mod, _ = _fresh_module(action="maybe later", feedback="focus on age and sex")

    state = {
        "agents": {
            "rag_db_qa": {
                "pending_column_review": {
                    "selection_id": "sel-4",
                    "question": "Which tables and columns should be used?",
                    "tables": ["form_d"],
                    "columns": [{"table": "form_d", "column": "sex", "description": "Sex"}],
                    "rationale": "Need a broader selection.",
                    "feedback_history": [],
                    "status": "awaiting_review",
                }
            }
        }
    }

    updated = mod.human_review_rag_db_column_selection_node(state)

    entry = updated["agents"]["rag_db_qa"]["pending_column_review"]["feedback_history"][-1]
    assert updated["agents"]["rag_db_qa"]["pending_column_review"]["status"] == "needs_revision"
    assert "unsupported action" in entry["feedback"]
    assert "Additional context: focus on age and sex" in entry["feedback"]
