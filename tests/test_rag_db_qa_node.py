from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pandas as pd


class _AIMessage:
    type = "ai"

    def __init__(self, content: str):
        self.content = content


class _HumanMessage:
    type = "human"

    def __init__(self, content: str):
        self.content = content


def _install_stubs() -> None:
    messages_mod = ModuleType("langchain_core.messages")
    messages_mod.BaseMessage = object
    messages_mod.AIMessage = _AIMessage
    messages_mod.HumanMessage = _HumanMessage

    graph_message_mod = ModuleType("langgraph.graph.message")
    graph_message_mod.add_messages = lambda current, new: (current or []) + (new or [])

    sys.modules["langchain_core.messages"] = messages_mod
    sys.modules["langgraph.graph.message"] = graph_message_mod


def _fresh_rag_module():
    _install_stubs()
    for mod in (
        "graph.nodes.rag_db_qa",
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
                "message": "DB-RAG assets are not initialized. Copy source data into local_data/db_rag_source/ and run python -m db_rag.bootstrap --rebuild.",
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

    assert "python -m db_rag.bootstrap --rebuild" in updated["output"]["qa_response"]
    assert updated["agents"]["rag_db_qa"]["status"] == "done"


def test_rag_db_qa_answers_then_offers_sql_followup() -> None:
    rag = _fresh_rag_module()

    class _Service:
        def readiness(self):
            return {"ready": True, "message": ""}

        def answer_question(self, question):
            assert "cohort A" in question
            return {
                "answer": "The relevant database concepts are cohort membership and sex in the screening and enrollment forms.",
                "retrieval_summary": {"tables": ["Form 1A"]},
            }

    state = {
        "messages": [_HumanMessage("How many male participants are in cohort A?")],
        "output": {},
        "observations": [],
        "meta": {},
        "agents": {"rag_db_qa": {}},
        "artifacts": {},
    }

    updated = rag.rag_db_qa_node(state, llm=object(), provider="openai", service=_Service())

    assert updated["meta"]["awaiting_user_clarification"] is True
    assert updated["meta"]["clarification_return_node"] == "rag_db_qa"
    assert updated["meta"]["clarification_kind"] == "rag_db_sql_offer"
    assert updated["agents"]["rag_db_qa"]["pending_sql_offer"] is True
    assert "Do you want me to extract a read-only subset" in updated["output"]["qa_response"]


def test_rag_db_qa_executes_sql_after_positive_confirmation() -> None:
    rag = _fresh_rag_module()

    class _Service:
        def readiness(self):
            return {"ready": True, "message": ""}

        def execute_sql_flow(self, question):
            assert "male participants" in question
            return {
                "answer": "I extracted 12 matching records.",
                "sql": 'SELECT * FROM "Form 1A" WHERE SEX = \'Male\'',
                "dataframe": pd.DataFrame({"SEX": ["Male"], "SUBJID": ["001"]}),
                "source_tables": ["Form 1A"],
            }

    state = {
        "messages": [
            _HumanMessage("How many male participants are in cohort A?"),
            _AIMessage("The relevant database concepts are cohort membership and sex. Do you want me to extract a read-only subset or run a read-only SQL query for this?"),
            _HumanMessage("yes, extract the subset"),
        ],
        "output": {},
        "observations": [],
        "meta": {},
        "agents": {
            "rag_db_qa": {
                "pending_sql_offer": True,
                "last_database_question": "How many male participants are in cohort A?",
            }
        },
        "artifacts": {"datasets": {}},
    }

    updated = rag.rag_db_qa_node(state, llm=object(), provider="openai", service=_Service())

    assert updated["agents"]["rag_db_qa"]["pending_sql_offer"] is False
    assert updated["artifacts"]["active_dataset_id"]
    dataset = updated["artifacts"]["datasets"][updated["artifacts"]["active_dataset_id"]]
    assert dataset["kind"] == "subset"
    assert dataset["provenance"]["sql"].startswith("SELECT *")
    assert "I extracted 12 matching records." in updated["output"]["qa_response"]
