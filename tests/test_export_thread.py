from __future__ import annotations

import io
import json
import sys
import zipfile
from types import ModuleType


def _install_langchain_message_stubs() -> None:
    langchain_core = sys.modules.get("langchain_core")
    if langchain_core is None:
        langchain_core = ModuleType("langchain_core")

    messages = sys.modules.get("langchain_core.messages")
    if messages is None:
        messages = ModuleType("langchain_core.messages")

    class BaseMessage:
        def __init__(self, content: str, additional_kwargs: dict | None = None) -> None:
            self.content = content
            self.additional_kwargs = additional_kwargs or {}

    class HumanMessage(BaseMessage):
        type = "human"

    class AIMessage(BaseMessage):
        type = "ai"

    messages.BaseMessage = BaseMessage
    messages.HumanMessage = HumanMessage
    messages.AIMessage = AIMessage
    langchain_core.messages = messages
    sys.modules["langchain_core"] = langchain_core
    sys.modules["langchain_core.messages"] = messages


_install_langchain_message_stubs()

from langchain_core.messages import AIMessage, HumanMessage

from graph.memory import complete_task
from utils.export_thread import build_thread_export


def _base_state() -> dict:
    return {
        "messages": [
            HumanMessage(content="Subset index cases with diabetes."),
            AIMessage(content="Saved the reviewed subset."),
        ],
        "output": {"text": "Saved the reviewed subset."},
        "artifacts": {
            "conversation_events": [],
            "files": {
                "sql-art-1": {
                    "artifact_id": "sql-art-1",
                    "kind": "sql",
                    "producer": "rag_db_qa",
                    "mime": "text/sql",
                    "summary": "Reviewed SQL",
                    "created_at": "2026-05-05T00:00:00+00:00",
                    "content": "SELECT sensitive_payload FROM patient_table;",
                },
                "selection-art-1": {
                    "artifact_id": "selection-art-1",
                    "kind": "json",
                    "producer": "rag_db_qa",
                    "mime": "application/json",
                    "summary": "Reviewed column selection",
                    "created_at": "2026-05-05T00:00:00+00:00",
                    "content": {"content": "do not duplicate this nested payload"},
                },
            },
            "datasets": {
                "dataset-art-1": {
                    "summary": "Generated subset dataset",
                }
            },
        },
        "next_action": None,
        "last_action": None,
        "observations": [],
        "orchestrator": {},
        "planner": {},
        "agents": {},
        "node_data": {},
        "meta": {},
    }


def _state_with_completed_memory() -> dict:
    return complete_task(
        _base_state(),
        kind="db_rag_sql_extraction",
        source_question="Subset index cases with diabetes.",
        goal_text="Subset index cases with diabetes.",
        label="Diabetes subset",
        summary="Reviewed SQL was executed and saved as a subset dataset.",
        artifact_refs={
            "selection_artifact_id": "selection-art-1",
            "sql_candidate_artifact_id": "sql-art-1",
            "dataset_artifact_id": "dataset-art-1",
        },
        event_refs={"user_event_id": "evt-user-1", "completion_event_id": "evt-done-1"},
        provenance={"producer_node": "rag_db_qa"},
    )


def _read_export(state: dict) -> tuple[zipfile.ZipFile, io.BytesIO]:
    data = build_thread_export("thread-1", "openai", "test-model", state)
    buffer = io.BytesIO(data)
    return zipfile.ZipFile(buffer), buffer


def test_thread_export_includes_memory_json() -> None:
    archive, buffer = _read_export(_state_with_completed_memory())
    with archive:
        names = set(archive.namelist())
        memory = json.loads(archive.read("memory.json"))

    buffer.close()
    assert {"conversation.json", "conversation.md", "artifacts.json", "memory.json"} <= names
    assert memory["task_order"]
    assert memory["completed_tasks"]
    assert memory["failed_tasks"] == {}
    assert memory["last_task_id"] == memory["task_order"][0]
    assert memory["last_task_id_by_kind"]["db_rag_sql_extraction"] == memory["last_task_id"]
    assert memory["last_failed_task_id_by_kind"] == {}
    assert memory["last_reference_resolution"] is None


def test_memory_json_contains_refs_not_full_artifact_payloads() -> None:
    archive, buffer = _read_export(_state_with_completed_memory())
    with archive:
        memory_text = archive.read("memory.json").decode("utf-8")
        memory = json.loads(memory_text)

    buffer.close()
    task_id = memory["task_order"][0]
    artifact_refs = memory["completed_tasks"][task_id]["artifact_refs"]

    assert artifact_refs["selection_artifact_id"] == "selection-art-1"
    assert artifact_refs["sql_candidate_artifact_id"] == "sql-art-1"
    assert artifact_refs["dataset_artifact_id"] == "dataset-art-1"
    assert "SELECT sensitive_payload FROM patient_table;" not in memory_text
    assert "do not duplicate this nested payload" not in memory_text
