from __future__ import annotations

import importlib
import sys
from types import ModuleType


def _load_execute_code_module():
    messages_mod = ModuleType("langchain_core.messages")
    messages_mod.BaseMessage = object
    sys.modules["langchain_core.messages"] = messages_mod
    graph_message_mod = ModuleType("langgraph.graph.message")
    graph_message_mod.add_messages = lambda current, new: (current or []) + (new or [])
    sys.modules["langgraph.graph.message"] = graph_message_mod

    lifelines_mod = ModuleType("lifelines")
    lifelines_mod.KaplanMeierFitter = object
    lifelines_mod.CoxPHFitter = object
    sys.modules["lifelines"] = lifelines_mod
    sys.modules.pop("tools.execution", None)
    sys.modules.pop("graph.nodes.execute_code", None)
    return importlib.import_module("graph.nodes.execute_code")


def test_execute_code_node_preserves_structured_sandbox_errors(monkeypatch) -> None:
    execute_code = _load_execute_code_module()

    def fake_run_python_user(_code, _df):
        return None, "", b"", {
            "category": "policy_blocked",
            "type": "PolicyBlockedError",
            "message": "Disallowed import: subprocess",
        }

    monkeypatch.setattr(execute_code, "run_python_user", fake_run_python_user)

    state = {
        "output": {"generated_code": "import subprocess"},
        "agents": {"executor": {"run_status": "pending"}},
    }

    updated = execute_code.execute_code_node(state, df=None)

    assert updated["output"]["error"] == {
        "category": "policy_blocked",
        "type": "PolicyBlockedError",
        "message": "Disallowed import: subprocess",
    }
    assert updated["agents"]["executor"]["run_status"] == "error"


def test_execute_code_node_supports_callable_dataset_resolver(monkeypatch) -> None:
    execute_code = _load_execute_code_module()
    captured = {}

    def fake_run_python_user(_code, df):
        captured["df"] = df
        return "ok", "done", b"", None

    monkeypatch.setattr(execute_code, "run_python_user", fake_run_python_user)

    state = {
        "output": {"generated_code": "print(df.shape)"},
        "agents": {"executor": {"run_status": "pending"}},
        "messages": [],
    }

    updated = execute_code.execute_code_node(
        state,
        df=lambda current_state: {"resolved_from_state": len(current_state.get("messages", []))},
    )

    assert captured["df"] == {"resolved_from_state": 0}
    assert updated["agents"]["executor"]["run_status"] == "ok"


def test_execute_code_node_emits_failure_events_when_code_is_missing() -> None:
    execute_code = _load_execute_code_module()

    state = {
        "output": {},
        "agents": {"executor": {"run_status": "pending"}},
        "meta": {},
        "messages": [],
        "artifacts": {
            "conversation_events": [],
            "conversation_events_version": 1,
            "artifact_manifest_version": 1,
            "files": {},
        },
    }

    updated = execute_code.execute_code_node(state, df=None)

    from graph.state_views import get_conversation_events

    events = get_conversation_events(updated)
    assert [event["type"] for event in events] == ["execution_finished", "error"]
    assert events[0]["status"] == "error"
    assert events[0]["text"] == "No code available to execute."
    assert events[1]["error"]["type"] == "NoCode"


def test_execute_code_node_emits_execution_events_and_result_artifacts(monkeypatch) -> None:
    execute_code = _load_execute_code_module()

    def fake_run_python_user(_code, _df):
        return {"rows": 2}, "row count: 2", b"png-bytes", None

    monkeypatch.setattr(execute_code, "run_python_user", fake_run_python_user)

    state = {
        "output": {"generated_code": "print(df.shape)"},
        "agents": {"executor": {"run_status": "pending"}},
        "meta": {},
        "messages": [],
        "artifacts": {
            "conversation_events": [],
            "conversation_events_version": 1,
            "artifact_manifest_version": 1,
            "files": {},
        },
    }

    updated = execute_code.execute_code_node(state, df=None)

    from graph.state_views import get_artifact_files, get_conversation_events

    events = get_conversation_events(updated)
    files = get_artifact_files(updated)

    assert [event["type"] for event in events] == ["execution_started", "execution_finished", "figure"]
    assert events[0]["text"] == "Executing approved Python code."
    assert events[1]["text"] == "row count: 2"
    assert events[1]["artifact_id"]
    assert events[2]["artifact_id"]
    assert updated["output"]["figure_artifact_id"] == events[2]["artifact_id"]
    assert "figure_png" not in updated["output"]

    text_artifact = files[events[1]["artifact_id"]]
    assert text_artifact["kind"] == "text"
    assert text_artifact["producer"] == "executor"
    assert text_artifact["mime"] == "text/plain"
    assert text_artifact["content"] == "row count: 2"

    figure_artifact = files[events[2]["artifact_id"]]
    assert figure_artifact["kind"] == "figure"
    assert figure_artifact["producer"] == "executor"
    assert figure_artifact["mime"] == "image/png"
    assert figure_artifact["content"]["path"]


def test_execute_code_node_clears_stale_error_and_figure_on_success(monkeypatch) -> None:
    execute_code = _load_execute_code_module()

    def fake_run_python_user(_code, _df):
        return {"rows": 1}, "ok", b"", None

    monkeypatch.setattr(execute_code, "run_python_user", fake_run_python_user)

    state = {
        "output": {
            "generated_code": "print(df.shape)",
            "error": {"type": "OldError", "message": "stale"},
            "figure_artifact_id": "stale-figure",
        },
        "agents": {"executor": {"run_status": "pending"}},
        "meta": {},
        "messages": [],
        "artifacts": {
            "conversation_events": [],
            "conversation_events_version": 1,
            "artifact_manifest_version": 1,
            "files": {},
        },
    }

    updated = execute_code.execute_code_node(state, df=None)

    assert updated["output"]["text"] == "ok"
    assert "error" not in updated["output"]
    assert "figure_artifact_id" not in updated["output"]


def test_execute_code_node_emits_failure_and_error_events(monkeypatch) -> None:
    execute_code = _load_execute_code_module()

    def fake_run_python_user(_code, _df):
        return None, "", b"", {
            "category": "policy_blocked",
            "type": "PolicyBlockedError",
            "message": "Disallowed import: subprocess",
        }

    monkeypatch.setattr(execute_code, "run_python_user", fake_run_python_user)

    state = {
        "output": {"generated_code": "import subprocess"},
        "agents": {"executor": {"run_status": "pending"}},
        "meta": {},
        "messages": [],
        "artifacts": {
            "conversation_events": [],
            "conversation_events_version": 1,
            "artifact_manifest_version": 1,
            "files": {},
        },
    }

    updated = execute_code.execute_code_node(state, df=None)

    from graph.state_views import get_conversation_events

    events = get_conversation_events(updated)

    assert [event["type"] for event in events] == ["execution_started", "execution_finished", "error"]
    assert events[1]["status"] == "error"
    assert events[1]["text"] == "Disallowed import: subprocess"
    assert events[2]["error"] == {
        "category": "policy_blocked",
        "type": "PolicyBlockedError",
        "message": "Disallowed import: subprocess",
    }
