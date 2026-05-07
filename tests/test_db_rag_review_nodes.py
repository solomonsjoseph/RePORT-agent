from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_STUBBED_MODULES = (
    "langchain_core",
    "langchain_core.messages",
    "langgraph.types",
    "langgraph.graph.message",
    "graph.nodes.rag_db_qa",
)


@pytest.fixture(autouse=True)
def _restore_stubbed_modules():
    original = {name: sys.modules.get(name) for name in _STUBBED_MODULES}
    yield
    for name, module in original.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _ensure_langchain_core_stubs() -> None:
    langchain_core = sys.modules.get("langchain_core")
    if langchain_core is None:
        langchain_core = ModuleType("langchain_core")

    messages = sys.modules.get("langchain_core.messages")
    if messages is None:
        messages = getattr(langchain_core, "messages", None)
    if messages is None:
        messages = ModuleType("langchain_core.messages")

    if not hasattr(messages, "BaseMessage"):
        class BaseMessage:
            type = "base"

            def __init__(self, content: str) -> None:
                self.content = content
                self.additional_kwargs = {}

        messages.BaseMessage = BaseMessage

    langchain_core.messages = messages
    sys.modules["langchain_core"] = langchain_core
    sys.modules["langchain_core.messages"] = messages


def _install_review_stubs(feedback: dict[str, object]) -> None:
    _ensure_langchain_core_stubs()
    langgraph_types = ModuleType("langgraph.types")
    langgraph_types.interrupt = lambda _payload: dict(feedback)
    graph_message_mod = ModuleType("langgraph.graph.message")
    graph_message_mod.add_messages = lambda current, new: (current or []) + (new or [])
    sys.modules["langgraph.types"] = langgraph_types
    sys.modules["langgraph.graph.message"] = graph_message_mod


def _install_sql_review_helper_stub(*, mode: str = "success") -> None:
    module = ModuleType("graph.nodes.rag_db_qa")
    module._deserialize_prepared_sql_candidate = lambda payload: SimpleNamespace(**payload)
    module.LAST_EXECUTION_RAG_STATE = None

    def _execute_prepared_sql_candidate(state, rag_state, candidate, service):
        del candidate, service
        module.LAST_EXECUTION_RAG_STATE = dict(rag_state or {})
        updated_rag_state = dict(rag_state or {})
        if mode == "error":
            updated_rag_state["thread_status"] = "error"
            updated_rag_state["pending_column_review"] = {
                **dict(updated_rag_state.get("pending_column_review") or {}),
                "status": "needs_revision",
            }
        else:
            updated_rag_state["thread_status"] = "executed"
        return {
            **state,
            "agents": {
                **dict(state.get("agents") or {}),
                "rag_db_qa": updated_rag_state,
            },
        }

    module._execute_prepared_sql_candidate = _execute_prepared_sql_candidate
    sys.modules["graph.nodes.rag_db_qa"] = module


def _state() -> dict:
    return {
        "messages": [],
        "meta": {"last_user_message_hash": "u1"},
        "output": {},
        "last_action": None,
        "agents": {
            "rag_db_qa": {
                "pending_column_review_artifact_id": "sel-art-1",
                "approved_column_selection_artifact_id": None,
                "pending_sql_candidate_artifact_id": "sql-art-1",
                "thread_status": "awaiting_sql_review",
            }
        },
        "artifacts": {
            "conversation_events": [],
            "conversation_events_version": 1,
            "artifact_manifest_version": 1,
            "files": {
                "sel-art-1": {
                    "artifact_id": "sel-art-1",
                    "kind": "db_rag_column_selection",
                    "producer": "rag_db_qa",
                    "mime": "application/json",
                    "summary": "selection",
                    "created_at": "2026-05-01T00:00:00Z",
                    "content": {
                        "selection_id": "sel-1",
                        "source_question": "Subset index cases",
                        "goal_text": "Subset index cases",
                        "intent_snapshot": {"intent_id": "intent-1", "goal_text": "Subset index cases"},
                        "retrieval_summary": {"tables": ["Form 2A"], "columns": ["IC_AGE"]},
                        "tables": ["Form 2A"],
                        "columns": [{"table": "Form 2A", "column": "IC_AGE", "description": "Age"}],
                        "rationale": "Needed",
                        "feedback_history": [],
                        "status": "awaiting_review",
                    },
                },
                "sql-art-1": {
                    "artifact_id": "sql-art-1",
                    "kind": "db_rag_sql_candidate",
                    "producer": "rag_db_qa",
                    "mime": "application/json",
                    "summary": "sql",
                    "created_at": "2026-05-01T00:00:01Z",
                    "content": {
                        "sql_candidate_id": "sql-1",
                        "selection_artifact_id": "sel-art-1",
                        "source_question": "Subset index cases",
                        "goal_text": "Subset index cases",
                        "intent_snapshot": {"intent_id": "intent-1", "goal_text": "Subset index cases"},
                        "tables": ["Form 2A"],
                        "columns": [{"table": "Form 2A", "column": "IC_AGE", "description": "Age"}],
                        "sql": "select IC_AGE from form_2a",
                        "status": "prepared",
                    },
                },
            },
        },
    }


def test_column_review_approval_promotes_artifact_pointer() -> None:
    _install_review_stubs({"action": "approve"})
    for mod in ("graph.nodes.human_review_rag_db_column_selection", "graph.state", "graph.nodes.state_helpers"):
        sys.modules.pop(mod, None)
    module = importlib.import_module("graph.nodes.human_review_rag_db_column_selection")

    updated = module.human_review_rag_db_column_selection_node(_state())

    rag_state = updated["agents"]["rag_db_qa"]
    assert rag_state["approved_column_selection_artifact_id"] == "sel-art-1"
    assert rag_state["pending_column_review_artifact_id"] is None
    assert rag_state["pending_sql_candidate_artifact_id"] is None
    assert rag_state["thread_status"] == "awaiting_sql_generation"


def test_column_review_missing_artifact_errors_instead_of_approving_empty_state() -> None:
    _install_review_stubs({"action": "approve"})
    for mod in ("graph.nodes.human_review_rag_db_column_selection", "graph.state", "graph.nodes.state_helpers"):
        sys.modules.pop(mod, None)
    module = importlib.import_module("graph.nodes.human_review_rag_db_column_selection")

    state = _state()
    state["artifacts"]["files"].pop("sel-art-1")
    updated = module.human_review_rag_db_column_selection_node(state)

    rag_state = updated["agents"]["rag_db_qa"]
    assert rag_state["thread_status"] == "error"
    assert rag_state["pending_column_review_artifact_id"] is None
    assert updated["output"]["error"]["category"] == "db_rag_review"


def test_sql_review_regenerate_with_selection_feedback_reopens_column_review() -> None:
    _install_review_stubs({"action": "regenerate", "feedback": "Use the outcome table instead."})
    _install_sql_review_helper_stub()
    for mod in ("graph.nodes.human_review_rag_db_sql_execution", "graph.state", "graph.nodes.state_helpers"):
        sys.modules.pop(mod, None)
    module = importlib.import_module("graph.nodes.human_review_rag_db_sql_execution")

    service = SimpleNamespace(
        classify_sql_review_feedback=lambda **_kwargs: {"label": "selection_revision", "confidence": 0.9}
    )
    updated = module.human_review_rag_db_sql_execution_node(_state(), service)

    rag_state = updated["agents"]["rag_db_qa"]
    assert rag_state["pending_sql_candidate_artifact_id"] is None
    assert rag_state["pending_column_review_artifact_id"] == "sel-art-1"
    assert rag_state["approved_column_selection_artifact_id"] is None
    assert rag_state["thread_status"] == "awaiting_column_review"


def test_sql_review_approval_preserves_selection_context_for_execution_then_clears_pointers() -> None:
    _install_review_stubs({"action": "approve"})
    _install_sql_review_helper_stub()
    for mod in ("graph.nodes.human_review_rag_db_sql_execution", "graph.state", "graph.nodes.state_helpers"):
        sys.modules.pop(mod, None)
    module = importlib.import_module("graph.nodes.human_review_rag_db_sql_execution")
    helper_stub = sys.modules["graph.nodes.rag_db_qa"]

    state = _state()
    state["agents"]["rag_db_qa"]["approved_column_selection_artifact_id"] = "sel-art-1"
    updated = module.human_review_rag_db_sql_execution_node(state, SimpleNamespace())

    execution_rag_state = helper_stub.LAST_EXECUTION_RAG_STATE
    assert execution_rag_state["pending_sql_candidate_artifact_id"] == "sql-art-1"
    assert execution_rag_state["approved_column_selection_artifact_id"] == "sel-art-1"
    assert execution_rag_state["pending_column_review_artifact_id"] == "sel-art-1"
    assert execution_rag_state["pending_column_review"]["selection_id"] == "sel-1"

    rag_state = updated["agents"]["rag_db_qa"]
    assert rag_state["thread_status"] == "completed"
    assert rag_state["pending_sql_candidate_artifact_id"] is None
    assert rag_state.get("pending_sql_candidate") is None
    assert rag_state["pending_column_review_artifact_id"] is None
    assert rag_state["approved_column_selection_artifact_id"] is None
    assert rag_state.get("pending_column_review") is None

    updated["last_action"] = "human_review_rag_db_sql_execution"
    from graph.nodes.orchestrator.workflow_status import derive_workflow_status

    assert derive_workflow_status(updated) == {
        "milestone": "db_rag_sql_completed",
        "completion_status": "complete",
        "blocker_signature": None,
    }


def test_sql_review_execution_error_reopens_column_review_pointer() -> None:
    _install_review_stubs({"action": "approve"})
    _install_sql_review_helper_stub(mode="error")
    for mod in ("graph.nodes.human_review_rag_db_sql_execution", "graph.state", "graph.nodes.state_helpers"):
        sys.modules.pop(mod, None)
    module = importlib.import_module("graph.nodes.human_review_rag_db_sql_execution")

    state = _state()
    state["agents"]["rag_db_qa"]["approved_column_selection_artifact_id"] = "sel-art-1"
    updated = module.human_review_rag_db_sql_execution_node(state, SimpleNamespace())

    rag_state = updated["agents"]["rag_db_qa"]
    assert rag_state["thread_status"] == "error"
    assert rag_state["pending_sql_candidate_artifact_id"] is None
    assert rag_state.get("pending_sql_candidate") is None
    assert rag_state["pending_column_review_artifact_id"] == "sel-art-1"
    assert rag_state["approved_column_selection_artifact_id"] is None
    assert rag_state["pending_column_review"]["status"] == "needs_revision"


def test_sql_review_missing_selection_artifact_errors_instead_of_executing_empty_state() -> None:
    _install_review_stubs({"action": "approve"})
    _install_sql_review_helper_stub()
    for mod in ("graph.nodes.human_review_rag_db_sql_execution", "graph.state", "graph.nodes.state_helpers"):
        sys.modules.pop(mod, None)
    module = importlib.import_module("graph.nodes.human_review_rag_db_sql_execution")

    state = _state()
    state["artifacts"]["files"].pop("sel-art-1")
    updated = module.human_review_rag_db_sql_execution_node(state, SimpleNamespace())

    rag_state = updated["agents"]["rag_db_qa"]
    assert rag_state["thread_status"] == "error"
    assert rag_state["pending_sql_candidate_artifact_id"] is None
    assert updated["output"]["error"]["category"] == "db_rag_sql"


def test_sql_review_missing_candidate_artifact_errors_instead_of_executing_empty_state() -> None:
    _install_review_stubs({"action": "approve"})
    _install_sql_review_helper_stub()
    for mod in ("graph.nodes.human_review_rag_db_sql_execution", "graph.state", "graph.nodes.state_helpers"):
        sys.modules.pop(mod, None)
    module = importlib.import_module("graph.nodes.human_review_rag_db_sql_execution")

    state = _state()
    state["artifacts"]["files"].pop("sql-art-1")
    updated = module.human_review_rag_db_sql_execution_node(state, SimpleNamespace())

    rag_state = updated["agents"]["rag_db_qa"]
    assert rag_state["thread_status"] == "error"
    assert rag_state["pending_sql_candidate_artifact_id"] is None
    assert updated["output"]["error"]["category"] == "db_rag_sql"


def test_sql_review_regenerate_with_sql_only_feedback_preserves_selection_pointer() -> None:
    _install_review_stubs({"action": "regenerate", "feedback": "Keep the same columns but filter to baseline rows."})
    _install_sql_review_helper_stub()
    for mod in ("graph.nodes.human_review_rag_db_sql_execution", "graph.state", "graph.nodes.state_helpers"):
        sys.modules.pop(mod, None)
    module = importlib.import_module("graph.nodes.human_review_rag_db_sql_execution")

    state = _state()
    state["agents"]["rag_db_qa"]["approved_column_selection_artifact_id"] = "sel-art-1"
    service = SimpleNamespace(
        classify_sql_review_feedback=lambda **_kwargs: {"label": "sql_only_revision", "confidence": 0.9}
    )
    updated = module.human_review_rag_db_sql_execution_node(state, service)

    rag_state = updated["agents"]["rag_db_qa"]
    assert rag_state["pending_sql_candidate_artifact_id"] is None
    assert rag_state["approved_column_selection_artifact_id"] == "sel-art-1"
    assert rag_state["pending_column_review_artifact_id"] is None
    assert rag_state["thread_status"] == "awaiting_sql_generation"
    selection_artifact = updated["artifacts"]["files"]["sel-art-1"]["content"]
    assert selection_artifact["status"] == "approved"
    assert selection_artifact["feedback_history"][-1]["feedback"].endswith(
        "Keep the same columns but filter to baseline rows."
    )


def test_workflow_status_uses_pending_column_review_artifact_pointer() -> None:
    from graph.nodes.orchestrator.workflow_status import derive_workflow_status

    state = _state()
    state["agents"]["rag_db_qa"]["pending_sql_candidate_artifact_id"] = None
    state["agents"]["rag_db_qa"]["pending_column_review"] = None

    status = derive_workflow_status(state)

    assert status == {
        "milestone": "awaiting_rag_db_column_review",
        "completion_status": "blocked_waiting",
        "blocker_signature": "waiting_for_rag_db_column_review:sel-art-1",
    }


def test_workflow_status_uses_pending_sql_candidate_artifact_pointer() -> None:
    from graph.nodes.orchestrator.workflow_status import derive_workflow_status

    state = _state()
    state["agents"]["rag_db_qa"]["pending_column_review_artifact_id"] = None
    state["agents"]["rag_db_qa"]["pending_sql_candidate"] = None

    status = derive_workflow_status(state)

    assert status == {
        "milestone": "awaiting_rag_db_sql_review",
        "completion_status": "blocked_waiting",
        "blocker_signature": "waiting_for_rag_db_sql_review:sql-art-1",
    }


def test_workflow_status_uses_pending_db_rag_recoverable_error() -> None:
    from graph.nodes.orchestrator.workflow_status import derive_workflow_status

    state = _state()
    state["meta"].pop("awaiting_user_clarification", None)
    state["agents"]["rag_db_qa"]["pending_column_review_artifact_id"] = None
    state["agents"]["rag_db_qa"]["pending_sql_candidate_artifact_id"] = None
    state["agents"]["rag_db_qa"]["pending_recoverable_error"] = {
        "status": "awaiting_reply",
        "stage": "sql_preparation",
    }

    status = derive_workflow_status(state)

    assert status == {
        "milestone": "awaiting_rag_db_recoverable_error_clarification",
        "completion_status": "blocked_waiting",
        "blocker_signature": "waiting_for_rag_db_recoverable_error:sql_preparation",
    }


def test_workflow_status_surfaces_db_rag_review_error_after_missing_artifact() -> None:
    from graph.nodes.orchestrator.workflow_status import derive_workflow_status

    state = _state()
    state["last_action"] = "human_review_rag_db_column_selection"
    state["agents"]["rag_db_qa"]["thread_status"] = "error"
    state["agents"]["rag_db_qa"]["pending_column_review_artifact_id"] = None
    state["agents"]["rag_db_qa"]["pending_sql_candidate_artifact_id"] = None
    state["output"]["error"] = {
        "category": "db_rag_review",
        "type": "MissingColumnSelectionArtifact",
        "message": "missing selection",
    }

    status = derive_workflow_status(state)

    assert status["milestone"] == "db_rag_review_error"
    assert status["completion_status"] == "blocked_waiting"


def test_workflow_status_marks_sql_completion_without_prior_metadata_answer() -> None:
    from graph.nodes.orchestrator.workflow_status import derive_workflow_status

    state = _state()
    state["last_action"] = "human_review_rag_db_sql_execution"
    state["agents"]["rag_db_qa"]["thread_status"] = "completed"
    state["agents"]["rag_db_qa"]["pending_column_review_artifact_id"] = None
    state["agents"]["rag_db_qa"]["pending_sql_candidate_artifact_id"] = None
    state["output"] = {}

    status = derive_workflow_status(state)

    assert status == {
        "milestone": "db_rag_sql_completed",
        "completion_status": "complete",
        "blocker_signature": None,
    }


def test_workflow_status_ignores_legacy_embedded_pending_reviews_without_pointers() -> None:
    from graph.nodes.orchestrator.workflow_status import derive_workflow_status

    state = _state()
    state["agents"]["rag_db_qa"] = {
        "pending_column_review": {"status": "awaiting_review", "selection_id": "legacy-sel"},
        "pending_sql_candidate": {"status": "prepared", "selection_id": "legacy-sel"},
    }

    status = derive_workflow_status(state)

    assert status == {
        "milestone": "needs_code",
        "completion_status": "incomplete",
        "blocker_signature": "missing_next_step",
    }
