from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_stubs(decision: str, suggestion: str | None = None) -> None:
    langgraph_types = ModuleType("langgraph.types")
    graph_message_mod = ModuleType("langgraph.graph.message")
    graph_message_mod.add_messages = lambda current, new: (current or []) + (new or [])

    def _interrupt(_payload):
        result = {"action": decision}
        if suggestion is not None:
            result["suggestion"] = suggestion
        return result

    langgraph_types.interrupt = _interrupt

    messages_mod = ModuleType("langchain_core.messages")

    class _HumanMessage:
        def __init__(self, content: str, additional_kwargs: dict | None = None):
            self.type = "human"
            self.content = content
            self.additional_kwargs = additional_kwargs or {}

    class _AIMessage:
        def __init__(self, content: str, additional_kwargs: dict | None = None):
            self.type = "ai"
            self.content = content
            self.additional_kwargs = additional_kwargs or {}

    messages_mod.HumanMessage = _HumanMessage
    messages_mod.AIMessage = _AIMessage
    messages_mod.BaseMessage = object

    sys.modules["langgraph.types"] = langgraph_types
    sys.modules["langgraph.graph.message"] = graph_message_mod
    sys.modules["langchain_core.messages"] = messages_mod


def test_human_review_before_run_emits_review_decision_event() -> None:
    _install_stubs(decision="approve")
    for mod in ("graph.nodes.human_review_before_run", "graph.state", "graph.nodes.state_helpers"):
        sys.modules.pop(mod, None)
    module = importlib.import_module("graph.nodes.human_review_before_run")

    state = {
        "messages": [],
        "output": {"generated_code": "print(1)"},
        "meta": {"current_code_hash": "hash-1", "last_user_message_hash": "u1"},
        "agents": {},
    }

    updated = module.human_review_before_run_node(state)

    event = updated["artifacts"]["conversation_events"][-1]
    assert event["type"] == "review_decision"
    assert event["review_kind"] == "before_run_review"
    assert event["decision"] == "approve"


def test_human_review_after_error_emits_review_decision_event() -> None:
    _install_stubs(decision="regenerate", suggestion="try a grouped summary instead")
    for mod in ("graph.nodes.human_review_after_error", "graph.state", "graph.nodes.state_helpers"):
        sys.modules.pop(mod, None)
    module = importlib.import_module("graph.nodes.human_review_after_error")

    state = {
        "messages": [],
        "output": {
            "generated_code": "print(1)",
            "error": {"category": "retryable_code", "type": "ValueError", "message": "bad column"},
        },
        "meta": {
            "error_iterations": 3,
            "current_code_hash": "hash-1",
            "final_approved_code_hash": "hash-1",
            "execution_ticket_hash": "hash-1",
            "last_user_message_hash": "u2",
        },
        "agents": {},
    }

    updated = module.human_review_after_error_node(state)

    event = updated["artifacts"]["conversation_events"][-1]
    assert event["type"] == "review_decision"
    assert event["review_kind"] == "after_error_review"
    assert event["decision"] == "regenerate"
    assert "grouped summary" in event["text"]


def test_review_cancel_helper_appends_assistant_and_decision_event() -> None:
    _install_stubs(decision="cancel")
    for mod in (
        "graph.nodes.human_review_cancel",
        "graph.conversation_events",
        "graph.state",
    ):
        sys.modules.pop(mod, None)
    module = importlib.import_module("graph.nodes.human_review_cancel")

    state = {
        "messages": [],
        "output": {},
        "meta": {"last_user_message_hash": "u-cancel"},
        "agents": {},
    }

    updated = module.append_review_cancel_audit(
        state,
        actor="human_review_before_run",
        review_kind="before_run_review",
    )

    assert updated["messages"][-1].content == module.CANCEL_REVIEW_MESSAGE
    assistant_event, decision_event = updated["artifacts"]["conversation_events"][-2:]
    assert assistant_event["type"] == "assistant"
    assert assistant_event["actor"] == "human_review_before_run"
    assert assistant_event["text"] == module.CANCEL_REVIEW_MESSAGE
    assert decision_event["type"] == "review_decision"
    assert decision_event["actor"] == "human_review_before_run"
    assert decision_event["review_kind"] == "before_run_review"
    assert decision_event["decision"] == "cancel"
    assert decision_event["text"] == module.CANCEL_REVIEW_MESSAGE

    sys.modules.pop("utils.display_history", None)
    display_history_module = importlib.import_module("utils.display_history")
    history = display_history_module.build_display_history(updated)
    assert history[0].type == "ai"
    assert history[0].content == module.CANCEL_REVIEW_MESSAGE
