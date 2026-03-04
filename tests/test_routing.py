from __future__ import annotations

import importlib
import sys
from types import ModuleType


def _install_dependency_stubs() -> None:
    graph_mod = ModuleType("langgraph.graph")
    graph_mod.END = "__END__"

    messages_mod = ModuleType("langchain_core.messages")
    messages_mod.BaseMessage = object

    graph_message_mod = ModuleType("langgraph.graph.message")
    graph_message_mod.add_messages = lambda current, new: (current or []) + (new or [])

    sys.modules["langgraph.graph"] = graph_mod
    sys.modules["langchain_core.messages"] = messages_mod
    sys.modules["langgraph.graph.message"] = graph_message_mod


def test_route_by_next_action_redirects_execute_without_approval() -> None:
    _install_dependency_stubs()
    sys.modules.pop("graph.routing", None)
    routing = importlib.import_module("graph.routing")

    state = {
        "next_action": "execute_code",
        "agents": {"human_review": {"before_run_decision": None}},
    }

    assert routing.route_by_next_action(state) == "human_review_before_run"


def test_route_by_next_action_allows_execute_with_approval() -> None:
    _install_dependency_stubs()
    sys.modules.pop("graph.routing", None)
    routing = importlib.import_module("graph.routing")

    state = {
        "next_action": "execute_code",
        "meta": {"current_code_hash": "h1"},
        "agents": {"human_review": {"before_run_decision": "approve", "approved_code_hash": "h1"}},
    }

    assert routing.route_by_next_action(state) == "execute_code"


def test_route_by_next_action_keeps_non_execute_actions() -> None:
    _install_dependency_stubs()
    sys.modules.pop("graph.routing", None)
    routing = importlib.import_module("graph.routing")

    assert routing.route_by_next_action({"next_action": "qa", "agents": {}}) == "qa"


def test_route_by_next_action_redirects_execute_when_approval_hash_is_stale() -> None:
    _install_dependency_stubs()
    sys.modules.pop("graph.routing", None)
    routing = importlib.import_module("graph.routing")

    state = {
        "next_action": "execute_code",
        "meta": {"current_code_hash": "new-hash"},
        "agents": {"human_review": {"before_run_decision": "approve", "approved_code_hash": "old-hash"}},
    }

    assert routing.route_by_next_action(state) == "human_review_before_run"


def test_route_by_next_action_allows_execute_with_matching_approval_hash() -> None:
    _install_dependency_stubs()
    sys.modules.pop("graph.routing", None)
    routing = importlib.import_module("graph.routing")

    state = {
        "next_action": "execute_code",
        "meta": {"current_code_hash": "same-hash"},
        "agents": {"human_review": {"before_run_decision": "approve", "approved_code_hash": "same-hash"}},
    }

    assert routing.route_by_next_action(state) == "execute_code"
