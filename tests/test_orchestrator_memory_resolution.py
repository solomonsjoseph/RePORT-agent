from __future__ import annotations

import importlib
import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

from graph.memory import complete_task
from graph.state import MetaKeys


class _HumanMessage:
    type = "human"

    def __init__(self, content: str, id: str | None = None):
        self.content = content
        self.id = id or ""


class _AIMessage:
    type = "ai"

    def __init__(self, content: str):
        self.content = content


class _FormattedPrompt:
    def __init__(self, rendered):
        self._rendered = rendered

    def to_messages(self):
        return self._rendered


class _PromptTemplate:
    def __init__(self, messages):
        self._messages = [m for m in messages if isinstance(m, tuple)]

    def format_prompt(self, **kwargs):
        rendered = []
        for role, template in self._messages:
            rendered.append({"role": role, "content": template.format(**kwargs)})
        return _FormattedPrompt(rendered)


class _ChatPromptTemplate:
    @staticmethod
    def from_messages(messages):
        return _PromptTemplate(messages)


class _MessagesPlaceholder:
    def __init__(self, variable_name: str, optional: bool = False) -> None:
        self.variable_name = variable_name
        self.optional = optional


class _FewShotChatMessagePromptTemplate:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _LLM:
    def __init__(self, action: str = "qa"):
        self.action = action
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return SimpleNamespace(
            content=json.dumps({"action": self.action, "thought": "test planner fallback"})
        )


def _install_stubs() -> None:
    messages_mod = ModuleType("langchain_core.messages")
    messages_mod.BaseMessage = object
    messages_mod.HumanMessage = _HumanMessage
    messages_mod.AIMessage = _AIMessage

    prompts_mod = ModuleType("langchain_core.prompts")
    prompts_mod.ChatPromptTemplate = _ChatPromptTemplate
    prompts_mod.MessagesPlaceholder = _MessagesPlaceholder
    prompts_mod.FewShotChatMessagePromptTemplate = _FewShotChatMessagePromptTemplate

    langchain_core_mod = ModuleType("langchain_core")
    langchain_core_mod.messages = messages_mod
    langchain_core_mod.prompts = prompts_mod

    graph_message_mod = ModuleType("langgraph.graph.message")
    graph_message_mod.add_messages = lambda current, new: (current or []) + (new or [])

    sys.modules["langchain_core"] = langchain_core_mod
    sys.modules["langchain_core.messages"] = messages_mod
    sys.modules["langchain_core.prompts"] = prompts_mod
    sys.modules["langgraph.graph.message"] = graph_message_mod


def _fresh_node_module():
    _install_stubs()
    for mod in (
        "prompts.planner_prompt",
        "graph.nodes.orchestrator",
        "graph.nodes.orchestrator.node",
        "graph.nodes.orchestrator.planner",
        "graph.nodes.orchestrator.policy",
        "graph.nodes.orchestrator.action_mask",
        "graph.nodes.orchestrator.context_builder",
    ):
        sys.modules.pop(mod, None)
    return importlib.import_module("graph.nodes.orchestrator.node")


def _base_state(user_message: str = "add gender too") -> dict[str, Any]:
    return {
        "messages": [_HumanMessage(user_message, id="turn-1")],
        "output": {},
        "artifacts": {
            "files": {
                "sql-art-1": {
                    "artifact_id": "sql-art-1",
                    "kind": "db_rag_sql_candidate",
                    "producer": "rag_db_qa",
                    "mime": "application/sql",
                    "summary": "Reviewed SQL candidate",
                    "created_at": "2026-05-05T00:00:00+00:00",
                    "content": "SELECT * FROM t",
                },
                "sel-art-1": {
                    "artifact_id": "sel-art-1",
                    "kind": "db_rag_column_selection",
                    "producer": "rag_db_qa",
                    "mime": "application/json",
                    "summary": "Reviewed column selection",
                    "created_at": "2026-05-05T00:00:00+00:00",
                    "content": {"columns": ["age"]},
                },
            },
            "datasets": {
                "dataset-1": {"kind": "subset", "name": "diabetes subset", "rows": 10}
            },
            "conversation_events": [],
            "conversation_events_version": 1,
            "artifact_manifest_version": 1,
        },
        "next_action": None,
        "last_action": None,
        "observations": [],
        "orchestrator": {},
        "planner": {},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": {
                "before_run_decision": None,
                "after_error_decision": None,
                "final_decision": None,
            },
            "qa": {},
            "generate_code": {},
            "rag_db_qa": {},
        },
        "node_data": {},
        "meta": {"error_iterations": 0, "workflow_trace": []},
    }


def _state_with_sql_task(user_message: str = "add gender too") -> dict[str, Any]:
    return complete_task(
        _base_state(user_message),
        kind="db_rag_sql_extraction",
        source_question="Create diabetes subset",
        goal_text="Create diabetes subset",
        label="diabetes subset",
        summary="Reviewed SQL was executed and saved as a subset dataset.",
        artifact_refs={
            "selection_artifact_id": "sel-art-1",
            "sql_candidate_artifact_id": "sql-art-1",
            "dataset_artifact_id": "dataset-1",
        },
        event_refs={"completion_event_id": "evt-1"},
        provenance={"producer_node": "rag_db_qa"},
    )


def _task_id(state: dict[str, Any]) -> str:
    return state["memory"]["last_task_id"]


def _state_with_two_sql_tasks(user_message: str) -> dict[str, Any]:
    state = _state_with_sql_task(user_message)
    return complete_task(
        state,
        kind="db_rag_sql_extraction",
        source_question="Create hypertension subset",
        goal_text="Create hypertension subset",
        label="hypertension subset",
        summary="Reviewed SQL was executed and saved as a subset dataset.",
        artifact_refs={
            "selection_artifact_id": "sel-art-1",
            "sql_candidate_artifact_id": "sql-art-1",
            "dataset_artifact_id": "dataset-1",
        },
        event_refs={"completion_event_id": "evt-2"},
        provenance={"producer_node": "rag_db_qa"},
    )


def _memory_clarification_state(
    user_reply: str,
    *,
    relationship: str | None = "inspect_artifact",
) -> dict[str, Any]:
    state = _state_with_two_sql_tasks(user_reply)
    memory = state["memory"]
    candidates = [
        {
            "task_id": task_id,
            "display_ordinal": memory["completed_tasks"][task_id]["display_ordinal"],
            "kind": memory["completed_tasks"][task_id]["kind"],
            "label": memory["completed_tasks"][task_id]["label"],
            "hint": memory["completed_tasks"][task_id]["summary"],
        }
        for task_id in memory["task_order"]
    ]
    state["meta"][MetaKeys.AWAITING_USER_CLARIFICATION] = True
    state["meta"][MetaKeys.CLARIFICATION_KIND] = "memory_reference_resolution"
    state["meta"][MetaKeys.CLARIFICATION_RETURN_NODE] = "orchestrator"
    memory["pending_reference_clarification"] = {
        "status": "awaiting_reply",
        "user_message_hash": "original-hash",
        "original_user_message": "rerun that query",
        "candidates": candidates,
    }
    if relationship is not None:
        memory["pending_reference_clarification"]["relationship"] = relationship
        memory["pending_reference_clarification"]["intended_action"] = "test_action"
    return state


def _fresh_clarification_module():
    _install_stubs()
    sys.modules.pop("graph.nodes.clarification", None)
    return importlib.import_module("graph.nodes.clarification")


def _resolver(
    label: str,
    *,
    task_id: str | None = None,
    relationship: str | None = None,
    needs_reference: bool | None = None,
):
    calls = []

    def resolve(_state, _llm, user_message, user_message_hash):
        calls.append((user_message, user_message_hash))
        result: dict[str, Any] = {
            "label": label,
            "task_id": task_id,
            "relationship": relationship,
            "intended_action": None,
            "confidence": "high",
            "needs_reference": False,
            "reason": "test resolver result",
        }
        if needs_reference is not None:
            result["needs_reference"] = needs_reference
        elif label in {"ambiguous", "unknown"}:
            result["needs_reference"] = True
        return result

    resolve.calls = calls
    return resolve


def test_pending_clarification_skips_memory_resolver(monkeypatch) -> None:
    node = _fresh_node_module()
    state = _state_with_sql_task("I mean the diabetes one")
    state["meta"][MetaKeys.AWAITING_USER_CLARIFICATION] = True
    state["meta"][MetaKeys.CLARIFICATION_KIND] = "qa_followup"
    state["meta"][MetaKeys.CLARIFICATION_RETURN_NODE] = "qa"
    resolver = _resolver("resolved", task_id=_task_id(state), relationship="revision")
    monkeypatch.setattr(node, "resolve_reference_for_turn", resolver)

    result = node.orchestrator_node(state, _LLM(), ["clarification", "qa", "rag_db_qa", "end"])

    assert resolver.calls == []
    assert result["next_action"] == "clarification"


def test_non_fresh_user_turn_skips_memory_resolver(monkeypatch) -> None:
    node = _fresh_node_module()
    state = _state_with_sql_task("add gender too")
    state["messages"].append(_AIMessage("Previous answer"))
    resolver = _resolver("resolved", task_id=_task_id(state), relationship="revision")
    monkeypatch.setattr(node, "resolve_reference_for_turn", resolver)

    result = node.orchestrator_node(state, _LLM(), ["rag_db_qa", "qa", "end"])

    assert resolver.calls == []
    assert result["next_action"] in {"qa", "end"}


def test_preselected_next_action_skips_memory_resolver(monkeypatch) -> None:
    node = _fresh_node_module()
    state = _state_with_sql_task("add gender too")
    state["agents"]["qa"]["tool_requests"] = [
        {"tool_name": "search", "payload": {"query": "diabetes"}}
    ]
    resolver = _resolver("resolved", task_id=_task_id(state), relationship="revision")
    monkeypatch.setattr(node, "resolve_reference_for_turn", resolver)

    result = node.orchestrator_node(state, _LLM(), ["tool_handler", "rag_db_qa", "qa", "end"])

    assert resolver.calls == []
    assert result["next_action"] == "tool_handler"


def test_no_completed_tasks_skips_memory_resolver(monkeypatch) -> None:
    node = _fresh_node_module()
    state = _base_state("add gender too")
    resolver = _resolver("resolved", task_id="task_missing", relationship="revision")
    monkeypatch.setattr(node, "resolve_reference_for_turn", resolver)

    result = node.orchestrator_node(state, _LLM(), ["rag_db_qa", "qa", "end"])

    assert resolver.calls == []
    assert "memory" in result


def test_pending_deterministic_workflow_skips_memory_resolver(monkeypatch) -> None:
    node = _fresh_node_module()
    state = _state_with_sql_task("add gender too")
    state["agents"]["rag_db_qa"]["pending_column_review"] = {"status": "awaiting_review"}
    resolver = _resolver("resolved", task_id=_task_id(state), relationship="revision")
    monkeypatch.setattr(node, "resolve_reference_for_turn", resolver)

    result = node.orchestrator_node(
        state,
        _LLM(),
        ["human_review_rag_db_column_selection", "rag_db_qa", "qa", "end"],
    )

    assert resolver.calls == []
    assert result["next_action"] == "human_review_rag_db_column_selection"


def test_resolved_db_rag_revision_routes_to_rag_db_qa(monkeypatch) -> None:
    node = _fresh_node_module()
    state = _state_with_sql_task("add gender too")
    task_id = _task_id(state)
    monkeypatch.setattr(node, "resolve_reference_for_turn", _resolver("resolved", task_id=task_id, relationship="revision"))

    result = node.orchestrator_node(state, _LLM(), ["rag_db_qa", "generate_code", "qa", "end"])

    assert result["next_action"] == "rag_db_qa"
    assert result["meta"][MetaKeys.RESOLVED_TASK_ID] == task_id
    assert result["meta"][MetaKeys.RESOLVED_TASK_KIND] == "db_rag_sql_extraction"
    assert result["meta"][MetaKeys.RESOLVED_TASK_RELATIONSHIP] == "revision"
    assert (
        f"task_id={task_id} relationship=revision routed_node=rag_db_qa"
        in result["observations"]
    )


def test_resolved_db_rag_inspect_artifact_routes_to_rag_db_qa(monkeypatch) -> None:
    node = _fresh_node_module()
    state = _state_with_sql_task("what SQL did you use?")
    task_id = _task_id(state)
    monkeypatch.setattr(
        node,
        "resolve_reference_for_turn",
        _resolver("resolved", task_id=task_id, relationship="inspect_artifact"),
    )

    result = node.orchestrator_node(state, _LLM(), ["rag_db_qa", "generate_code", "qa", "end"])

    assert result["next_action"] == "rag_db_qa"
    assert result["meta"][MetaKeys.RESOLVED_TASK_ID] == task_id
    assert result["meta"][MetaKeys.RESOLVED_TASK_RELATIONSHIP] == "inspect_artifact"


def test_resolved_db_rag_unsupported_sql_relationships_route_to_db_rag_for_error(monkeypatch) -> None:
    node = _fresh_node_module()
    for relationship in ("rerun", "explain", "compare"):
        state = _state_with_sql_task(f"{relationship} that")
        task_id = _task_id(state)
        monkeypatch.setattr(
            node,
            "resolve_reference_for_turn",
            _resolver("resolved", task_id=task_id, relationship=relationship),
        )

        result = node.orchestrator_node(state, _LLM(), ["rag_db_qa", "generate_code", "qa", "end"])

        assert result["next_action"] == "rag_db_qa"
        assert result["meta"][MetaKeys.RESOLVED_TASK_ID] == task_id
        assert result["meta"][MetaKeys.RESOLVED_TASK_RELATIONSHIP] == relationship


def test_fresh_resolved_meta_does_not_reroute_after_action_consumes_it(monkeypatch) -> None:
    node = _fresh_node_module()
    state = _state_with_sql_task("what SQL did you use?")
    task_id = _task_id(state)
    monkeypatch.setattr(
        node,
        "resolve_reference_for_turn",
        _resolver("resolved", task_id=task_id, relationship="inspect_artifact"),
    )

    first = node.orchestrator_node(state, _LLM(), ["rag_db_qa", "generate_code", "qa", "end"])
    after_action = {
        **first,
        "last_action": "rag_db_qa",
        "next_action": None,
        "orchestrator": {},
    }
    second = node.orchestrator_node(
        after_action,
        _LLM(),
        ["rag_db_qa", "generate_code", "qa", "end"],
    )

    assert first["next_action"] == "rag_db_qa"
    assert first["meta"][MetaKeys.RESOLVED_TASK_ID] == task_id
    assert MetaKeys.RESOLVED_TASK_ID not in second["meta"]
    assert MetaKeys.RESOLVED_TASK_RELATIONSHIP not in second["meta"]
    assert (
        f"task_id={task_id} relationship=inspect_artifact routed_node=rag_db_qa"
        not in second["observations"][len(first["observations"]):]
    )


def test_consumed_resolved_meta_clears_before_deterministic_followup(monkeypatch) -> None:
    node = _fresh_node_module()
    state = _state_with_sql_task("what SQL did you use?")
    task_id = _task_id(state)
    monkeypatch.setattr(
        node,
        "resolve_reference_for_turn",
        _resolver("resolved", task_id=task_id, relationship="inspect_artifact"),
    )

    first = node.orchestrator_node(
        state,
        _LLM(),
        ["rag_db_qa", "human_review_rag_db_column_selection", "qa", "end"],
    )
    agents = dict(first["agents"])
    rag_state = dict(agents["rag_db_qa"])
    rag_state["pending_column_review"] = {"status": "awaiting_review"}
    agents["rag_db_qa"] = rag_state
    after_rag_db = {
        **first,
        "agents": agents,
        "last_action": "rag_db_qa",
        "next_action": None,
        "orchestrator": {},
    }

    result = node.orchestrator_node(
        after_rag_db,
        _LLM(),
        ["rag_db_qa", "human_review_rag_db_column_selection", "qa", "end"],
    )

    assert first["next_action"] == "rag_db_qa"
    assert first["meta"][MetaKeys.RESOLVED_TASK_ID] == task_id
    assert result["next_action"] == "human_review_rag_db_column_selection"
    assert MetaKeys.RESOLVED_TASK_ID not in result["meta"]
    assert MetaKeys.RESOLVED_TASK_RELATIONSHIP not in result["meta"]
    assert "resolved_task_meta_consumed" not in result["meta"]


def test_resolved_db_rag_use_as_input_sets_analysis_dataset_id_and_routes_generate_code(monkeypatch) -> None:
    node = _fresh_node_module()
    state = _state_with_sql_task("analyze that subset")
    task_id = _task_id(state)
    monkeypatch.setattr(
        node,
        "resolve_reference_for_turn",
        _resolver("resolved", task_id=task_id, relationship="use_as_input"),
    )

    result = node.orchestrator_node(state, _LLM(), ["rag_db_qa", "generate_code", "qa", "end"])

    assert result["next_action"] == "generate_code"
    assert result["meta"][MetaKeys.ANALYSIS_DATASET_ID] == "dataset-1"
    assert result["meta"][MetaKeys.RESOLVED_TASK_ID] == task_id
    assert result["meta"][MetaKeys.RESOLVED_TASK_RELATIONSHIP] == "use_as_input"


def test_new_task_resolution_falls_through_to_normal_policy(monkeypatch) -> None:
    node = _fresh_node_module()
    state = _state_with_sql_task("query the rag database for diabetes variables")
    state["meta"][MetaKeys.RESOLVED_TASK_ID] = _task_id(state)
    state["meta"][MetaKeys.RESOLVED_TASK_RELATIONSHIP] = "revision"
    resolver = _resolver("new_task")
    monkeypatch.setattr(node, "resolve_reference_for_turn", resolver)

    result = node.orchestrator_node(state, _LLM(), ["rag_db_qa", "generate_code", "qa", "end"])

    assert len(resolver.calls) == 1
    assert result["next_action"] == "rag_db_qa"
    assert MetaKeys.RESOLVED_TASK_ID not in result["meta"]
    assert MetaKeys.RESOLVED_TASK_RELATIONSHIP not in result["meta"]


def test_unknown_without_reference_need_clears_meta_and_falls_through(monkeypatch) -> None:
    node = _fresh_node_module()
    state = _state_with_sql_task("query the rag database for diabetes variables")
    state["meta"][MetaKeys.RESOLVED_TASK_ID] = _task_id(state)
    state["meta"][MetaKeys.RESOLVED_TASK_KIND] = "db_rag_sql_extraction"
    state["meta"][MetaKeys.RESOLVED_TASK_RELATIONSHIP] = "revision"
    state["meta"][MetaKeys.RESOLVED_TASK_INTENDED_ACTION] = "add_fields"
    state["meta"][MetaKeys.RESOLVED_TASK_USER_MESSAGE_HASH] = "old-hash"
    resolver = _resolver("unknown", needs_reference=False)
    monkeypatch.setattr(node, "resolve_reference_for_turn", resolver)

    result = node.orchestrator_node(state, _LLM(), ["rag_db_qa", "generate_code", "qa", "end"])

    assert len(resolver.calls) == 1
    assert result["next_action"] == "rag_db_qa"
    assert MetaKeys.RESOLVED_TASK_ID not in result["meta"]
    assert MetaKeys.RESOLVED_TASK_KIND not in result["meta"]
    assert MetaKeys.RESOLVED_TASK_RELATIONSHIP not in result["meta"]
    assert MetaKeys.RESOLVED_TASK_INTENDED_ACTION not in result["meta"]
    assert MetaKeys.RESOLVED_TASK_USER_MESSAGE_HASH not in result["meta"]


def test_ambiguous_reference_routes_to_memory_clarification(monkeypatch) -> None:
    node = _fresh_node_module()
    state = _state_with_sql_task("rerun that")
    state = complete_task(
        state,
        kind="db_rag_sql_extraction",
        source_question="Create hypertension subset",
        goal_text="Create hypertension subset",
        label="hypertension subset",
        summary="Reviewed SQL was executed and saved as a subset dataset.",
        artifact_refs={
            "selection_artifact_id": "sel-art-1",
            "sql_candidate_artifact_id": "sql-art-1",
            "dataset_artifact_id": "dataset-1",
        },
    )
    resolver = _resolver("ambiguous")
    monkeypatch.setattr(node, "resolve_reference_for_turn", resolver)

    result = node.orchestrator_node(state, _LLM(), ["clarification", "rag_db_qa", "qa", "end"])

    assert result["next_action"] == "clarification"
    assert result["meta"][MetaKeys.AWAITING_USER_CLARIFICATION] is True
    assert result["meta"][MetaKeys.CLARIFICATION_KIND] == "memory_reference_resolution"
    assert result["meta"][MetaKeys.CLARIFICATION_RETURN_NODE] == "orchestrator"
    pending = result["memory"]["pending_reference_clarification"]
    assert pending["status"] == "awaiting_reply"
    assert [candidate["label"] for candidate in pending["candidates"]] == [
        "diabetes subset",
        "hypertension subset",
    ]
    assert "Task 1: diabetes subset" in result["output"]["qa_response"]
    assert "Task 2: hypertension subset" in result["output"]["qa_response"]
    assert "task_id=None relationship=None routed_node=clarification" in result["observations"]


def test_memory_clarification_route_survives_stale_recurrence_state(monkeypatch) -> None:
    node = _fresh_node_module()
    state = _state_with_sql_task("rerun that")
    state["meta"][MetaKeys.STAGNATION_COUNT] = 4
    resolver = _resolver("ambiguous")
    monkeypatch.setattr(node, "resolve_reference_for_turn", resolver)

    result = node.orchestrator_node(state, _LLM(), ["clarification", "rag_db_qa", "qa", "end"])

    assert result["next_action"] == "clarification"
    assert result["meta"][MetaKeys.AWAITING_USER_CLARIFICATION] is True


def test_memory_clarification_appends_active_clarification_event(monkeypatch) -> None:
    node = _fresh_node_module()
    state = _state_with_sql_task("rerun that")
    resolver = _resolver("ambiguous")
    monkeypatch.setattr(node, "resolve_reference_for_turn", resolver)

    result = node.orchestrator_node(state, _LLM(), ["clarification", "rag_db_qa", "qa", "end"])

    clarification_events = [
        event
        for event in result["artifacts"]["conversation_events"]
        if event["type"] == "clarification"
    ]
    assert len(clarification_events) == 1
    assert clarification_events[0]["text"] == result["output"]["qa_response"]
    assert clarification_events[0]["status"] == "active"


def test_unknown_with_reference_need_routes_to_memory_clarification(monkeypatch) -> None:
    node = _fresh_node_module()
    state = _state_with_sql_task("do that again")
    resolver = _resolver("unknown", needs_reference=True)
    monkeypatch.setattr(node, "resolve_reference_for_turn", resolver)

    result = node.orchestrator_node(state, _LLM(), ["clarification", "rag_db_qa", "qa", "end"])

    assert result["next_action"] == "clarification"
    assert result["meta"][MetaKeys.AWAITING_USER_CLARIFICATION] is True
    assert result["meta"][MetaKeys.CLARIFICATION_KIND] == "memory_reference_resolution"
    assert result["meta"][MetaKeys.CLARIFICATION_RETURN_NODE] == "orchestrator"
    assert result["memory"]["pending_reference_clarification"]["status"] == "awaiting_reply"
    assert "task_id=None relationship=None routed_node=clarification" in result["observations"]


def test_memory_reference_clarification_reply_by_display_ordinal_routes_parent_request() -> None:
    node = _fresh_clarification_module()
    state = _memory_clarification_state("task 2")
    selected_task_id = state["memory"]["task_order"][1]

    result = node.clarification_node(state, _LLM())

    assert MetaKeys.AWAITING_USER_CLARIFICATION not in result["meta"]
    assert MetaKeys.CLARIFICATION_KIND not in result["meta"]
    assert result["memory"]["pending_reference_clarification"] is None
    cached = result["memory"]["last_reference_resolution"]
    assert cached["user_message_hash"] == "original-hash"
    assert cached["result"]["label"] == "resolved"
    assert cached["result"]["task_id"] == selected_task_id
    assert cached["result"]["relationship"] == "inspect_artifact"
    assert cached["result"]["intended_action"] == "test_action"
    assert result["meta"][MetaKeys.RESOLVED_TASK_ID] == selected_task_id
    assert result["meta"][MetaKeys.RESOLVED_TASK_RELATIONSHIP] == "inspect_artifact"
    assert result["meta"][MetaKeys.RESOLVED_TASK_USER_MESSAGE_HASH] == "original-hash"
    assert result["next_action"] is None
    assert result["last_action"] == state["last_action"]


def test_memory_reference_clarification_reply_by_task_id_routes_parent_request() -> None:
    node = _fresh_clarification_module()
    state = _memory_clarification_state("placeholder")
    selected_task_id = state["memory"]["task_order"][0]
    state["messages"][-1] = _HumanMessage(selected_task_id, id="turn-2")

    result = node.clarification_node(state, _LLM())

    assert MetaKeys.AWAITING_USER_CLARIFICATION not in result["meta"]
    assert result["memory"]["pending_reference_clarification"] is None
    cached = result["memory"]["last_reference_resolution"]
    assert cached["user_message_hash"] == "original-hash"
    assert cached["result"]["task_id"] == selected_task_id
    assert cached["result"]["relationship"] == "inspect_artifact"
    assert result["meta"][MetaKeys.RESOLVED_TASK_ID] == selected_task_id
    assert result["next_action"] is None


def test_memory_reference_clarification_reply_without_match_asks_again() -> None:
    node = _fresh_clarification_module()
    state = _memory_clarification_state("the other one")
    pending = state["memory"]["pending_reference_clarification"]

    result = node.clarification_node(state, _LLM())

    assert result["meta"][MetaKeys.AWAITING_USER_CLARIFICATION] is True
    assert result["meta"][MetaKeys.CLARIFICATION_KIND] == "memory_reference_resolution"
    assert result["meta"][MetaKeys.CLARIFICATION_RETURN_NODE] == "orchestrator"
    assert result["memory"]["pending_reference_clarification"] == pending
    assert result["memory"]["last_reference_resolution"] is None
    assert "Which prior task did you mean?" in result["output"]["qa_response"]
    assert "Task 1: diabetes subset" in result["output"]["qa_response"]
    assert "Task 2: hypertension subset" in result["output"]["qa_response"]
    clarification_events = [
        event
        for event in result["artifacts"]["conversation_events"]
        if event["type"] == "clarification"
    ]
    assert len(clarification_events) == 1
    assert clarification_events[0]["text"] == result["output"]["qa_response"]
    assert clarification_events[0]["status"] == "active"


def test_memory_reference_clarification_same_turn_original_message_waits_without_duplicate_event(monkeypatch) -> None:
    orchestrator = _fresh_node_module()
    clarification = _fresh_clarification_module()
    state = _state_with_two_sql_tasks("rerun that")
    resolver = _resolver("ambiguous", relationship="revision")
    monkeypatch.setattr(orchestrator, "resolve_reference_for_turn", resolver)
    routed = orchestrator.orchestrator_node(
        state,
        _LLM(),
        ["clarification", "rag_db_qa", "qa", "end"],
    )
    event_count = len(
        [
            event
            for event in routed["artifacts"]["conversation_events"]
            if event["type"] == "clarification"
        ]
    )

    result = clarification.clarification_node(routed, _LLM())

    assert result["next_action"] in {None, "end"}
    assert result["meta"][MetaKeys.AWAITING_USER_CLARIFICATION] is True
    assert result["memory"]["pending_reference_clarification"]["status"] == "awaiting_reply"
    assert result["output"]["qa_response"] == routed["output"]["qa_response"]
    clarification_events = [
        event
        for event in result["artifacts"]["conversation_events"]
        if event["type"] == "clarification"
    ]
    assert len(clarification_events) == event_count


def test_memory_reference_clarification_reply_routes_on_next_orchestrator_pass() -> None:
    clarification = _fresh_clarification_module()
    orchestrator = _fresh_node_module()
    state = _memory_clarification_state("Task 2", relationship="revision")
    selected_task_id = state["memory"]["task_order"][1]
    state["meta"][MetaKeys.LAST_USER_MESSAGE_HASH] = "reply-hash"

    clarified = clarification.clarification_node(state, _LLM())
    result = orchestrator.orchestrator_node(
        clarified,
        _LLM(),
        ["rag_db_qa", "generate_code", "qa", "end"],
    )

    assert result["next_action"] == "rag_db_qa"
    assert result["meta"][MetaKeys.RESOLVED_TASK_ID] == selected_task_id
    assert result["meta"][MetaKeys.RESOLVED_TASK_RELATIONSHIP] == "revision"
    assert (
        f"task_id={selected_task_id} relationship=revision routed_node=rag_db_qa"
        in result["observations"]
    )


def test_unconsumed_clarification_resolved_meta_preempts_ready_deterministic_review() -> None:
    clarification = _fresh_clarification_module()
    orchestrator = _fresh_node_module()
    state = _memory_clarification_state("Task 2", relationship="revision")
    selected_task_id = state["memory"]["task_order"][1]
    state["meta"][MetaKeys.LAST_USER_MESSAGE_HASH] = "reply-hash"

    clarified = clarification.clarification_node(state, _LLM())
    agents = dict(clarified["agents"])
    rag_state = dict(agents["rag_db_qa"])
    rag_state["pending_column_review"] = {"status": "awaiting_review"}
    agents["rag_db_qa"] = rag_state
    clarified = {**clarified, "agents": agents}

    result = orchestrator.orchestrator_node(
        clarified,
        _LLM(),
        ["rag_db_qa", "human_review_rag_db_column_selection", "qa", "end"],
    )

    assert result["next_action"] == "rag_db_qa"
    assert result["meta"][MetaKeys.RESOLVED_TASK_ID] == selected_task_id
    assert result["meta"][MetaKeys.RESOLVED_TASK_RELATIONSHIP] == "revision"
    assert result["meta"]["resolved_task_meta_consumed"] == "original-hash"
    assert (
        f"task_id={selected_task_id} relationship=revision routed_node=rag_db_qa"
        in result["observations"]
    )


def test_memory_reference_resolved_meta_routes_only_once_after_clarification() -> None:
    clarification = _fresh_clarification_module()
    orchestrator = _fresh_node_module()
    state = _memory_clarification_state("Task 2", relationship="revision")
    selected_task_id = state["memory"]["task_order"][1]
    state["meta"][MetaKeys.LAST_USER_MESSAGE_HASH] = "reply-hash"

    clarified = clarification.clarification_node(state, _LLM())
    first = orchestrator.orchestrator_node(
        clarified,
        _LLM(),
        ["rag_db_qa", "generate_code", "qa", "end"],
    )
    after_action = {
        **first,
        "last_action": "rag_db_qa",
        "next_action": None,
        "orchestrator": {},
    }
    second = orchestrator.orchestrator_node(
        after_action,
        _LLM(),
        ["rag_db_qa", "generate_code", "qa", "end"],
    )

    assert first["next_action"] == "rag_db_qa"
    assert first["meta"][MetaKeys.RESOLVED_TASK_ID] == selected_task_id
    assert first["meta"][MetaKeys.RESOLVED_TASK_RELATIONSHIP] == "revision"
    assert MetaKeys.RESOLVED_TASK_ID not in second["meta"]
    assert MetaKeys.RESOLVED_TASK_RELATIONSHIP not in second["meta"]
    assert (
        f"task_id={selected_task_id} relationship=revision routed_node=rag_db_qa"
        not in second["observations"][len(first["observations"]):]
    )


def test_memory_reference_clarification_reply_without_known_relationship_retries_resolver_for_selected_candidate(monkeypatch) -> None:
    node = _fresh_clarification_module()
    state = _memory_clarification_state("Task 2", relationship=None)
    selected_task_id = state["memory"]["task_order"][1]
    calls = []

    def resolve(selected_state, _llm, user_message, user_message_hash):
        calls.append((selected_state, user_message, user_message_hash))
        assert user_message == "rerun that query"
        assert user_message_hash == f"original-hash:selected:{selected_task_id}"
        assert selected_state["memory"]["task_order"] == [selected_task_id]
        assert list(selected_state["memory"]["completed_tasks"]) == [selected_task_id]
        assert selected_state["memory"]["last_task_id"] == selected_task_id
        assert selected_state["memory"]["last_reference_resolution"] is None
        return {
            "label": "resolved",
            "task_id": selected_task_id,
            "relationship": "revision",
            "intended_action": "add_fields",
            "confidence": "high",
            "needs_reference": False,
            "reason": "selected candidate resolved",
        }

    monkeypatch.setattr(node, "resolve_reference_for_turn", resolve)

    result = node.clarification_node(state, _LLM())

    assert len(calls) == 1
    assert MetaKeys.AWAITING_USER_CLARIFICATION not in result["meta"]
    assert result["memory"]["pending_reference_clarification"] is None
    cached = result["memory"]["last_reference_resolution"]
    assert cached["user_message_hash"] == "original-hash"
    assert cached["result"]["task_id"] == selected_task_id
    assert cached["result"]["relationship"] == "revision"
    assert cached["result"]["intended_action"] == "add_fields"
    assert result["meta"][MetaKeys.RESOLVED_TASK_ID] == selected_task_id
    assert result["meta"][MetaKeys.RESOLVED_TASK_RELATIONSHIP] == "revision"
    assert "What should I do with" not in result["output"].get("qa_response", "")


def test_memory_reference_clarification_selected_candidate_retry_failure_reasks(monkeypatch) -> None:
    node = _fresh_clarification_module()
    state = _memory_clarification_state("Task 2", relationship=None)
    selected_task_id = state["memory"]["task_order"][1]

    def resolve(_selected_state, _llm, _user_message, _user_message_hash):
        return {
            "label": "unknown",
            "task_id": None,
            "relationship": None,
            "intended_action": None,
            "confidence": "low",
            "needs_reference": True,
            "reason": "test unresolved retry",
        }

    monkeypatch.setattr(node, "resolve_reference_for_turn", resolve)

    result = node.clarification_node(state, _LLM())

    assert result["meta"][MetaKeys.AWAITING_USER_CLARIFICATION] is True
    assert result["meta"][MetaKeys.CLARIFICATION_KIND] == "memory_reference_resolution"
    assert result["memory"]["pending_reference_clarification"]["status"] == "awaiting_reply"
    assert "Task 2" in result["output"]["qa_response"]


def test_memory_reference_clarification_selected_candidate_retry_exception_reasks(monkeypatch) -> None:
    node = _fresh_clarification_module()
    state = _memory_clarification_state("Task 2", relationship=None)

    def resolve(_selected_state, _llm, _user_message, _user_message_hash):
        raise ValueError("malformed resolver response")

    monkeypatch.setattr(node, "resolve_reference_for_turn", resolve)

    result = node.clarification_node(state, _LLM())

    assert result["meta"][MetaKeys.AWAITING_USER_CLARIFICATION] is True
    assert result["meta"][MetaKeys.CLARIFICATION_KIND] == "memory_reference_resolution"
    assert result["memory"]["pending_reference_clarification"]["status"] == "awaiting_reply"
    assert "Task 2" in result["output"]["qa_response"]


def test_orchestrator_memory_clarification_pending_preserves_resolver_relationship(monkeypatch) -> None:
    node = _fresh_node_module()
    state = _state_with_two_sql_tasks("rerun that")
    resolver = _resolver("ambiguous", relationship="revision")
    monkeypatch.setattr(node, "resolve_reference_for_turn", resolver)

    result = node.orchestrator_node(state, _LLM(), ["clarification", "rag_db_qa", "qa", "end"])

    pending = result["memory"]["pending_reference_clarification"]
    assert pending["relationship"] == "revision"
    assert pending["result"]["relationship"] == "revision"
