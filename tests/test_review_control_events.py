from __future__ import annotations

import importlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

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

    prompts_mod = ModuleType("langchain_core.prompts")
    prompts_mod.ChatPromptTemplate = _ChatPromptTemplate
    prompts_mod.MessagesPlaceholder = _MessagesPlaceholder
    prompts_mod.FewShotChatMessagePromptTemplate = _FewShotChatMessagePromptTemplate

    langchain_core_mod = ModuleType("langchain_core")
    langchain_core_mod.messages = messages_mod
    langchain_core_mod.prompts = prompts_mod

    sys.modules["langgraph.types"] = langgraph_types
    sys.modules["langgraph.graph.message"] = graph_message_mod
    sys.modules["langchain_core"] = langchain_core_mod
    sys.modules["langchain_core.messages"] = messages_mod
    sys.modules["langchain_core.prompts"] = prompts_mod


class _FormattedPrompt:
    def __init__(self, rendered):
        self._rendered = rendered

    def to_messages(self):
        return self._rendered


class _PromptTemplate:
    def __init__(self, messages):
        self._messages = [message for message in messages if isinstance(message, tuple)]

    def format_prompt(self, **kwargs):
        return _FormattedPrompt(
            [
                {"role": role, "content": template.format(**kwargs)}
                for role, template in self._messages
            ]
        )


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


class _OrchestratorHumanMessage:
    type = "human"

    def __init__(self, content: str, id: str | None = None):
        self.content = content
        self.id = id or ""


class _StaticLLM:
    def __init__(self, action: str = "end"):
        self.action = action

    def invoke(self, _messages):
        return SimpleNamespace(
            content=json.dumps({"action": self.action, "thought": "test route"})
        )


def _install_orchestrator_stubs() -> None:
    messages_mod = ModuleType("langchain_core.messages")
    messages_mod.BaseMessage = object
    messages_mod.HumanMessage = _OrchestratorHumanMessage
    messages_mod.AIMessage = object

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


@contextmanager
def _isolated_orchestrator_stubs():
    module_names = (
        "langchain_core",
        "langchain_core.messages",
        "langchain_core.prompts",
        "langgraph.graph.message",
        "prompts.planner_prompt",
        "graph.nodes.orchestrator",
        "graph.nodes.orchestrator.node",
        "graph.nodes.orchestrator.planner",
        "graph.nodes.orchestrator.policy",
        "graph.nodes.orchestrator.action_mask",
        "graph.nodes.orchestrator.context_builder",
        "graph.nodes.orchestrator.state_logic",
        "graph.state",
    )
    sentinel = object()
    previous = {name: sys.modules.get(name, sentinel) for name in module_names}
    try:
        yield
    finally:
        for name, module in previous.items():
            if module is sentinel:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        importlib.invalidate_caches()


def _fresh_orchestrator_module():
    _install_orchestrator_stubs()
    for mod in (
        "prompts.planner_prompt",
        "graph.nodes.orchestrator",
        "graph.nodes.orchestrator.node",
        "graph.nodes.orchestrator.planner",
        "graph.nodes.orchestrator.policy",
        "graph.nodes.orchestrator.action_mask",
        "graph.nodes.orchestrator.context_builder",
        "graph.nodes.orchestrator.state_logic",
        "graph.state",
    ):
        sys.modules.pop(mod, None)
    return importlib.import_module("graph.nodes.orchestrator.node")


def _orchestrator_cancel_state(last_action: str, human_review: dict) -> dict:
    return {
        "messages": [_OrchestratorHumanMessage("write python code for this analysis", id="turn-1")],
        "output": {"generated_code": "print(1)"},
        "artifacts": {
            "conversation_events": [],
            "conversation_events_version": 1,
            "artifact_manifest_version": 1,
        },
        "next_action": None,
        "last_action": last_action,
        "observations": [],
        "orchestrator": {},
        "planner": {},
        "agents": {
            "executor": {"run_status": "idle"},
            "human_review": human_review,
        },
        "node_data": {},
        "meta": {
            "execution_ticket_hash": "hash-1",
            "final_approved_code_hash": "hash-1",
            "error_recovery_active": True,
            "workflow_trace": [last_action],
        },
        "memory": {},
    }


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
    cancel_messages = [message for message in history if message.content == module.CANCEL_REVIEW_MESSAGE]
    assert len(cancel_messages) == 1
    assert cancel_messages[0].type == "ai"


def test_human_review_before_run_cancel_clears_live_code_state() -> None:
    _install_stubs(decision="cancel", suggestion="do not use this")
    for mod in (
        "graph.nodes.human_review_before_run",
        "graph.nodes.human_review_cancel",
        "graph.state",
        "graph.nodes.state_helpers",
    ):
        sys.modules.pop(mod, None)
    module = importlib.import_module("graph.nodes.human_review_before_run")

    state = {
        "messages": [],
        "output": {"generated_code": "print(1)", "code_summary": "summary"},
        "meta": {
            "current_code_hash": "hash-1",
            "execution_ticket_hash": "hash-1",
            "final_approved_code_hash": "hash-1",
            "error_recovery_active": True,
            "last_user_message_hash": "u-cancel-run",
        },
        "agents": {"executor": {"run_status": "pending"}},
    }

    updated = module.human_review_before_run_node(state)

    assert updated["agents"]["human_review"]["before_run_decision"] == "cancel"
    assert updated["agents"]["human_review"]["approved_code_hash"] is None
    assert updated["agents"]["executor"]["run_status"] == "idle"
    assert "generated_code" not in updated["output"]
    assert "execution_ticket_hash" not in updated["meta"]
    assert "final_approved_code_hash" not in updated["meta"]
    assert "error_recovery_active" not in updated["meta"]
    assert len(updated["messages"]) == 1
    assert updated["messages"][0].content.startswith("Cancelled the pending review")
    assert updated["artifacts"]["conversation_events"][-1]["decision"] == "cancel"


def test_human_review_after_error_cancel_clears_retry_state_without_feedback_message() -> None:
    _install_stubs(decision="cancel", suggestion="try again anyway")
    for mod in (
        "graph.nodes.human_review_after_error",
        "graph.nodes.human_review_cancel",
        "graph.state",
        "graph.nodes.state_helpers",
    ):
        sys.modules.pop(mod, None)
    module = importlib.import_module("graph.nodes.human_review_after_error")

    state = {
        "messages": [],
        "output": {
            "generated_code": "print(1)",
            "error": {"category": "retryable_code", "type": "ValueError", "message": "bad column"},
        },
        "meta": {
            "error_iterations": 5,
            "current_code_hash": "hash-1",
            "execution_ticket_hash": "hash-1",
            "final_approved_code_hash": "hash-1",
            "error_recovery_active": True,
            "last_user_message_hash": "u-cancel-error",
        },
        "agents": {"executor": {"run_status": "error"}},
    }

    updated = module.human_review_after_error_node(state)

    assert updated["agents"]["human_review"]["after_error_decision"] == "cancel"
    assert updated["agents"]["executor"]["run_status"] == "idle"
    assert updated["meta"]["error_iterations"] == 0
    assert "generated_code" not in updated["output"]
    assert "execution_ticket_hash" not in updated["meta"]
    assert "final_approved_code_hash" not in updated["meta"]
    assert "error_recovery_active" not in updated["meta"]
    assert len(updated["messages"]) == 1
    assert "try again anyway" not in updated["messages"][0].content
    assert updated["artifacts"]["conversation_events"][-1]["decision"] == "cancel"


def test_human_review_before_output_cancel_does_not_append_final_result() -> None:
    _install_stubs(decision="cancel", suggestion="change the output")
    for mod in (
        "graph.nodes.human_review_before_output",
        "graph.nodes.human_review_cancel",
        "graph.state",
        "graph.nodes.state_helpers",
    ):
        sys.modules.pop(mod, None)
    module = importlib.import_module("graph.nodes.human_review_before_output")

    state = {
        "messages": [],
        "output": {
            "generated_code": "print(1)",
            "text": "result text",
            "code_summary": "summary",
        },
        "meta": {
            "current_code_hash": "hash-1",
            "execution_ticket_hash": "hash-1",
            "final_approved_code_hash": "old-hash",
            "error_recovery_active": True,
            "last_user_message_hash": "u-cancel-final",
        },
        "agents": {"executor": {"run_status": "ok"}},
        "artifacts": {"files": {}},
    }

    updated = module.human_review_before_output_node(state)

    assert updated["agents"]["human_review"]["final_decision"] == "cancel"
    assert updated["agents"]["executor"]["run_status"] == "idle"
    assert "execution_ticket_hash" not in updated["meta"]
    assert "final_approved_code_hash" not in updated["meta"]
    assert "error_recovery_active" not in updated["meta"]
    assert len(updated["messages"]) == 1
    assert "result text" not in updated["messages"][0].content
    assert updated["artifacts"]["conversation_events"][-1]["review_kind"] == "final_review"
    assert updated["artifacts"]["conversation_events"][-1]["decision"] == "cancel"


def test_rag_db_column_review_cancel_clears_live_review_pointers() -> None:
    _install_stubs(decision="cancel", suggestion="ignore this")
    for mod in (
        "graph.nodes.human_review_rag_db_column_selection",
        "graph.nodes.human_review_cancel",
        "graph.state",
        "graph.nodes.state_helpers",
    ):
        sys.modules.pop(mod, None)
    module = importlib.import_module("graph.nodes.human_review_rag_db_column_selection")

    state = {
        "messages": [],
        "output": {},
        "meta": {"last_user_message_hash": "u-rag-col-cancel"},
        "artifacts": {
            "files": {
                "sel-1": {
                    "kind": "db_rag_column_selection",
                    "artifact_id": "sel-1",
                    "created_at": "2026-05-07T00:00:00+00:00",
                    "producer": "rag_db_qa",
                    "mime": "application/json",
                    "summary": "Column selection awaiting review",
                    "content": {
                        "status": "awaiting_review",
                        "goal_text": "extract diabetes rows",
                        "source_question": "extract diabetes rows",
                    }
                }
            }
        },
        "agents": {
            "rag_db_qa": {
                "pending_column_review_artifact_id": "sel-1",
                "approved_column_selection_artifact_id": "sel-1",
                "pending_sql_candidate_artifact_id": "sql-1",
                "pending_column_review": {"status": "awaiting_review"},
                "pending_sql_candidate": {"status": "prepared"},
                "sql_review_approved_artifact_id": "sql-1",
                "thread_status": "awaiting_column_review",
                "active_thread": True,
            }
        },
    }

    updated = module.human_review_rag_db_column_selection_node(state)
    rag_state = updated["agents"]["rag_db_qa"]

    assert updated["artifacts"]["files"]["sel-1"]["content"]["status"] == "cancelled"
    assert rag_state["pending_column_review_artifact_id"] is None
    assert rag_state["approved_column_selection_artifact_id"] is None
    assert rag_state["pending_sql_candidate_artifact_id"] is None
    assert rag_state["pending_column_review"] is None
    assert rag_state["pending_sql_candidate"] is None
    assert "sql_review_approved_artifact_id" not in rag_state
    assert rag_state["thread_status"] == "cancelled"
    assert rag_state["active_thread"] is False
    assert updated["artifacts"]["conversation_events"][-1]["decision"] == "cancel"


def test_rag_db_sql_review_cancel_clears_sql_and_selection_pointers() -> None:
    _install_stubs(decision="cancel", suggestion="ignore this")
    for mod in (
        "graph.nodes.human_review_rag_db_sql_execution",
        "graph.nodes.human_review_cancel",
        "graph.nodes.rag_db_qa",
        "graph.state",
        "graph.nodes.state_helpers",
    ):
        sys.modules.pop(mod, None)
    module = importlib.import_module("graph.nodes.human_review_rag_db_sql_execution")

    state = {
        "messages": [],
        "output": {},
        "meta": {"last_user_message_hash": "u-rag-sql-cancel"},
        "artifacts": {
            "files": {
                "sel-1": {
                    "kind": "db_rag_column_selection",
                    "artifact_id": "sel-1",
                    "created_at": "2026-05-07T00:00:00+00:00",
                    "producer": "rag_db_qa",
                    "mime": "application/json",
                    "summary": "Approved column selection",
                    "content": {
                        "status": "approved",
                        "selection_id": "selection-1",
                        "tables": ["Form 2A"],
                        "columns": [{"table": "Form 2A", "column": "IC_AGE"}],
                    }
                },
                "sql-1": {
                    "kind": "db_rag_sql_candidate",
                    "artifact_id": "sql-1",
                    "created_at": "2026-05-07T00:00:00+00:00",
                    "producer": "rag_db_qa",
                    "mime": "application/json",
                    "summary": "Prepared SQL candidate",
                    "content": {
                        "status": "prepared",
                        "selection_artifact_id": "sel-1",
                        "selection_id": "selection-1",
                        "tables": ["Form 2A"],
                        "columns": [{"table": "Form 2A", "column": "IC_AGE"}],
                        "sql": "select 1",
                    }
                },
            }
        },
        "agents": {
            "rag_db_qa": {
                "pending_sql_candidate_artifact_id": "sql-1",
                "pending_sql_candidate": {"status": "prepared"},
                "approved_column_selection_artifact_id": "sel-1",
                "pending_column_review_artifact_id": "sel-1",
                "pending_column_review": {"status": "approved"},
                "sql_review_approved_artifact_id": "sql-1",
                "thread_status": "awaiting_sql_review",
                "active_thread": True,
            }
        },
    }

    updated = module.human_review_rag_db_sql_execution_node(state, service=object())
    rag_state = updated["agents"]["rag_db_qa"]

    assert updated["artifacts"]["files"]["sql-1"]["content"]["status"] == "cancelled"
    assert rag_state["pending_sql_candidate_artifact_id"] is None
    assert rag_state["pending_sql_candidate"] is None
    assert rag_state["approved_column_selection_artifact_id"] is None
    assert rag_state["pending_column_review_artifact_id"] is None
    assert rag_state["pending_column_review"] is None
    assert "sql_review_approved_artifact_id" not in rag_state
    assert rag_state["thread_status"] == "cancelled"
    assert rag_state["active_thread"] is False
    assert updated["artifacts"]["conversation_events"][-1]["review_kind"] == "rag_db_sql_execution"
    assert updated["artifacts"]["conversation_events"][-1]["decision"] == "cancel"


def test_consume_after_error_cancel_is_terminal_without_loop_guard_bypass() -> None:
    for mod in (
        "graph.nodes.orchestrator.state_logic",
        "graph.state",
    ):
        sys.modules.pop(mod, None)
    module = importlib.import_module("graph.nodes.orchestrator.state_logic")
    state_module = importlib.import_module("graph.state")
    meta_keys = state_module.MetaKeys

    output = {}
    agents = {
        "human_review": {
            "after_error_decision": "cancel",
            "before_run_decision": "approve",
            "approved_code_hash": "hash-1",
        }
    }
    meta = {
        meta_keys.EXECUTION_TICKET_HASH: "hash-1",
        meta_keys.ERROR_RECOVERY_ACTIVE: True,
    }

    _updated_output, updated_agents, updated_meta, action = module._consume_after_error_decision(
        output,
        agents,
        meta,
    )

    assert action == "end"
    assert updated_agents["human_review"]["after_error_decision"] is None
    assert updated_agents["human_review"]["before_run_decision"] is None
    assert updated_agents["human_review"]["approved_code_hash"] is None
    assert meta_keys.EXECUTION_TICKET_HASH not in updated_meta
    assert meta_keys.ERROR_RECOVERY_ACTIVE not in updated_meta
    assert "generate_code" not in updated_meta.get(meta_keys.LOOP_GUARD_BYPASS_ACTIONS, [])


def test_consume_after_error_feedback_routes_to_generate_code_with_loop_guard_bypass() -> None:
    for mod in (
        "graph.nodes.orchestrator.state_logic",
        "graph.state",
    ):
        sys.modules.pop(mod, None)
    module = importlib.import_module("graph.nodes.orchestrator.state_logic")
    state_module = importlib.import_module("graph.state")
    meta_keys = state_module.MetaKeys

    output = {}
    agents = {"human_review": {"after_error_decision": "feedback"}}
    meta = {}

    _updated_output, updated_agents, updated_meta, action = module._consume_after_error_decision(
        output,
        agents,
        meta,
    )

    assert action == "generate_code"
    assert updated_agents["human_review"]["after_error_decision"] is None
    assert "generate_code" in updated_meta.get(meta_keys.LOOP_GUARD_BYPASS_ACTIONS, [])


def test_consume_before_run_cancel_clears_decision_after_one_consume() -> None:
    for mod in (
        "graph.nodes.orchestrator.state_logic",
        "graph.state",
    ):
        sys.modules.pop(mod, None)
    module = importlib.import_module("graph.nodes.orchestrator.state_logic")
    state_module = importlib.import_module("graph.state")
    meta_keys = state_module.MetaKeys

    output = {"generated_code": "print(1)"}
    agents = {
        "executor": {"run_status": "pending"},
        "human_review": {
            "before_run_decision": "cancel",
            "approved_code_hash": "hash-1",
        },
    }
    meta = {
        meta_keys.EXECUTION_TICKET_HASH: "hash-1",
        meta_keys.FINAL_APPROVED_CODE_HASH: "hash-1",
        meta_keys.ERROR_RECOVERY_ACTIVE: True,
    }

    updated_output, updated_agents, updated_meta, cancelled = module._consume_before_run_cancel(
        output,
        agents,
        meta,
    )
    _second_output, _second_agents, _second_meta, second_cancelled = module._consume_before_run_cancel(
        updated_output,
        updated_agents,
        updated_meta,
    )

    assert cancelled is True
    assert second_cancelled is False
    assert updated_agents["human_review"]["before_run_decision"] is None
    assert updated_agents["human_review"]["approved_code_hash"] is None
    assert updated_agents["executor"]["run_status"] == "idle"
    assert "generated_code" not in updated_output
    assert meta_keys.EXECUTION_TICKET_HASH not in updated_meta
    assert meta_keys.FINAL_APPROVED_CODE_HASH not in updated_meta
    assert meta_keys.ERROR_RECOVERY_ACTIVE not in updated_meta


def test_consume_final_review_cancel_clears_decision_after_one_consume() -> None:
    for mod in (
        "graph.nodes.orchestrator.state_logic",
        "graph.state",
    ):
        sys.modules.pop(mod, None)
    module = importlib.import_module("graph.nodes.orchestrator.state_logic")
    state_module = importlib.import_module("graph.state")
    meta_keys = state_module.MetaKeys

    output = {"generated_code": "print(1)"}
    agents = {
        "executor": {"run_status": "ok"},
        "human_review": {
            "final_decision": "cancel",
            "before_run_decision": "approve",
            "approved_code_hash": "hash-1",
        },
    }
    meta = {
        meta_keys.EXECUTION_TICKET_HASH: "hash-1",
        meta_keys.FINAL_APPROVED_CODE_HASH: "hash-1",
        meta_keys.ERROR_RECOVERY_ACTIVE: True,
    }

    updated_output, updated_agents, updated_meta, cancelled = module._consume_final_review_cancel(
        output,
        agents,
        meta,
    )
    _second_output, _second_agents, _second_meta, second_cancelled = module._consume_final_review_cancel(
        updated_output,
        updated_agents,
        updated_meta,
    )

    assert cancelled is True
    assert second_cancelled is False
    assert updated_agents["human_review"]["final_decision"] is None
    assert updated_agents["human_review"]["before_run_decision"] is None
    assert updated_agents["human_review"]["approved_code_hash"] is None
    assert updated_agents["executor"]["run_status"] == "idle"
    assert "generated_code" not in updated_output
    assert meta_keys.EXECUTION_TICKET_HASH not in updated_meta
    assert meta_keys.FINAL_APPROVED_CODE_HASH not in updated_meta
    assert meta_keys.ERROR_RECOVERY_ACTIVE not in updated_meta


def test_orchestrator_before_run_cancel_routes_to_end_without_planner_fallback() -> None:
    with _isolated_orchestrator_stubs():
        module = _fresh_orchestrator_module()
        state = _orchestrator_cancel_state(
            "human_review_before_run",
            {
                "before_run_decision": "cancel",
                "approved_code_hash": None,
            },
        )

        updated = module.orchestrator_node(
            state,
            _StaticLLM("end"),
            ["generate_code", "execute_code", "qa", "end"],
        )

    assert updated["next_action"] == "end"
    assert updated["orchestrator"]["next_action"] == "end"
    assert "generated_code" not in updated["output"]
    assert "execution_ticket_hash" not in updated["meta"]
    assert "final_approved_code_hash" not in updated["meta"]
    assert "error_recovery_active" not in updated["meta"]
    assert any("before-run cancel" in item for item in updated["observations"])


def test_orchestrator_after_error_cancel_routes_to_end_without_planner_fallback() -> None:
    with _isolated_orchestrator_stubs():
        module = _fresh_orchestrator_module()
        state = _orchestrator_cancel_state(
            "human_review_after_error",
            {
                "after_error_decision": "cancel",
                "before_run_decision": "approve",
                "approved_code_hash": "hash-1",
            },
        )

        updated = module.orchestrator_node(
            state,
            _StaticLLM("end"),
            ["generate_code", "qa", "end"],
        )

    assert updated["next_action"] == "end"
    assert updated["orchestrator"]["next_action"] == "end"
    assert updated["agents"]["human_review"]["after_error_decision"] is None
    assert "execution_ticket_hash" not in updated["meta"]
    assert "error_recovery_active" not in updated["meta"]
    assert "generate_code" not in updated["meta"].get("loop_guard_bypass_actions", [])
    assert any("after-error cancel" in item for item in updated["observations"])


def test_orchestrator_after_error_feedback_still_routes_to_generate_code() -> None:
    with _isolated_orchestrator_stubs():
        module = _fresh_orchestrator_module()
        state = _orchestrator_cancel_state(
            "human_review_after_error",
            {
                "after_error_decision": "feedback",
                "before_run_decision": None,
                "approved_code_hash": None,
            },
        )

        updated = module.orchestrator_node(
            state,
            _StaticLLM("end"),
            ["generate_code", "qa", "end"],
        )

    assert updated["next_action"] == "generate_code"
    assert "generate_code" in updated["meta"].get("loop_guard_bypass_actions", [])


def test_orchestrator_final_review_cancel_routes_to_end_with_live_generated_code() -> None:
    with _isolated_orchestrator_stubs():
        module = _fresh_orchestrator_module()
        state = _orchestrator_cancel_state(
            "human_review_before_output",
            {
                "final_decision": "cancel",
                "before_run_decision": None,
                "approved_code_hash": None,
            },
        )

        updated = module.orchestrator_node(
            state,
            _StaticLLM("end"),
            ["human_review_before_run", "generate_code", "qa", "end"],
        )

    assert updated["next_action"] == "end"
    assert updated["orchestrator"]["next_action"] == "end"
    assert updated["agents"]["executor"]["run_status"] == "idle"
    assert "execution_ticket_hash" not in updated["meta"]
    assert "final_approved_code_hash" not in updated["meta"]
    assert "error_recovery_active" not in updated["meta"]
    assert any("final-review cancel" in item for item in updated["observations"])
