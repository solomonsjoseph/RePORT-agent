import os
import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from utils.dataset_artifacts import get_active_dataset_artifact, load_dataset_artifact
from .state import AgentState, MetaKeys
from .routing import route_by_next_action
from .nodes.node_registry import validate_registry

from .nodes.orchestrator import orchestrator_node
from .nodes.generate_code import generate_code_node
from .nodes.execute_code import execute_code_node
from .nodes.error_handler import error_handler_node
from .nodes.qa import qa_node
from .nodes.clarification import clarification_node
from .nodes.tool_handler import tool_handler_node
from .nodes.terminal_execution_error import terminal_execution_error_node
from .nodes.human_review_before_run import human_review_before_run_node
from .nodes.human_review_after_error import human_review_after_error_node
from .nodes.human_review_before_output import human_review_before_output_node
from .nodes.human_review_rag_db_column_selection import human_review_rag_db_column_selection_node
from .nodes.human_review_rag_db_sql_execution import human_review_rag_db_sql_execution_node
from .nodes.rag_db_qa import rag_db_qa_node
from db_rag.config import resolve_db_rag_reranker_model
from db_rag.service import DbRagService

def _run_and_mark(node_name, fn):
    def _wrapped(state):
        updated_state = fn(state)
        if not isinstance(updated_state, dict):
            updated_state = {}
        merged_state = {**state, **updated_state}

        orchestrator_state = dict(merged_state.get("orchestrator", {}))
        orchestrator_state.pop("next_action", None)
        meta = dict(merged_state.get("meta", {}))
        # Keep this function self-contained so it can be unit-tested by loading
        # only the function body via ``ast`` (without module-level imports).
        workflow_trace = list(meta.get("workflow_trace", []))
        workflow_trace.append(node_name)
        meta["workflow_trace"] = workflow_trace[-100:]

        # Consume one-shot loop-guard bypass for the action that just ran.
        bypass_actions = [
            a for a in meta.get("loop_guard_bypass_actions", [])
            if a != node_name
        ]
        if bypass_actions:
            meta["loop_guard_bypass_actions"] = bypass_actions
        else:
            meta.pop("loop_guard_bypass_actions", None)

        return {
            **merged_state,
            "next_action": None,
            "last_action": node_name,
            "orchestrator": orchestrator_state,
            "planner": dict(merged_state.get("planner", {})),
            "meta": meta,
        }

    return _wrapped


def _dispatch_rag_db_qa_with_override(
    state,
    *,
    llm,
    provider,
    service,
    reranker_model=None,
    rag_db_qa_node_fn=rag_db_qa_node,
):
    meta = dict(state.get("meta") or {})
    question_override = meta.get(MetaKeys.RAG_DB_QUESTION_OVERRIDE)
    updated = rag_db_qa_node_fn(
        state,
        llm,
        provider=provider,
        service=service,
        reranker_model=reranker_model,
        question_override=question_override if isinstance(question_override, str) else None,
    )
    if not isinstance(updated, dict):
        return updated
    updated_meta = dict(updated.get("meta") or {})
    updated_meta.pop(MetaKeys.RAG_DB_QUESTION_OVERRIDE, None)
    return {**updated, "meta": updated_meta}


def build_graph(llm, provider, db_path):
    workflow = StateGraph(AgentState)
    db_rag_service = DbRagService(llm=llm)
    db_rag_reranker_model = resolve_db_rag_reranker_model()

    context_bundle = {
        "runtime_datasets": True,
        "provider": provider,
        "db_rag_service": db_rag_service,
        "db_rag_reranker_model": db_rag_reranker_model,
    }

    def _resolve_analysis_dataframe(state):
        artifact = get_active_dataset_artifact(state)
        meta = dict(state.get("meta") or {})
        datasets = dict((state.get("artifacts") or {}).get("datasets") or {})
        selected_id = meta.get("analysis_dataset_id")
        if selected_id and selected_id in datasets:
            artifact = datasets[selected_id]
        if not artifact:
            return None
        df, _schema = load_dataset_artifact(artifact)
        return df

    action_nodes = {
        # Risk-2 fix: validate_registry() is called with the exact set of node names
        # registered here.  Any mismatch (node in registry but not wired, or wired
        # but missing a NodeDefinition) raises an AssertionError at startup.
        "generate_code": _run_and_mark("generate_code", lambda s: generate_code_node(s, llm, context_bundle)),
        "execute_code": _run_and_mark("execute_code", lambda s: execute_code_node(s, _resolve_analysis_dataframe)),
        "error_handler": _run_and_mark("error_handler", lambda s: error_handler_node(s, llm, context_bundle)),
        "terminal_execution_error": _run_and_mark("terminal_execution_error", terminal_execution_error_node),
        "clarification": _run_and_mark("clarification", lambda s: clarification_node(s, llm, context_bundle)),
        "human_review_after_error": _run_and_mark("human_review_after_error", human_review_after_error_node),
        "human_review_before_run": _run_and_mark("human_review_before_run", human_review_before_run_node),
        "human_review_before_output": _run_and_mark(
            "human_review_before_output",
            human_review_before_output_node,
        ),
        "tool_handler": _run_and_mark("tool_handler", tool_handler_node),
        "qa": _run_and_mark("qa", lambda s: qa_node(s, llm, context_bundle)),
        "rag_db_qa": _run_and_mark(
            "rag_db_qa",
            lambda s: _dispatch_rag_db_qa_with_override(
                s,
                llm=llm,
                provider=provider,
                service=db_rag_service,
                reranker_model=db_rag_reranker_model,
            ),
        ),
        "human_review_rag_db_column_selection": _run_and_mark(
            "human_review_rag_db_column_selection",
            human_review_rag_db_column_selection_node,
        ),
        "human_review_rag_db_sql_execution": _run_and_mark(
            "human_review_rag_db_sql_execution",
            lambda s: human_review_rag_db_sql_execution_node(s, db_rag_service),
        ),
    }
    available_actions = [*action_nodes.keys(), "end"]
    validate_registry(known_action_names=list(action_nodes.keys()))

    workflow.add_node(
        "orchestrator",
        lambda s: orchestrator_node(s, llm, available_actions)
    )
    
    for node_name, node_fn in action_nodes.items():
        workflow.add_node(node_name, node_fn)


    # Orchestrator-driven control flow
    workflow.add_edge(START, "orchestrator")
    workflow.add_conditional_edges(
        "orchestrator",
        route_by_next_action,
        {
            **{name: name for name in action_nodes},
            END: END,
        }
    )

    for node_name in action_nodes:
        workflow.add_edge(node_name, "orchestrator")

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return workflow.compile(
        checkpointer=checkpointer
    )
