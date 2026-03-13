import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from utils.context import build_context
from .state import AgentState
from .routing import route_by_next_action

from .nodes.orchestrator import orchestrator_node
from .nodes.generate_code import generate_code_node
from .nodes.execute_code import execute_code_node
from .nodes.error_handler import error_handler_node
from .nodes.qa import qa_node
from .nodes.tool_handler import tool_handler_node
from .nodes.human_review_before_run import human_review_before_run_node
from .nodes.human_review_after_error import human_review_after_error_node
from .nodes.human_review_final import human_review_final_node

def _run_and_mark(node_name, fn):
    def _wrapped(state):
        updated_state = fn(state)
        if not isinstance(updated_state, dict):
            updated_state = dict(state)

        orchestrator_state = dict(updated_state.get("orchestrator", {}))
        orchestrator_state.pop("next_action", None)
        meta = dict(updated_state.get("meta", {}))
        workflow_trace = list(meta.get("workflow_trace", []))
        workflow_trace.append(node_name)
        meta["workflow_trace"] = workflow_trace[-100:]

        return {
            **updated_state,
            "next_action": None,
            "last_action": node_name,
            "orchestrator": orchestrator_state,
            "meta": meta,
        }

    return _wrapped


def build_graph(llm, df, schema, db_path):
    workflow = StateGraph(AgentState)
    context = build_context(df, schema)
    action_nodes = {
        "generate_code": _run_and_mark("generate_code", lambda s: generate_code_node(s, llm, context)),
        "execute_code": _run_and_mark("execute_code", lambda s: execute_code_node(s, df)),
        "error_handler": _run_and_mark("error_handler", lambda s: error_handler_node(s, llm, context)),
        "human_review_after_error": _run_and_mark("human_review_after_error", human_review_after_error_node),
        "human_review_before_run": _run_and_mark("human_review_before_run", human_review_before_run_node),
        "human_review_final": _run_and_mark("human_review_final", human_review_final_node),
        "tool_handler": _run_and_mark("tool_handler", tool_handler_node),
        "qa": _run_and_mark("qa", lambda s: qa_node(s, llm, context)),
    }
    available_actions = [*action_nodes.keys(), "end"]

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

    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return workflow.compile(
        checkpointer=checkpointer
    )
