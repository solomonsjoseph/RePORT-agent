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

def build_graph(llm, df, schema, db_path):
    workflow = StateGraph(AgentState)
    context = build_context(df, schema)

    available_actions = [
        "generate_code",
        "execute_code",
        "error_handler",
        "human_review_after_error",
        "human_review_before_run",
        "human_review_final",
        "tool_handler",
        "qa",
        "end",
    ]

    workflow.add_node(
        "orchestrator",
        lambda s: orchestrator_node(s, llm, available_actions)
    )
    workflow.add_node(
        "generate_code",
        lambda s: generate_code_node(s, llm, context)
    )
    workflow.add_node(
        "execute_code",
        lambda s: execute_code_node(s, df)
    )
    workflow.add_node(
        "error_handler",
        lambda s: error_handler_node(s, llm, context)
    )
    workflow.add_node("human_review_after_error", human_review_after_error_node)
    workflow.add_node("tool_handler", tool_handler_node)
    workflow.add_node(
        "qa",
        lambda s: qa_node(s, llm)
    )

    # Interruptable review nodes
    workflow.add_node("human_review_before_run", human_review_before_run_node)
    workflow.add_node("human_review_final", human_review_final_node)

    # Orchestrator-driven control flow
    workflow.add_edge(START, "orchestrator")
    workflow.add_conditional_edges(
        "orchestrator",
        route_by_next_action,
        {
            "generate_code": "generate_code",
            "execute_code": "execute_code",
            "error_handler": "error_handler",
            "human_review_after_error": "human_review_after_error",
            "human_review_before_run": "human_review_before_run",
            "human_review_final": "human_review_final",
            "tool_handler": "tool_handler",
            "qa": "qa",
            END: END,
        }
    )

    for node_name in [
        "generate_code",
        "execute_code",
        "error_handler",
        "human_review_after_error",
        "human_review_before_run",
        "human_review_final",
        "tool_handler",
        "qa",
    ]:
        workflow.add_edge(node_name, "orchestrator")

    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return workflow.compile(
        checkpointer=checkpointer
    )
