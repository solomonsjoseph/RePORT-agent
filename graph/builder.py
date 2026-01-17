import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from utils.context import build_context
from .state import AgentState
from .routing import (
    route_after_generate, 
    route_after_execute,
    route_after_human_review_before_run,
    route_after_human_review_final)

from .nodes.generate_code import generate_code_node
from .nodes.execute_code import execute_code_node
from .nodes.handle_error import handle_error_node
from .nodes.human_checkpoints import (
    human_review_before_run_node, 
    human_review_final_node)

def build_graph(llm, df, schema, db_path):
    workflow = StateGraph(AgentState)
    context = build_context(df, schema)

    workflow.add_node(
        "generate_code",
        lambda s: generate_code_node(s, llm, context)
    )
    workflow.add_node(
        "execute_code",
        lambda s: execute_code_node(s, df)
    )
    workflow.add_node(
        "handle_error",
        lambda s: handle_error_node(s, llm, context)
    )

    # Interruptable review nodes
    workflow.add_node("human_review_before_run", human_review_before_run_node)
    workflow.add_node("human_review_final", human_review_final_node)

    # Normal control flow
    workflow.add_edge(START, "generate_code")
    workflow.add_edge("generate_code", "human_review_before_run")
    workflow.add_conditional_edges(
        "human_review_before_run",
        route_after_human_review_before_run,
        {
            "execute_code": "execute_code",
            "generate_code": "generate_code"
        }
    )
    workflow.add_conditional_edges(
        "execute_code",
        route_after_execute,
        {
            "handle_error": "handle_error",
            "human_review_final": "human_review_final",
        }
    )
    workflow.add_edge("handle_error", "execute_code")
    workflow.add_conditional_edges(
        "human_review_final",
        route_after_human_review_final,
        {
            "generate_code": "generate_code",
            END: END,
        }
    )

    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return workflow.compile(
        checkpointer=checkpointer
    )
