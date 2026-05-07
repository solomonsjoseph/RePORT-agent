from __future__ import annotations

from graph.memory import complete_task


class _HumanMessage:
    type = "human"

    def __init__(self, content: str):
        self.content = content


class _AIMessage:
    type = "ai"

    def __init__(self, content: str):
        self.content = content


def _state() -> dict:
    state = {
        "messages": [
            _HumanMessage("Query my database for age, gender, and TB outcome."),
            _AIMessage("Read-only SQL execution completed. Saved dataset id: subset-1"),
            _HumanMessage("Now find the relationship between gender and TB outcome."),
        ],
        "output": {},
        "artifacts": {
            "files": {
                "sql-art-1": {
                    "artifact_id": "sql-art-1",
                    "kind": "db_rag_sql_candidate",
                    "producer": "rag_db_qa",
                    "mime": "application/sql",
                    "summary": "Reviewed SQL candidate",
                    "content": "SELECT 1",
                    "created_at": "2026-05-06T00:00:00+00:00",
                },
                "sel-art-1": {
                    "artifact_id": "sel-art-1",
                    "kind": "db_rag_column_selection",
                    "producer": "rag_db_qa",
                    "mime": "application/json",
                    "summary": "Reviewed column selection",
                    "content": {"columns": ["age", "gender", "tb_outcome"]},
                    "created_at": "2026-05-06T00:00:00+00:00",
                },
            },
            "datasets": {
                "subset-1": {
                    "id": "subset-1",
                    "kind": "subset",
                    "row_count": 1894,
                    "columns": ["age", "gender", "tb_outcome"],
                    "provenance": {"source": "db_rag_sql"},
                }
            },
            "conversation_events": [
                {
                    "event_id": "evt-1",
                    "seq": 1,
                    "created_at": "2026-05-06T00:00:00+00:00",
                    "actor": "human_review",
                    "actor_role": "system",
                    "type": "review_decision",
                    "user_turn_hash": "u1",
                    "review_kind": "db_rag_column_selection",
                    "decision": "approve",
                    "text": "Column selection approved",
                },
                {
                    "event_id": "evt-2",
                    "seq": 2,
                    "created_at": "2026-05-06T00:00:01+00:00",
                    "actor": "rag_db_qa",
                    "actor_role": "assistant",
                    "type": "execution_finished",
                    "user_turn_hash": "u1",
                    "text": "SQL executed and dataset saved",
                },
            ],
        },
        "next_action": None,
        "last_action": "rag_db_qa",
        "observations": [],
        "orchestrator": {},
        "planner": {
            "memory": {
                "active_user_goal": "Analyze gender and TB outcome using the saved subset",
                "conversation_intent_summary": "User is working with a DB-RAG subset.",
                "unresolved_user_constraints": [],
            }
        },
        "agents": {},
        "node_data": {},
        "meta": {"workflow_trace": ["rag_db_qa"]},
    }
    return complete_task(
        state,
        kind="db_rag_sql_extraction",
        source_question="Query my database for age, gender, and TB outcome.",
        goal_text="Subset index cases.",
        label="index-case subset",
        summary="Reviewed SQL was executed and saved as a subset dataset.",
        artifact_refs={
            "selection_artifact_id": "sel-art-1",
            "sql_candidate_artifact_id": "sql-art-1",
            "dataset_artifact_id": "subset-1",
        },
        provenance={"producer_node": "rag_db_qa"},
    )


def test_planner_environment_includes_recent_user_messages_tasks_and_datasets() -> None:
    from graph.nodes.orchestrator.context_builder import build_planner_context

    context = build_planner_context(_state(), ["qa", "rag_db_qa", "generate_code"])
    env = context["planner_environment"]

    assert env["latest_user_message"] == "Now find the relationship between gender and TB outcome."
    assert env["recent_user_messages"] == [
        "Query my database for age, gender, and TB outcome.",
        "Now find the relationship between gender and TB outcome.",
    ]
    assert env["active_goal"] == "Analyze gender and TB outcome using the saved subset"
    assert env["candidate_datasets"] == [
        {
            "dataset_id": "subset-1",
            "kind": "subset",
            "columns": ["age", "gender", "tb_outcome"],
            "row_count": 1894,
            "source_task_id": env["candidate_tasks"][0]["task_id"],
        }
    ]
    assert env["candidate_tasks"][0]["kind"] == "db_rag_sql_extraction"
    assert env["candidate_tasks"][0]["dataset_id"] == "subset-1"
    assert env["available_actions"] == ["qa", "rag_db_qa", "generate_code"]
