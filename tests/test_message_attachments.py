from __future__ import annotations

from types import SimpleNamespace


def test_build_user_message_attachments_includes_dataset_and_schema_refs() -> None:
    from utils.message_attachments import build_user_message_attachments

    artifact = {
        "id": "uploaded-abcd1234",
        "kind": "uploaded",
        "schema_path": "/tmp/uploaded-abcd1234.schema.json",
    }

    attachments = build_user_message_attachments(artifact)

    assert attachments == [
        {"artifact_id": "uploaded-abcd1234", "kind": "dataset", "role": "primary"},
        {"artifact_id": "uploaded-abcd1234.schema", "kind": "schema", "role": "supporting"},
    ]


def test_latest_message_attachments_reads_latest_human_message_metadata() -> None:
    from utils.message_attachments import latest_message_attachments

    state = {
        "messages": [
            SimpleNamespace(type="human", content="old", additional_kwargs={}),
            SimpleNamespace(type="ai", content="old answer", additional_kwargs={}),
            SimpleNamespace(
                type="human",
                content="analyze this dataset",
                additional_kwargs={
                    "attachments": [
                        {"artifact_id": "uploaded-abcd1234", "kind": "dataset", "role": "primary"}
                    ]
                },
            ),
        ]
    }

    assert latest_message_attachments(state) == [
        {"artifact_id": "uploaded-abcd1234", "kind": "dataset", "role": "primary"}
    ]


def test_latest_attachment_summary_reports_kinds_and_payload() -> None:
    from utils.message_attachments import latest_attachment_summary

    state = {
        "messages": [
            SimpleNamespace(
                type="human",
                content="analyze this dataset",
                additional_kwargs={
                    "attachments": [
                        {"artifact_id": "uploaded-abcd1234", "kind": "dataset", "role": "primary"},
                        {"artifact_id": "uploaded-abcd1234.schema", "kind": "schema", "role": "supporting"},
                    ]
                },
            ),
        ]
    }

    assert latest_attachment_summary(state) == {
        "has_attachments": True,
        "attachment_kinds": ["dataset", "schema"],
        "attachments": [
            {"artifact_id": "uploaded-abcd1234", "kind": "dataset", "role": "primary"},
            {"artifact_id": "uploaded-abcd1234.schema", "kind": "schema", "role": "supporting"},
        ],
    }


def test_build_planner_context_includes_latest_turn_attachment_summary() -> None:
    from graph.nodes.orchestrator.context_builder import build_planner_context

    state = {
        "messages": [
            SimpleNamespace(
                type="human",
                content="analyze this dataset",
                additional_kwargs={
                    "attachments": [
                        {"artifact_id": "uploaded-abcd1234", "kind": "dataset", "role": "primary"}
                    ]
                },
            )
        ],
        "artifacts": {
            "conversation_events": [],
            "conversation_events_version": 1,
            "artifact_manifest_version": 1,
            "files": {},
        },
        "planner": {"memory": {}},
        "orchestrator": {},
        "meta": {},
        "observations": [],
        "agents": {"executor": {"run_status": "idle"}, "human_review": {}},
        "node_data": {},
    }

    context = build_planner_context(state, ["qa", "generate_code"])

    assert context["latest_turn_attachments"] == {
        "has_attachments": True,
        "attachment_kinds": ["dataset"],
        "attachments": [
            {"artifact_id": "uploaded-abcd1234", "kind": "dataset", "role": "primary"}
        ],
    }
