from __future__ import annotations

from copy import deepcopy
import math

import pytest


def _event_record(seq: int, *, event_type: str = "assistant", **extra) -> dict:
    base = {
        "event_id": f"e{seq}",
        "seq": seq,
        "created_at": f"2026-04-29T00:00:0{seq}Z",
        "type": event_type,
        "actor": "qa",
        "actor_role": "assistant",
        "user_turn_hash": "u1",
        "text": f"message {seq}",
    }
    base.update(extra)
    return base


def test_ensure_conversation_state_initializes_empty_state_containers() -> None:
    from graph.conversation_events import ensure_conversation_state

    updated = ensure_conversation_state({})

    assert updated["artifacts"]["conversation_events"] == []
    assert updated["artifacts"]["conversation_events_version"] == 1
    assert updated["artifacts"]["artifact_manifest_version"] == 1
    assert updated["artifacts"]["files"] == {}
    assert updated["meta"]["next_event_seq"] == 1


def test_ensure_conversation_state_recovers_next_seq_from_history_when_meta_missing() -> None:
    from graph.conversation_events import ensure_conversation_state

    state = {
        "artifacts": {
            "conversation_events": [
                _event_record(1, event_type="user"),
                _event_record(2),
            ],
            "conversation_events_version": 1,
            "artifact_manifest_version": 1,
            "files": {},
        },
    }

    updated = ensure_conversation_state(state)

    assert updated["meta"]["next_event_seq"] == 3


def test_ensure_conversation_state_preserves_existing_legacy_artifacts() -> None:
    from graph.conversation_events import ensure_conversation_state

    state = {
        "artifacts": {
            "generated_code": "print('ok')\n",
            "execution_output": "done",
            "conversation_events": [],
            "files": {},
        }
    }

    updated = ensure_conversation_state(state)

    assert updated["artifacts"]["generated_code"] == "print('ok')\n"
    assert updated["artifacts"]["execution_output"] == "done"


def test_ensure_conversation_state_rejects_non_json_existing_history() -> None:
    from graph.conversation_events import ensure_conversation_state

    with pytest.raises(TypeError, match="JSON-serializable"):
        ensure_conversation_state(
            {
                "artifacts": {
                    "conversation_events": [{"type": "assistant", "payload": object()}],
                    "files": {},
                }
            }
        )


def test_ensure_conversation_state_bumps_stale_next_seq_forward() -> None:
    from graph.conversation_events import ensure_conversation_state

    state = {
        "artifacts": {
            "conversation_events": [
                _event_record(1, event_type="user"),
                _event_record(2),
            ],
            "conversation_events_version": 1,
            "artifact_manifest_version": 1,
            "files": {},
        },
        "meta": {"next_event_seq": 2},
    }

    updated = ensure_conversation_state(state)

    assert updated["meta"]["next_event_seq"] == 3


def test_append_conversation_event_rejects_non_json_payloads() -> None:
    from graph.conversation_events import append_conversation_event

    state = {
        "artifacts": {
            "conversation_events": [],
            "conversation_events_version": 1,
            "artifact_manifest_version": 1,
            "files": {},
        },
        "meta": {},
    }

    with pytest.raises(TypeError, match="JSON-serializable"):
        append_conversation_event(
            state,
            {
                "type": "tool_call",
                "actor": "qa",
                "actor_role": "assistant",
                "user_turn_hash": "u1",
                "tool": "weather",
                "args": {"cities": {"boston", "seattle"}},
            },
        )


def test_append_conversation_event_requires_user_turn_hash() -> None:
    from graph.conversation_events import append_conversation_event

    state = {
        "artifacts": {
            "conversation_events": [],
            "conversation_events_version": 1,
            "artifact_manifest_version": 1,
            "files": {},
        },
        "meta": {},
    }

    with pytest.raises(ValueError, match="user_turn_hash"):
        append_conversation_event(
            state,
            {
                "type": "assistant",
                "actor": "qa",
                "actor_role": "assistant",
                "text": "hello",
            },
        )


def test_append_conversation_event_rejects_unknown_type() -> None:
    from graph.conversation_events import append_conversation_event

    state = {
        "artifacts": {
            "conversation_events": [],
            "conversation_events_version": 1,
            "artifact_manifest_version": 1,
            "files": {},
        },
        "meta": {},
    }

    with pytest.raises(ValueError, match="unsupported type"):
        append_conversation_event(
            state,
            {
                "type": "bogus",
                "actor": "qa",
                "actor_role": "assistant",
                "user_turn_hash": "u1",
            },
        )


def test_append_conversation_event_requires_type_specific_fields() -> None:
    from graph.conversation_events import append_conversation_event

    state = {
        "artifacts": {
            "conversation_events": [],
            "conversation_events_version": 1,
            "artifact_manifest_version": 1,
            "files": {},
        },
        "meta": {},
    }

    with pytest.raises(ValueError, match="text"):
        append_conversation_event(
            state,
            {
                "type": "assistant",
                "actor": "qa",
                "actor_role": "assistant",
                "user_turn_hash": "u1",
            },
        )


def test_append_conversation_event_rejects_wrong_json_field_type() -> None:
    from graph.conversation_events import append_conversation_event

    state = {
        "artifacts": {
            "conversation_events": [],
            "conversation_events_version": 1,
            "artifact_manifest_version": 1,
            "files": {},
        },
        "meta": {},
    }

    with pytest.raises(TypeError, match="event.text must be a string"):
        append_conversation_event(
            state,
            {
                "type": "assistant",
                "actor": "qa",
                "actor_role": "assistant",
                "user_turn_hash": "u1",
                "text": {"not": "a string"},
            },
        )


def test_append_conversation_event_rejects_non_finite_floats() -> None:
    from graph.conversation_events import append_conversation_event

    state = {
        "artifacts": {
            "conversation_events": [],
            "conversation_events_version": 1,
            "artifact_manifest_version": 1,
            "files": {},
        },
        "meta": {},
    }

    with pytest.raises(TypeError, match="non-finite float"):
        append_conversation_event(
            state,
            {
                "type": "tool_call",
                "actor": "qa",
                "actor_role": "assistant",
                "user_turn_hash": "u1",
                "tool": "calculator",
                "args": {"value": math.nan},
            },
        )


def test_store_thread_artifact_rejects_non_json_payloads() -> None:
    from graph.conversation_events import store_thread_artifact

    state = {
        "artifacts": {
            "conversation_events": [],
            "conversation_events_version": 1,
            "artifact_manifest_version": 1,
            "files": {},
        },
        "meta": {},
    }

    with pytest.raises(TypeError, match="JSON-serializable"):
        store_thread_artifact(
            state,
            {
                "kind": "text",
                "producer": "qa",
                "mime": "text/plain",
                "summary": "bad payload",
                "content": object(),
            },
        )


def test_store_thread_artifact_requires_all_fields() -> None:
    from graph.conversation_events import store_thread_artifact

    state = {
        "artifacts": {
            "conversation_events": [],
            "conversation_events_version": 1,
            "artifact_manifest_version": 1,
            "files": {},
        },
        "meta": {},
    }

    with pytest.raises(ValueError, match="content"):
        store_thread_artifact(
            state,
            {
                "kind": "text",
                "producer": "qa",
                "mime": "text/plain",
                "summary": "missing content",
            },
        )


def test_store_thread_artifact_rejects_wrong_field_type() -> None:
    from graph.conversation_events import store_thread_artifact

    state = {
        "artifacts": {
            "conversation_events": [],
            "conversation_events_version": 1,
            "artifact_manifest_version": 1,
            "files": {},
        },
        "meta": {},
    }

    with pytest.raises(TypeError, match="artifact.kind must be a string"):
        store_thread_artifact(
            state,
            {
                "kind": ["not", "a", "string"],
                "producer": "qa",
                "mime": "text/plain",
                "summary": "bad kind",
                "content": "hello",
            },
        )


def test_append_conversation_event_keeps_caller_payload_unchanged() -> None:
    from graph.conversation_events import append_conversation_event

    event = {
        "type": "assistant",
        "actor": "qa",
        "actor_role": "assistant",
        "user_turn_hash": "u1",
        "text": "hello",
        "details": {"tags": ["a", "b"]},
    }
    original = deepcopy(event)

    append_conversation_event(
        {
            "artifacts": {
                "conversation_events": [],
                "conversation_events_version": 1,
                "artifact_manifest_version": 1,
                "files": {},
            },
            "meta": {},
        },
        event,
    )

    assert event == original


def test_store_thread_artifact_keeps_caller_payload_unchanged() -> None:
    from graph.conversation_events import store_thread_artifact

    artifact = {
        "kind": "text",
        "producer": "qa",
        "mime": "text/plain",
        "summary": "note",
        "content": {"lines": ["a", "b"]},
    }
    original = deepcopy(artifact)

    store_thread_artifact(
        {
            "artifacts": {
                "conversation_events": [],
                "conversation_events_version": 1,
                "artifact_manifest_version": 1,
                "files": {},
            },
            "meta": {},
        },
        artifact,
    )

    assert artifact == original


def test_append_conversation_event_assigns_seq_and_event_id() -> None:
    from graph.conversation_events import append_conversation_event

    state = {
        "artifacts": {
            "conversation_events": [],
            "conversation_events_version": 1,
            "artifact_manifest_version": 1,
            "files": {},
        },
        "meta": {"next_event_seq": 1},
    }

    updated = append_conversation_event(
        state,
        {
            "type": "assistant",
            "actor": "qa",
            "actor_role": "assistant",
            "user_turn_hash": "u1",
            "text": "Boston today: cool and rainy.",
        },
    )

    event = updated["artifacts"]["conversation_events"][0]
    assert event["seq"] == 1
    assert event["type"] == "assistant"
    assert event["actor"] == "qa"
    assert event["user_turn_hash"] == "u1"
    assert event["event_id"]
    assert updated["meta"]["next_event_seq"] == 2


def test_store_thread_artifact_creates_manifest_entry_without_mutating_existing_payloads() -> None:
    from graph.conversation_events import store_thread_artifact

    state = {
        "artifacts": {
            "conversation_events": [],
            "conversation_events_version": 1,
            "artifact_manifest_version": 1,
            "files": {},
        },
        "meta": {"next_event_seq": 1},
    }

    updated = store_thread_artifact(
        state,
        {
            "kind": "code",
            "producer": "generate_code",
            "mime": "text/x-python",
            "summary": "Generated analysis script",
            "content": "print('ok')\n",
        },
    )

    artifact_id = next(iter(updated["artifacts"]["files"]))
    payload = updated["artifacts"]["files"][artifact_id]
    assert payload["kind"] == "code"
    assert payload["producer"] == "generate_code"
    assert payload["summary"] == "Generated analysis script"
    assert payload["content"] == "print('ok')\n"


def test_state_view_accessors_return_deep_copies() -> None:
    from graph.state_views import get_artifact_files, get_conversation_events

    state = {
        "artifacts": {
            "conversation_events": [
                {
                    "event_id": "e1",
                    "seq": 1,
                    "created_at": "2026-04-29T00:00:00Z",
                    "type": "assistant",
                    "actor": "qa",
                    "actor_role": "assistant",
                    "user_turn_hash": "u1",
                    "text": "hello",
                }
            ],
            "files": {
                "a1": {
                    "artifact_id": "a1",
                    "kind": "text",
                    "producer": "qa",
                    "mime": "text/plain",
                    "summary": "note",
                    "created_at": "2026-04-29T00:00:00Z",
                    "content": {"value": "x"},
                }
            },
        }
    }

    events = get_conversation_events(state)
    files = get_artifact_files(state)
    events[0]["text"] = "mutated"
    files["a1"]["content"]["value"] = "mutated"

    assert state["artifacts"]["conversation_events"][0]["text"] == "hello"
    assert state["artifacts"]["files"]["a1"]["content"]["value"] == "x"


def test_state_view_accessors_validate_conversation_state() -> None:
    from graph.state_views import get_artifacts, get_conversation_events

    with pytest.raises(TypeError, match="JSON-serializable"):
        get_conversation_events(
            {
                "artifacts": {
                    "conversation_events": [{"payload": object()}],
                    "files": {},
                }
            }
        )

    with pytest.raises(TypeError, match="JSON-serializable"):
        get_artifacts(
            {
                "artifacts": {
                    "conversation_events": [{"payload": object()}],
                    "files": {},
                }
            }
        )


def test_ensure_conversation_state_rejects_non_string_artifact_map_keys() -> None:
    from graph.conversation_events import ensure_conversation_state

    with pytest.raises(TypeError, match="string keys"):
        ensure_conversation_state(
            {
                "artifacts": {
                    "conversation_events": [],
                    "files": {
                        1: {
                            "artifact_id": "a1",
                            "kind": "text",
                            "producer": "qa",
                            "mime": "text/plain",
                            "summary": "note",
                            "created_at": "2026-04-29T00:00:00Z",
                            "content": "hello",
                        }
                    },
                }
            }
        )


def test_merge_state_patch_validates_conversation_artifact_updates() -> None:
    from graph.state_views import merge_state_patch

    with pytest.raises(TypeError, match="JSON-serializable"):
        merge_state_patch(
            {
                "messages": [],
                "output": {},
                "artifacts": {},
                "next_action": None,
                "last_action": None,
                "observations": [],
                "orchestrator": {},
                "planner": {},
                "agents": {},
                "node_data": {},
                "meta": {},
            },
            {
                "artifacts": {
                    "conversation_events": [{"payload": object()}],
                }
            },
        )
