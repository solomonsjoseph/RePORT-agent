from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ..conversation_events import (
    append_conversation_event,
    build_error_event,
    build_execution_finished_event,
    build_execution_started_event,
    build_figure_event,
    store_thread_artifact,
)
from ..state import MetaKeys
from tools.execution import run_python_user
from .orchestrator.state_logic import _user_message_hash
from .state_helpers import update_agent_state

NODE_NAME = "execute_code"
NODE_CAPABILITY = (
    "Execute previously generated Python code after approval and collect outputs or errors."
)


def _current_user_turn_hash(state: dict) -> str | None:
    meta = dict(state.get("meta") or {})
    return str(meta.get(MetaKeys.LAST_USER_MESSAGE_HASH) or _user_message_hash(state) or "") or None


def _store_text_artifact(state: dict, *, text: str, summary: str) -> tuple[dict, str]:
    updated_state = store_thread_artifact(
        state,
        {
            "kind": "text",
            "producer": "executor",
            "mime": "text/plain",
            "summary": summary,
            "content": text,
        },
    )
    artifact_id = next(reversed(dict(updated_state.get("artifacts") or {}).get("files") or {}))
    return updated_state, artifact_id


def _store_figure_artifact(state: dict, *, figure_png: bytes, summary: str) -> tuple[dict, str]:
    fd, path = tempfile.mkstemp(prefix="report-agent-", suffix=".png")
    os.close(fd)
    Path(path).write_bytes(figure_png)
    updated_state = store_thread_artifact(
        state,
        {
            "kind": "figure",
            "producer": "executor",
            "mime": "image/png",
            "summary": summary,
            "content": {"path": Path(path)},
        },
    )
    artifact_id = next(reversed(dict(updated_state.get("artifacts") or {}).get("files") or {}))
    return updated_state, artifact_id


def execute_code_node(state, df):
    user_turn_hash = _current_user_turn_hash(state)
    code = (state.get("output") or {}).get("generated_code")
    if not code:
        output = dict(state.get("output") or {})
        output.pop("text", None)
        output.pop("figure_png", None)
        output.pop("figure_artifact_id", None)
        output.pop("text_artifact_id", None)
        output["error"] = {
            "type": "NoCode",
            "message": "No code available to execute.",
        }
        updated_state = append_conversation_event(
            state,
            build_execution_finished_event(
                actor="executor",
                user_turn_hash=user_turn_hash,
                text="No code available to execute.",
                status="error",
            ),
        )
        updated_state = append_conversation_event(
            updated_state,
            build_error_event(
                actor="executor",
                user_turn_hash=user_turn_hash,
                text="No code available to execute.",
                error=output["error"],
                status="error",
            ),
        )
        updated_state = {
            **updated_state,
            "output": output,
        }
        return update_agent_state(
            updated_state,
            "executor",
            {
                "status": "error",
                "run_status": "error",
                "error": "No code available to execute.",
            },
        )

    resolved_df = df(state) if callable(df) else df
    updated_state = append_conversation_event(
        state,
        build_execution_started_event(
            actor="executor",
            user_turn_hash=user_turn_hash,
            text="Executing approved Python code.",
            status="started",
        ),
    )
    meta = dict(state.get("meta") or {})
    result, stdout, figure_png, error = run_python_user(
        code,
        resolved_df,
        dataset_id=str(meta.get(MetaKeys.ANALYSIS_DATASET_ID) or "").strip() or None,
    )

    if error:
        output = dict(state.get("output") or {})
        output.pop("text", None)
        output.pop("figure_png", None)
        output.pop("figure_artifact_id", None)
        output.pop("text_artifact_id", None)
        output["error"] = error
        meta = dict(state.get("meta", {}) or {})
        if error.get("category") == "retryable_code":
            meta[MetaKeys.ERROR_RECOVERY_ACTIVE] = True
        else:
            meta.pop(MetaKeys.ERROR_RECOVERY_ACTIVE, None)
            meta.pop(MetaKeys.EXECUTION_TICKET_HASH, None)
        updated_state = {
            **updated_state,
            "output": output,
            "meta": meta,
        }
        failure_text = str(error.get("message", "Execution failed."))
        updated_state = append_conversation_event(
            updated_state,
            build_execution_finished_event(
                actor="executor",
                user_turn_hash=user_turn_hash,
                text=failure_text,
                status="error",
            ),
        )
        updated_state = append_conversation_event(
            updated_state,
            build_error_event(
                actor="executor",
                user_turn_hash=user_turn_hash,
                text=failure_text,
                error=error,
                status="error",
            ),
        )
        return update_agent_state(
            updated_state,
            "executor",
            {
                "status": "error",
                "run_status": "error",
                "error": error,
            },
        )

    output_text = stdout if stdout else (str(result) if result is not None else "")
    output_payload = dict(state.get("output") or {})
    output_payload.pop("error", None)
    output_payload.pop("figure_png", None)
    output_payload.pop("figure_artifact_id", None)
    output_payload.pop("text_artifact_id", None)
    output_payload["text"] = output_text
    meta = dict(state.get("meta", {}))
    meta[MetaKeys.ERROR_ITERATIONS] = 0
    meta.pop(MetaKeys.ERROR_RECOVERY_ACTIVE, None)
    updated_state = {
        **updated_state,
        "output": output_payload,
        "meta": meta,
    }

    text_artifact_id = None
    if output_text:
        updated_state, text_artifact_id = _store_text_artifact(
            updated_state,
            text=output_text,
            summary="Execution output from approved Python code.",
        )
        output_payload["text_artifact_id"] = text_artifact_id

    updated_state = append_conversation_event(
        updated_state,
        build_execution_finished_event(
            actor="executor",
            user_turn_hash=user_turn_hash,
            text=output_text,
            artifact_id=text_artifact_id,
            status="done",
        ),
    )

    if figure_png:
        updated_state, figure_artifact_id = _store_figure_artifact(
            updated_state,
            figure_png=figure_png,
            summary="Figure generated by approved Python code.",
        )
        output_payload["figure_artifact_id"] = figure_artifact_id
        updated_state["output"] = output_payload
        updated_state = append_conversation_event(
            updated_state,
            build_figure_event(
                actor="executor",
                user_turn_hash=user_turn_hash,
                artifact_id=figure_artifact_id,
                text="Figure generated by approved Python code.",
                status="done",
            ),
        )

    return update_agent_state(
        updated_state,
        "executor",
        {
            "status": "done",
            "run_status": "ok",
            "output": output_text,
        },
    )
