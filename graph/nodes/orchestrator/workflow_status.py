from __future__ import annotations

from hashlib import sha256

from ...state import MetaKeys
from ...state_views import get_artifacts, get_node_data


def _error_signature(error: dict) -> str:
    payload = f"{error.get('category')}|{error.get('type')}|{error.get('message')}"
    return sha256(payload.encode()).hexdigest()[:12]


def derive_workflow_status(state: dict) -> dict:
    artifacts = get_artifacts(state)
    meta = dict(state.get("meta") or {})
    executor = get_node_data(state, "executor")
    review = get_node_data(state, "human_review")
    error = dict(artifacts.get("error") or {})
    output = dict(state.get("output") or {})
    has_code = bool(artifacts.get("generated_code"))
    current_hash = meta.get(MetaKeys.CURRENT_CODE_HASH)
    ticket_hash = meta.get(MetaKeys.EXECUTION_TICKET_HASH)
    has_ticket = bool(current_hash and ticket_hash and current_hash == ticket_hash)

    if meta.get(MetaKeys.AWAITING_USER_CLARIFICATION):
        kind = meta.get(MetaKeys.CLARIFICATION_KIND, "unknown")
        return {
            "milestone": "awaiting_clarification",
            "completion_status": "blocked_waiting",
            "blocker_signature": f"waiting_for_user_clarification:{kind}",
        }

    if meta.get(MetaKeys.TOOL_REQUEST_QUEUE):
        return {
            "milestone": "waiting_for_tool_results",
            "completion_status": "blocked_waiting",
            "blocker_signature": "waiting_for_tool_results",
        }

    if executor.get("run_status") == "error" and error.get("category") == "retryable_code":
        max_iters = 5
        if int(meta.get(MetaKeys.ERROR_ITERATIONS, 0)) >= max_iters:
            return {
                "milestone": "awaiting_after_error_review",
                "completion_status": "blocked_waiting",
                "blocker_signature": f"waiting_for_after_error_review:{error.get('category')}:{error.get('type')}",
            }
        return {
            "milestone": "retrying_after_error",
            "completion_status": "incomplete",
            "blocker_signature": f"retryable_error:{error.get('type')}:{_error_signature(error)}",
        }

    if executor.get("run_status") == "error" and error.get("category") in {
        "policy_blocked",
        "unsupported_runtime",
        "infrastructure",
        "timeout",
    }:
        return {
            "milestone": "terminal_error",
            "completion_status": "complete",
            "blocker_signature": f"terminal_error:{error.get('category')}",
        }

    if has_code and not has_ticket and executor.get("run_status") != "ok":
        return {
            "milestone": "awaiting_run_review",
            "completion_status": "blocked_waiting",
            "blocker_signature": "waiting_for_before_run_review",
        }

    if has_code and has_ticket and executor.get("run_status") in ("idle", "pending"):
        return {
            "milestone": "ready_to_execute",
            "completion_status": "incomplete",
            "blocker_signature": "ready_for_execution",
        }

    if executor.get("run_status") == "ok" and review.get("final_decision") is None:
        return {
            "milestone": "awaiting_final_review",
            "completion_status": "blocked_waiting",
            "blocker_signature": "waiting_for_final_review",
        }

    if state.get("last_action") == "qa" and output.get("qa_response"):
        return {
            "milestone": "answered",
            "completion_status": "complete",
            "blocker_signature": None,
        }

    return {
        "milestone": "needs_code",
        "completion_status": "incomplete",
        "blocker_signature": "missing_next_step",
    }
