from __future__ import annotations

from hashlib import sha256

from ...state import MetaKeys
from ...state_views import get_artifacts, get_node_data
from ...workflow_config import MAX_ERROR_ITERATIONS
from .state_logic import _has_unanswered_human_message

TERMINAL_EXECUTION_ERROR_CATEGORIES = frozenset(
    {
        "policy_blocked",
        "unsupported_runtime",
        "infrastructure",
        "timeout",
    }
)


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
    final_approved_hash = meta.get(MetaKeys.FINAL_APPROVED_CODE_HASH)
    has_ticket = bool(current_hash and ticket_hash and current_hash == ticket_hash)
    is_current_code_final_approved = bool(
        current_hash and final_approved_hash and current_hash == final_approved_hash
    )
    terminal_error_category = error.get("category") if executor.get("run_status") == "error" else None
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

    rag_db_qa = get_node_data(state, "rag_db_qa")
    pending_column_review = dict(rag_db_qa.get("pending_column_review") or {})
    if pending_column_review.get("status") == "awaiting_review":
        selection_id = pending_column_review.get("selection_id")
        return {
            "milestone": "awaiting_rag_db_column_review",
            "completion_status": "blocked_waiting",
            "blocker_signature": f"waiting_for_rag_db_column_review:{selection_id}",
        }

    pending_sql_candidate = dict(rag_db_qa.get("pending_sql_candidate") or {})
    if pending_sql_candidate.get("status") == "prepared":
        selection_id = pending_sql_candidate.get("selection_id")
        return {
            "milestone": "awaiting_rag_db_sql_review",
            "completion_status": "blocked_waiting",
            "blocker_signature": f"waiting_for_rag_db_sql_review:{selection_id}",
        }

    if error.get("category") == "db_rag_sql" and state.get("last_action") in {
        "rag_db_qa",
        "human_review_rag_db_sql_execution",
    }:
        return {
            "milestone": "db_rag_sql_error",
            "completion_status": "complete",
            "blocker_signature": f"db_rag_sql_error:{error.get('type')}:{_error_signature(error)}",
        }

    if state.get("last_action") == "terminal_execution_error" and terminal_error_category in TERMINAL_EXECUTION_ERROR_CATEGORIES:
        return {
            "milestone": "terminal_error",
            "completion_status": "complete",
            "blocker_signature": f"terminal_error:{terminal_error_category}",
        }

    if executor.get("run_status") == "error" and error.get("category") == "retryable_code":
        if int(meta.get(MetaKeys.ERROR_ITERATIONS, 0)) >= MAX_ERROR_ITERATIONS:
            if review.get("after_error_decision") is not None:
                return {
                    "milestone": "retrying_after_error",
                    "completion_status": "incomplete",
                    "blocker_signature": f"retryable_error:{error.get('type')}:{_error_signature(error)}",
                }
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

    if meta.get(MetaKeys.ERROR_RECOVERY_ACTIVE) and has_code and executor.get("run_status") in ("idle", "pending"):
        return {
            "milestone": "retrying_after_error",
            "completion_status": "incomplete",
            "blocker_signature": f"retryable_error:{error.get('type')}:{_error_signature(error)}",
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

    if executor.get("run_status") == "ok" and not is_current_code_final_approved:
        return {
            "milestone": "awaiting_final_review",
            "completion_status": "blocked_waiting",
            "blocker_signature": "waiting_for_final_review",
        }

    if (
        executor.get("run_status") == "ok"
        and is_current_code_final_approved
        and not _has_unanswered_human_message(state)
    ):
        return {
            "milestone": "analysis_complete",
            "completion_status": "complete",
            "blocker_signature": None,
        }

    if (
        output.get("qa_response")
        and not has_code
        and state.get("last_action") in {"qa", "rag_db_qa"}
        and not _has_unanswered_human_message(state)
    ):
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
