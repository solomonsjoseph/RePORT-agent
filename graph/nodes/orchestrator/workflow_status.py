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
    effective_last_action = (
        meta.get(MetaKeys.SEMANTIC_LAST_ACTION)
        if state.get("last_action") == "clarification"
        else state.get("last_action")
    )
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
    pending_extraction_opt_in = dict(rag_db_qa.get("pending_extraction_opt_in") or {})
    if pending_extraction_opt_in.get("status") == "awaiting_reply":
        intent_id = pending_extraction_opt_in.get("intent_id")
        return {
            "milestone": "awaiting_rag_db_extraction_opt_in",
            "completion_status": "blocked_waiting",
            "blocker_signature": f"waiting_for_rag_db_extraction_opt_in:{intent_id}",
        }

    pending_column_review_artifact_id = str(rag_db_qa.get("pending_column_review_artifact_id") or "").strip()
    if pending_column_review_artifact_id:
        return {
            "milestone": "awaiting_rag_db_column_review",
            "completion_status": "blocked_waiting",
            "blocker_signature": f"waiting_for_rag_db_column_review:{pending_column_review_artifact_id}",
        }

    pending_sql_candidate_artifact_id = str(rag_db_qa.get("pending_sql_candidate_artifact_id") or "").strip()
    if pending_sql_candidate_artifact_id:
        return {
            "milestone": "awaiting_rag_db_sql_review",
            "completion_status": "blocked_waiting",
            "blocker_signature": f"waiting_for_rag_db_sql_review:{pending_sql_candidate_artifact_id}",
        }

    approved_column_selection_artifact_id = str(
        rag_db_qa.get("approved_column_selection_artifact_id") or ""
    ).strip()
    if (
        approved_column_selection_artifact_id
        and rag_db_qa.get("thread_status") == "awaiting_sql_generation"
    ):
        return {
            "milestone": "ready_for_rag_db_sql_generation",
            "completion_status": "incomplete",
            "blocker_signature": (
                f"ready_for_rag_db_sql_generation:{approved_column_selection_artifact_id}"
            ),
        }

    pending_recoverable_error = dict(rag_db_qa.get("pending_recoverable_error") or {})
    if pending_recoverable_error.get("status") == "awaiting_reply":
        stage = str(pending_recoverable_error.get("stage") or "unknown")
        return {
            "milestone": "awaiting_rag_db_recoverable_error_clarification",
            "completion_status": "blocked_waiting",
            "blocker_signature": f"waiting_for_rag_db_recoverable_error:{stage}",
        }

    if error.get("category") == "db_rag_sql" and effective_last_action in {
        "rag_db_qa",
        "human_review_rag_db_sql_execution",
    }:
        return {
            "milestone": "db_rag_sql_error",
            "completion_status": "blocked_waiting",
            "blocker_signature": f"db_rag_sql_error:{error.get('type')}:{_error_signature(error)}",
        }

    if error.get("category") == "db_rag_review" and effective_last_action in {
        "human_review_rag_db_column_selection",
        "human_review_rag_db_sql_execution",
    }:
        return {
            "milestone": "db_rag_review_error",
            "completion_status": "blocked_waiting",
            "blocker_signature": f"db_rag_review_error:{error.get('type')}:{_error_signature(error)}",
        }

    if (
        rag_db_qa.get("thread_status") == "completed"
        and effective_last_action == "human_review_rag_db_sql_execution"
        and not _has_unanswered_human_message(state)
    ):
        return {
            "milestone": "db_rag_sql_completed",
            "completion_status": "complete",
            "blocker_signature": None,
        }

    if effective_last_action == "terminal_execution_error" and terminal_error_category in TERMINAL_EXECUTION_ERROR_CATEGORIES:
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
        and effective_last_action in {"qa", "rag_db_qa"}
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
