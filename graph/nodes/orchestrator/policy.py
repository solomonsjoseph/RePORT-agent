from __future__ import annotations

from functools import lru_cache

from ...state import AgentState, MetaKeys
from ..state_helpers import get_agent_state
from .policy_contract import DEFAULT_POLICY_PATH, OrchestratorPolicyContract, load_policy_contract


def _latest_human_message(state: AgentState) -> str:
    for message in reversed(list(state.get("messages", []))):
        if getattr(message, "type", None) == "human":
            return str(getattr(message, "content", "") or "").strip().lower()
    return ""


def _is_explicit_code_request(state: AgentState) -> bool:
    latest = " ".join(_latest_human_message(state).split())
    if not latest:
        return False
    direct_markers = (
        "write code",
        "generate code",
        "show code",
        "sample code",
        "example code",
        "python code",
        "ready-to-run code",
        "ready to run code",
        "give me code",
        "give me python",
    )
    if any(marker in latest for marker in direct_markers):
        return True
    return "python" in latest and any(
        token in latest for token in ("write", "generate", "show", "ready-to-run", "ready to run")
    )


def _prefer_rag_db_qa(state: AgentState) -> bool:
    rag_state = get_agent_state(state, "rag_db_qa")
    if rag_state.get("pending_sql_offer") or rag_state.get("active_thread"):
        return True

    latest = _latest_human_message(state)
    if not latest or _is_explicit_code_request(state):
        return False

    quantitative_cues = (
        "how many",
        "count",
        "number of",
        "proportion",
        "percentage",
        "compare",
        "subset",
        "extract",
        "filter",
        "cohort",
    )
    database_cues = (
        "participant",
        "participants",
        "patient",
        "patients",
        "subject",
        "subjects",
        "records",
        "rows",
        "database",
        "table",
        "form",
        "column",
        "columns",
        "site",
        "culture",
    )
    referential_followup_cues = ("what about", "same", "those", "them", "that subset", "break down by")

    has_quantitative_cue = any(cue in latest for cue in quantitative_cues)
    has_database_cue = any(cue in latest for cue in database_cues)
    has_referential_cue = any(cue in latest for cue in referential_followup_cues)
    return (has_quantitative_cue and has_database_cue) or has_referential_cue


def _always(_state: AgentState) -> bool:
    return True


PREDICATE_REGISTRY = {
    "explicit_code_request": _is_explicit_code_request,
    "prefer_rag_db_qa": _prefer_rag_db_qa,
    "always": _always,
}


@lru_cache(maxsize=1)
def get_policy_contract() -> OrchestratorPolicyContract:
    return load_policy_contract(
        DEFAULT_POLICY_PATH,
        known_actions={
            "tool_handler",
            "error_handler",
            "terminal_execution_error",
            "human_review_after_error",
            "human_review_before_run",
            "execute_code",
            "human_review_before_output",
            "clarification",
            "qa",
            "rag_db_qa",
            "generate_code",
            "end",
        },
        known_predicates=set(PREDICATE_REGISTRY),
    )


DETERMINISTIC_CONTROL_ACTIONS = get_policy_contract().deterministic_control_actions


def select_planner_fallback_action(state: AgentState, available_actions: set[str]) -> str:
    policy = get_policy_contract()
    for rule in policy.planner_fallback_rules:
        predicate = PREDICATE_REGISTRY[rule.predicate]
        if predicate(state) and rule.action in available_actions:
            return rule.action
    if "end" in available_actions:
        return "end"
    return next(iter(available_actions), "end")


def should_prefer_rag_db_qa(state: AgentState) -> bool:
    return _prefer_rag_db_qa(state)


def _tool_request_queue(state: AgentState) -> list[str]:
    return list((state.get("meta") or {}).get(MetaKeys.TOOL_REQUEST_QUEUE, []))


def _next_tool_requester(state: AgentState) -> str | None:
    for requester in _tool_request_queue(state):
        agent_state = get_agent_state(state, requester)
        if agent_state.get("tool_results"):
            return requester
    return None
