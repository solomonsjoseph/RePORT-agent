from __future__ import annotations

from functools import lru_cache

from db_rag.service.classifier import classify_database_source_intent

from ...state import AgentState, MetaKeys
from ..state_helpers import get_agent_state
from .policy_contract import DEFAULT_POLICY_PATH, OrchestratorPolicyContract, load_policy_contract
from utils.message_attachments import latest_attachment_summary, latest_message_attachments


def _latest_human_message(state: AgentState) -> str:
    for message in reversed(list(state.get("messages", []))):
        if getattr(message, "type", None) == "human":
            return str(getattr(message, "content", "") or "").strip().lower()
    return ""


def _latest_message_attachments(state: AgentState) -> list[dict[str, str]]:
    return latest_message_attachments(state)


def _latest_attachment_summary(state: AgentState) -> dict[str, object]:
    return latest_attachment_summary(state)


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


def _has_uploaded_dataset(state: AgentState) -> bool:
    datasets = dict((state.get("artifacts") or {}).get("datasets") or {})
    return any(dataset.get("kind") == "uploaded" for dataset in datasets.values())


def _has_local_dataframe_artifact(state: AgentState) -> bool:
    datasets = dict((state.get("artifacts") or {}).get("datasets") or {})
    return bool(datasets)


def _dataset_analysis_text(state: AgentState) -> str:
    latest = " ".join(_latest_human_message(state).split())
    meta = dict(state.get("meta") or {})
    if (
        meta.get(MetaKeys.AWAITING_USER_CLARIFICATION)
        and meta.get(MetaKeys.CLARIFICATION_RETURN_NODE) == "qa"
    ):
        pending_question = " ".join(str(meta.get(MetaKeys.PENDING_QUESTION) or "").strip().lower().split())
        if pending_question:
            return f"{pending_question} {latest}".strip()
    return latest


def _is_dataset_analysis_request(state: AgentState) -> bool:
    latest = _dataset_analysis_text(state)
    if not latest:
        return False

    attachments = _latest_message_attachments(state)
    has_dataset_attachment = any(item.get("kind") == "dataset" for item in attachments)
    if not has_dataset_attachment and not _has_local_dataframe_artifact(state):
        return False

    action_markers = (
        "perform",
        "analy",
        "fit",
        "plot",
        "stratif",
        "regress",
        "summar",
        "clean",
        "transform",
        "compare",
        "estimate",
        "calculate",
        "compute",
        "run",
        "model",
        "relationship",
        "association",
        "associat",
        "correlat",
        "find out",
    )
    analysis_domain_markers = (
        "survival analysis",
        "kaplan-meier",
        "kaplan meier",
        "cox model",
        "cox regression",
        "treatment adherence",
        "adherence",
        "dataset",
        "data",
        "csv",
        "cohort",
        "schema",
        "outcome",
        "tb outcome",
        "gender",
        "sex",
        "bacteriologic",
        "bacterialogic",
        "relapse",
        "cure",
    )

    return any(marker in latest for marker in action_markers) and any(
        marker in latest for marker in analysis_domain_markers
    )


def _mentions_explicit_rag_database(text: str) -> bool:
    explicit_rag_database_cues = (
        "rag database",
        "rag databse",
        "rag db",
        "db rag",
        "db-rag",
        "rag datab",
    )
    return any(cue in text for cue in explicit_rag_database_cues)


def _mentions_database_reference(text: str) -> bool:
    database_reference_cues = (
        "database",
        "db-rag",
    )
    return any(cue in text for cue in database_reference_cues)


def is_database_source_reply(text: str, *, allow_short: bool = False) -> bool:
    normalized = " ".join(str(text or "").strip().lower().split())
    if not normalized:
        return False
    explicit_cues = (
        "query my database",
        "query the database",
        "query for my database",
        "query report database",
        "query the report database",
        "query database",
        "look at my database",
        "look at the database",
        "look into my database",
        "look into the database",
        "use my database",
        "use the database",
        "check my database",
        "check the database",
        "inspect my database",
        "inspect the database",
        "use report database",
        "use the report database",
        "search my database",
        "search the database",
        "search report database",
        "search the report database",
        "ask the database",
        "ask report database",
        "ask the report database",
        "use db-rag",
        "use db rag",
        "search db-rag",
        "search db rag",
    )
    if any(cue in normalized for cue in explicit_cues):
        return True
    return allow_short and normalized in {"database", "db", "db-rag", "db rag"}


def _pending_clarification_has_database_context(state: AgentState) -> bool:
    meta = dict(state.get("meta") or {})
    if not meta.get(MetaKeys.AWAITING_USER_CLARIFICATION):
        return False
    if meta.get(MetaKeys.CLARIFICATION_RETURN_NODE) == "rag_db_qa":
        return True
    pending_question = str(meta.get(MetaKeys.PENDING_QUESTION) or "").strip().lower()
    return _mentions_explicit_rag_database(pending_question) or _mentions_database_reference(pending_question)


def _semantically_prefers_database_source(state: AgentState, latest: str) -> bool:
    if not latest or _has_local_dataframe_artifact(state):
        return False
    result = classify_database_source_intent(
        pending_question=latest,
        user_reply=latest,
    )
    return str(result.get("label") or "").strip().lower() == "database"


def _prefer_rag_db_qa(state: AgentState) -> bool:
    rag_state = get_agent_state(state, "rag_db_qa")
    if rag_state.get("active_thread"):
        return True

    latest = _latest_human_message(state)
    if not latest or _is_explicit_code_request(state):
        return False
    if is_database_source_reply(latest):
        return True

    return (
        _mentions_explicit_rag_database(latest)
        or _pending_clarification_has_database_context(state)
        or (not _has_uploaded_dataset(state) and _mentions_database_reference(latest))
        or _semantically_prefers_database_source(state, latest)
    )


def _always(_state: AgentState) -> bool:
    return True


PREDICATE_REGISTRY = {
    "dataset_analysis_request": _is_dataset_analysis_request,
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
            "human_review_rag_db_column_selection",
            "human_review_rag_db_sql_execution",
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
