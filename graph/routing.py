from langgraph.graph import END

from .state import AgentState


def _before_run_decision(state: AgentState) -> str | None:
    agents = state.get("agents", {})
    review = agents.get("human_review", {}) if isinstance(agents, dict) else {}
    decision = review.get("before_run_decision")
    return str(decision) if decision is not None else None


def _is_current_code_approved(state: AgentState) -> bool:
    agents = state.get("agents", {})
    review = agents.get("human_review", {}) if isinstance(agents, dict) else {}
    approved_hash = review.get("approved_code_hash")
    current_hash = (state.get("meta", {}) or {}).get("current_code_hash")
    return bool(approved_hash and current_hash and approved_hash == current_hash)


def _executor_run_status(state: AgentState) -> str | None:
    agents = state.get("agents", {})
    executor = agents.get("executor", {}) if isinstance(agents, dict) else {}
    status = executor.get("run_status")
    return str(status) if status is not None else None


def _final_review_pending(state: AgentState) -> bool:
    agents = state.get("agents", {})
    review = agents.get("human_review", {}) if isinstance(agents, dict) else {}
    return review.get("final_decision") is None


def route_by_next_action(state: AgentState):
    next_action = state.get("next_action")
    if not next_action:
        return END
    if next_action == "end":
        return END

    # Deterministic safety gate: never execute generated code without explicit
    # human approval captured in human_review.before_run_decision.
    if next_action == "execute_code":
        if _before_run_decision(state) != "approve" or not _is_current_code_approved(state):
            return "human_review_before_run"

        # Prevent repeated execute loops after a successful run. Once run_status
        # is "ok", route to final review instead of re-entering execute_code.
        if _executor_run_status(state) == "ok" and _final_review_pending(state):
            return "human_review_final"

    return next_action
