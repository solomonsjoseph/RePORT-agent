from langgraph.graph import END

from .state import AgentState, MetaKeys


def _has_execution_ticket(state: AgentState) -> bool:
    meta = dict(state.get("meta") or {})
    ticket_hash = meta.get(MetaKeys.EXECUTION_TICKET_HASH)
    current_hash = meta.get(MetaKeys.CURRENT_CODE_HASH)
    return bool(ticket_hash and current_hash and ticket_hash == current_hash)

def route_by_next_action(state: AgentState):
    next_action = state.get("next_action")
    if not next_action:
        return END
    if next_action == "end":
        return END

    # Deterministic safety gate: never execute generated code without a live
    # execution ticket for the current code lineage.
    if next_action == "execute_code":
        if not _has_execution_ticket(state):
            return "human_review_before_run"

    return next_action
