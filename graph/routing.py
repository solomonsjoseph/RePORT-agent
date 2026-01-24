from langgraph.graph import END

from .state import AgentState


def route_by_next_action(state: AgentState):
    next_action = state.get("next_action")
    if not next_action:
        return END
    if next_action == "end":
        return END
    return next_action
