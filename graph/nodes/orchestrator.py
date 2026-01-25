import json
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from ..state import AgentState
from .state_helpers import get_agent_state
from prompts.planner_prompt import make_planner_prompt


MAX_ERROR_ITERATIONS = 5


@dataclass(frozen=True)
class AgentPolicy:
    name: str
    is_ready: Callable[[AgentState], bool]


def _is_tool_requested(state: AgentState) -> bool:
    agents = state.get("agents", {})
    for agent_state in agents.values():
        if agent_state.get("tool_requests"):
            return True
    return False


def _error_iterations(state: AgentState) -> int:
    return int(state.get("meta", {}).get("error_iterations", 0))


def build_default_policies() -> list[AgentPolicy]:
    return [
        AgentPolicy(
            name="error_handler",
            is_ready=lambda s: get_agent_state(s, "executor").get("run_status") == "error"
            and _error_iterations(s) < MAX_ERROR_ITERATIONS,
        ),
        AgentPolicy(
            name="human_review_after_error",
            is_ready=lambda s: get_agent_state(s, "executor").get("run_status") == "error"
            and _error_iterations(s) >= MAX_ERROR_ITERATIONS
            and get_agent_state(s, "human_review").get("after_error_decision") is None,
        ),
        AgentPolicy(
            name="generate_code",
            is_ready=lambda s: not (s.get("output") or {}).get("generated_code"),
        ),
        AgentPolicy(
            name="tool_handler",
            is_ready=_is_tool_requested,
        ),
        AgentPolicy(
            name="human_review_before_run",
            is_ready=lambda s: (s.get("output") or {}).get("generated_code")
            and get_agent_state(s, "executor").get("run_status") in ("idle", "pending")
            and get_agent_state(s, "human_review").get("before_run_decision") is None,
        ),
        AgentPolicy(
            name="execute_code",
            is_ready=lambda s: (s.get("output") or {}).get("generated_code")
            and get_agent_state(s, "executor").get("run_status") in ("idle", "pending")
            and get_agent_state(s, "human_review").get("before_run_decision") == "approve",
        ),
        AgentPolicy(
            name="qa",
            is_ready=lambda s: s.get("meta", {}).get("intent") == "qa",
        ),
        AgentPolicy(
            name="human_review_final",
            is_ready=lambda s: get_agent_state(s, "executor").get("run_status") == "ok"
            and get_agent_state(s, "human_review").get("final_decision") is None,
        ),
    ]


def choose_next_action(state: AgentState, available_actions: Iterable[str]) -> str:
    available = set(available_actions)
    policies = build_default_policies()
    for policy in policies:
        if policy.name in available and policy.is_ready(state):
            return policy.name
    if get_agent_state(state, "human_review").get("final_decision") == "approve":
        return "end"
    return "end"


def _format_tool_results(state: AgentState) -> str:
    agents = state.get("agents", {})
    formatted: list[str] = []
    for agent_name, agent_state in agents.items():
        results = agent_state.get("tool_results", [])
        if not results:
            continue
        for result in results[-3:]:
            formatted.append(
                f"{agent_name}: {json.dumps(result, default=str, ensure_ascii=False)}"
            )
    return "\n".join(formatted) if formatted else "none"


def _format_state_summary(state: AgentState) -> str:
    executor_state = get_agent_state(state, "executor")
    review_state = get_agent_state(state, "human_review")
    parts = [
        f"generated_code_present={bool((state.get('output') or {}).get('generated_code'))}",
        f"executor_run_status={executor_state.get('run_status')}",
        f"before_run_decision={review_state.get('before_run_decision')}",
        f"final_decision={review_state.get('final_decision')}",
        f"error_iterations={_error_iterations(state)}",
        f"tool_requests_pending={_is_tool_requested(state)}",
        f"last_action={state.get('last_action')}",
        "tool_results:\n" + _format_tool_results(state),
    ]
    return "\n".join(parts)


def llm_select_next_action(
    state: AgentState,
    llm,
    available_actions: Iterable[str],
) -> str:
    actions = ", ".join(sorted(set(available_actions)))
    prompt = make_planner_prompt().format_prompt(
        actions=actions,
        summary=_format_state_summary(state),
    )
    response = llm.invoke(prompt.to_messages())
    choice = response.content.strip()
    return choice if choice in set(available_actions) else "end"


def orchestrator_node(state: AgentState, llm, available_actions: Iterable[str]) -> AgentState:
    orchestrator_state = dict(state.get("orchestrator", {}))
    next_action = orchestrator_state.get("next_action")
    if not next_action:
        llm_choice = llm_select_next_action(state, llm, available_actions)
        if llm_choice != "end":
            next_action = llm_choice
        else:
            next_action = choose_next_action(state, available_actions)
    orchestrator_state["next_action"] = next_action
    observations = list(state.get("observations", []))
    observations.append(f"orchestrator: next_action={next_action}")

    return {
        **state,
        "next_action": next_action,
        "last_action": "orchestrator",
        "orchestrator": orchestrator_state,
        "observations": observations,
    }
