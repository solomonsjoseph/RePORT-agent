from __future__ import annotations

import json

from dataclasses import dataclass
from typing import Callable, Iterable

from ..state import AgentState
from .state_helpers import get_agent_state
from .tool_routing import is_tool_requested
from prompts.planner_prompt import make_planner_prompt


MAX_ERROR_ITERATIONS = 5

NODE_CAPABILITIES: dict[str, str] = {
    "generate_code": "Generate Python analysis code from the user's analytical request and available context.",
    "execute_code": "Execute previously generated Python code against the loaded dataframe and collect outputs/errors.",
    "error_handler": "Revise broken code after execution failures and increment retry state.",
    "human_review_after_error": "Ask human for guidance after repeated execution failures.",
    "human_review_before_run": "Ask human approval before running generated code.",
    "human_review_final": "Ask human approval of final successful output.",
    "tool_handler": "Execute requested external tools and store tool results back to requesting agents.",
    "qa": "Answer user questions directly in natural language (optionally using tool results), without code unless requested.",
    "end": "Stop graph execution for the current turn.",
}


@dataclass(frozen=True)
class AgentPolicy:
    name: str
    is_ready: Callable[[AgentState], bool]


def _error_iterations(state: AgentState) -> int:
    return int(state.get("meta", {}).get("error_iterations", 0))


def _tool_request_queue(state: AgentState) -> list[str]:
    return list(state.get("meta", {}).get("tool_request_queue", []))


def _next_tool_requester(state: AgentState) -> str | None:
    for requester in _tool_request_queue(state):
        agent_state = get_agent_state(state, requester)
        if agent_state.get("tool_results"):
            return requester
    return None


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
            name="tool_handler",
            is_ready=is_tool_requested,
        ),
        AgentPolicy(
            name="qa",
            is_ready=lambda s: s.get("meta", {}).get("intent") == "qa"
            and not is_tool_requested(s),
        ),
        AgentPolicy(
            name="generate_code",
            is_ready=lambda s: not (s.get("output") or {}).get("generated_code")
            and not is_tool_requested(s),
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
        f"tool_requests_pending={is_tool_requested(state)}",
        f"last_action={state.get('last_action')}",
        f"workflow_trace_tail={list(state.get('meta', {}).get('workflow_trace', []))[-8:]}",
        "tool_results:\n" + _format_tool_results(state),
    ]
    return "\n".join(parts)


def _parse_planner_response(
    content: str,
    available_actions: Iterable[str],
) -> tuple[str, str]:
    content = content.strip()
    if not content:
        return "end", ""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        action = content
        return action if action in set(available_actions) else "end", ""
    if not isinstance(payload, dict):
        return "end", ""
    action = payload.get("action", "")
    thought = payload.get("thought", "")
    if action in set(available_actions):
        return action, str(thought or "")
    return "end", str(thought or "")

def _format_node_capabilities(available_actions: Iterable[str]) -> str:
    lines: list[str] = []
    for action in sorted(set(available_actions)):
        description = NODE_CAPABILITIES.get(action, "No description provided.")
        lines.append(f"- {action}: {description}")
    return "\n".join(lines)


def llm_select_next_action(
    state: AgentState,
    llm,
    available_actions: Iterable[str],
) -> tuple[str, str]:
    actions = ", ".join(sorted(set(available_actions)))
    prompt = make_planner_prompt().format_prompt(
        actions=actions,
        summary=_format_state_summary(state),
        node_capabilities=_format_node_capabilities(available_actions),
    )
    response = llm.invoke(prompt.to_messages())
    return _parse_planner_response(response.content, available_actions)


def orchestrator_node(state: AgentState, llm, available_actions: Iterable[str]) -> AgentState:
    orchestrator_state = dict(state.get("orchestrator", {}))
    next_action = orchestrator_state.get("next_action")
    thought = orchestrator_state.get("thought", "")
    meta = dict(state.get("meta", {}))
    if not next_action:
        if state.get("last_action") == "tool_handler":
            requester = _next_tool_requester(state)
            if requester and requester in set(available_actions):
                next_action = requester
                queue = _tool_request_queue(state)
                meta["tool_request_queue"] = [name for name in queue if name != requester]
        if not next_action:
            llm_choice, thought = llm_select_next_action(state, llm, available_actions)
            if llm_choice != "end":
                next_action = llm_choice
            else:
                next_action = choose_next_action(state, available_actions)
    orchestrator_state["next_action"] = next_action
    if thought:
        orchestrator_state["thought"] = thought
        thoughts = list(orchestrator_state.get("thoughts", []))
        thoughts.append(thought)
        orchestrator_state["thoughts"] = thoughts
    observations = list(state.get("observations", []))
    observations.append(f"orchestrator: next_action={next_action}")

    workflow_trace = list(meta.get("workflow_trace", []))
    workflow_trace.append("orchestrator")
    meta["workflow_trace"] = workflow_trace[-100:]

    return {
        **state,
        "next_action": next_action,
        "last_action": state.get("last_action"),
        "orchestrator": orchestrator_state,
        "observations": observations,
        "meta": meta,
    }
