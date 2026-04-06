from __future__ import annotations

from copy import deepcopy


def get_artifacts(state: dict) -> dict:
    artifacts = dict(state.get("artifacts") or {})
    output = dict(state.get("output") or {})
    artifacts.setdefault("generated_code", output.get("generated_code"))
    artifacts.setdefault("execution_output", output.get("text"))
    artifacts.setdefault("error", output.get("error"))
    artifacts.setdefault("tool_results", output.get("tool_results"))
    return artifacts


def get_node_data(state: dict, node_name: str) -> dict:
    node_data = dict(state.get("node_data") or {})
    if node_name in node_data:
        return dict(node_data[node_name] or {})
    return dict((state.get("agents") or {}).get(node_name) or {})


def get_planner_state(state: dict) -> dict:
    planner = dict(state.get("planner") or {})
    orchestrator = dict(state.get("orchestrator") or {})
    planner.setdefault("last_decision", orchestrator.get("last_decision"))
    planner.setdefault("decision_trace", list(orchestrator.get("thoughts", [])))
    return planner


def merge_state_patch(state: dict, patch: dict) -> dict:
    updated = deepcopy(state)
    for key, value in patch.items():
        if key in {"artifacts", "node_data", "planner", "meta"}:
            current = dict(updated.get(key) or {})
            current.update(value)
            updated[key] = current
        else:
            updated[key] = value

    artifacts = dict(updated.get("artifacts") or {})
    if artifacts:
        output = dict(updated.get("output") or {})
        if "generated_code" in artifacts:
            output["generated_code"] = artifacts["generated_code"]
        if "execution_output" in artifacts:
            output["text"] = artifacts["execution_output"]
        if "error" in artifacts:
            output["error"] = artifacts["error"]
        if "tool_results" in artifacts:
            output["tool_results"] = artifacts["tool_results"]
        updated["output"] = output

    node_data = dict(updated.get("node_data") or {})
    if node_data:
        agents = dict(updated.get("agents") or {})
        for name, value in node_data.items():
            agents[name] = dict(agents.get(name) or {}) | dict(value or {})
        updated["agents"] = agents

    return updated
