from __future__ import annotations

from copy import deepcopy


def _merge_dicts(base: dict | None, patch: dict | None) -> dict:
    merged = dict(base or {})
    for key, value in dict(patch or {}).items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(existing, value)
        else:
            merged[key] = value
    return merged


def get_artifacts(state: dict) -> dict:
    artifacts = dict(state.get("artifacts") or {})
    output = dict(state.get("output") or {})
    artifacts.setdefault("generated_code", output.get("generated_code"))
    artifacts.setdefault("execution_output", output.get("text"))
    artifacts.setdefault("error", output.get("error"))
    artifacts.setdefault("tool_results", output.get("tool_results"))
    return artifacts


def get_node_data(state: dict, node_name: str) -> dict:
    agents = dict((state.get("agents") or {}).get(node_name) or {})
    node_data = dict((state.get("node_data") or {}).get(node_name) or {})
    return _merge_dicts(agents, node_data)


def get_planner_state(state: dict) -> dict:
    orchestrator = dict(state.get("orchestrator") or {})
    planner = dict(state.get("planner") or {})
    return _merge_dicts(orchestrator, planner)


def merge_state_patch(state: dict, patch: dict) -> dict:
    updated = deepcopy(state)
    for key, value in patch.items():
        if key in {"artifacts", "node_data", "planner", "meta", "orchestrator"}:
            updated[key] = _merge_dicts(updated.get(key), value)
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
            agents[name] = _merge_dicts(agents.get(name), value)
        updated["agents"] = agents

    planner = dict(updated.get("planner") or {})
    if planner:
        updated["orchestrator"] = _merge_dicts(updated.get("orchestrator"), planner)

    return updated
