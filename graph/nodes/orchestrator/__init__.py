"""Orchestrator package split into focused modules."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "_apply_loop_guards",
    "_detect_two_node_cycle",
    "choose_next_action",
    "llm_select_next_action",
    "orchestrator_node",
]


def __getattr__(name: str):
    if name in {"_apply_loop_guards", "_detect_two_node_cycle"}:
        module = import_module(".loop_guards", __name__)
        return getattr(module, name)
    if name == "choose_next_action":
        return import_module(".policy", __name__).choose_next_action
    if name == "llm_select_next_action":
        return import_module(".planner", __name__).llm_select_next_action
    if name == "orchestrator_node":
        return import_module(".node", __name__).orchestrator_node
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
