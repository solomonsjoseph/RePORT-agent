"""Orchestrator package split into focused modules."""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "apply_recurrence_guard",
    "classify_progress",
    "llm_select_next_action",
    "orchestrator_node",
    "update_recurrence_state",
]


def __getattr__(name: str):
    if name in {"apply_recurrence_guard", "classify_progress", "update_recurrence_state"}:
        module = import_module(".progress_controller", __name__)
        return getattr(module, name)
    if name == "llm_select_next_action":
        return import_module(".planner", __name__).llm_select_next_action
    if name == "orchestrator_node":
        return import_module(".node", __name__).orchestrator_node
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
