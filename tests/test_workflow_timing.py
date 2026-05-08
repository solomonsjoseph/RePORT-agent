from __future__ import annotations

import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.performance import append_workflow_timings, collect_timings, timing_stage


def _load_builder_wrappers():
    source = Path("graph/builder.py").read_text()
    tree = ast.parse(source)
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_run_and_mark", "_run_orchestrator_with_timing"}
    ]
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "append_workflow_timings": append_workflow_timings,
        "collect_timings": collect_timings,
        "timing_stage": timing_stage,
    }
    exec(compile(module, "graph/builder.py", "exec"), namespace)
    return namespace["_run_and_mark"], namespace["_run_orchestrator_with_timing"]


def test_run_and_mark_records_action_node_timing() -> None:
    _run_and_mark, _run_orchestrator_with_timing = _load_builder_wrappers()
    del _run_orchestrator_with_timing

    wrapped = _run_and_mark(
        "qa",
        lambda state: {
            **state,
            "output": {"qa_response": "done"},
        },
    )

    updated = wrapped({"meta": {}, "orchestrator": {}, "planner": {}})

    stages = updated["meta"]["workflow_timing"]["stages"]
    assert stages[-1]["stage"] == "node.qa"
    assert stages[-1]["elapsed_ms"] >= 0
    assert updated["last_action"] == "qa"
    assert updated["next_action"] is None


def test_orchestrator_wrapper_records_timing_without_clearing_next_action() -> None:
    _run_and_mark, _run_orchestrator_with_timing = _load_builder_wrappers()
    del _run_and_mark

    wrapped = _run_orchestrator_with_timing(
        lambda state: {
            **state,
            "next_action": "qa",
            "orchestrator": {"next_action": "qa"},
        }
    )

    updated = wrapped({"meta": {}, "orchestrator": {}, "planner": {}})

    stages = updated["meta"]["workflow_timing"]["stages"]
    assert stages[-1]["stage"] == "node.orchestrator"
    assert stages[-1]["elapsed_ms"] >= 0
    assert updated["next_action"] == "qa"
    assert updated["orchestrator"]["next_action"] == "qa"
