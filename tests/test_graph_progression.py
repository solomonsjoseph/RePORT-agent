from __future__ import annotations

import ast
from pathlib import Path


def _load_run_and_mark():
    source = Path("graph/builder.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    target = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_run_and_mark":
            target = node
            break
    assert target is not None

    module = ast.Module(body=[target], type_ignores=[])
    ast.fix_missing_locations(module)
    scope: dict[str, object] = {}
    exec(compile(module, filename="graph/builder.py", mode="exec"), scope)
    return scope["_run_and_mark"]


def test_run_and_mark_clears_next_action_and_marks_last_action() -> None:
    run_and_mark = _load_run_and_mark()

    wrapped = run_and_mark(
        "qa",
        lambda state: {
            **state,
            "orchestrator": {"next_action": "qa", "thought": "done"},
        },
    )
    result = wrapped(
        {
            "next_action": "qa",
            "orchestrator": {"next_action": "qa"},
            "meta": {"workflow_trace": ["orchestrator"]},
        }
    )

    assert result["next_action"] is None
    assert result["last_action"] == "qa"
    assert result["orchestrator"] == {"thought": "done"}
    assert result["meta"]["workflow_trace"] == ["orchestrator", "qa"]


def test_run_and_mark_consumes_loop_guard_bypass_for_current_node() -> None:
    run_and_mark = _load_run_and_mark()

    wrapped = run_and_mark(
        "generate_code",
        lambda state: {**state, "meta": {"workflow_trace": ["orchestrator"], "loop_guard_bypass_actions": ["generate_code", "execute_code"]}},
    )
    result = wrapped({"meta": {"workflow_trace": ["orchestrator"], "loop_guard_bypass_actions": ["generate_code", "execute_code"]}})

    assert result["meta"]["workflow_trace"] == ["orchestrator", "generate_code"]
    assert result["meta"].get("loop_guard_bypass_actions") == ["execute_code"]
