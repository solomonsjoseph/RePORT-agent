from __future__ import annotations

import ast
from pathlib import Path


_ACTION_MODULES = (
    "qa",
    "clarification",
    "generate_code",
    "execute_code",
    "error_handler",
    "tool_handler",
    "human_review_before_run",
    "human_review_after_error",
    "human_review_final",
    "terminal_execution_error",
)


def _load_node_action_metadata(module_name: str) -> tuple[str, str]:
    path = Path(__file__).with_name(f"{module_name}.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    metadata: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            if target.id not in {"NODE_NAME", "NODE_CAPABILITY"}:
                continue
            metadata[target.id] = ast.literal_eval(node.value)
        if len(metadata) == 2:
            break

    missing = {"NODE_NAME", "NODE_CAPABILITY"} - set(metadata)
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(f"{path.name} is missing required action metadata: {missing_names}")

    return metadata["NODE_NAME"], metadata["NODE_CAPABILITY"]


ACTION_CAPABILITIES: dict[str, str] = dict(
    _load_node_action_metadata(module_name)
    for module_name in _ACTION_MODULES
)

