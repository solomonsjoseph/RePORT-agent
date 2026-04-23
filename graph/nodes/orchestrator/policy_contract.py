from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import yaml


DEFAULT_POLICY_PATH = (
    Path(__file__).resolve().parents[3] / "docs" / "superpowers" / "specs" / "orchestrator-gating-policy.md"
)

_YAML_BLOCK_RE = re.compile(r"```yaml\s*(.*?)```", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class PlannerFallbackRule:
    predicate: str
    action: str


@dataclass(frozen=True)
class RecurrencePolicy:
    max_stagnation_steps: int
    max_weak_progress_steps: int


@dataclass(frozen=True)
class OrchestratorPolicyContract:
    deterministic_control_actions: tuple[str, ...]
    planner_fallback_rules: tuple[PlannerFallbackRule, ...]
    recurrence: RecurrencePolicy


def _extract_yaml_block(text: str) -> str:
    match = _YAML_BLOCK_RE.search(text)
    if not match:
        raise ValueError("Policy markdown is missing a fenced ```yaml``` block.")
    return match.group(1).strip()


def load_policy_contract(
    path: str | Path = DEFAULT_POLICY_PATH,
    *,
    known_actions: set[str],
    known_predicates: set[str],
) -> OrchestratorPolicyContract:
    policy_path = Path(path)
    if not policy_path.exists():
        raise ValueError(f"Policy markdown does not exist: {policy_path}")

    payload = yaml.safe_load(_extract_yaml_block(policy_path.read_text(encoding="utf-8"))) or {}
    if not isinstance(payload, dict):
        raise ValueError("Policy YAML block must decode to a mapping.")

    deterministic_control_actions = tuple(payload.get("deterministic_control_actions") or ())
    for action in deterministic_control_actions:
        if action not in known_actions:
            raise ValueError(f"Unknown action in deterministic_control_actions: {action}")

    fallback_rules: list[PlannerFallbackRule] = []
    for entry in payload.get("planner_fallback_rules") or ():
        if not isinstance(entry, dict):
            raise ValueError("Each planner_fallback_rules entry must be a mapping.")
        predicate = str(entry.get("predicate") or "").strip()
        action = str(entry.get("action") or "").strip()
        if predicate not in known_predicates:
            raise ValueError(f"Unknown predicate in planner_fallback_rules: {predicate}")
        if action not in known_actions:
            raise ValueError(f"Unknown action in planner_fallback_rules: {action}")
        fallback_rules.append(PlannerFallbackRule(predicate=predicate, action=action))

    recurrence_data = dict(payload.get("recurrence") or {})
    recurrence = RecurrencePolicy(
        max_stagnation_steps=int(recurrence_data.get("max_stagnation_steps", 4)),
        max_weak_progress_steps=int(recurrence_data.get("max_weak_progress_steps", 4)),
    )

    return OrchestratorPolicyContract(
        deterministic_control_actions=deterministic_control_actions,
        planner_fallback_rules=tuple(fallback_rules),
        recurrence=recurrence,
    )
