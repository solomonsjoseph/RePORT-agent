from __future__ import annotations

from pathlib import Path

import pytest


def test_load_policy_contract_parses_yaml_block(tmp_path: Path) -> None:
    from graph.nodes.orchestrator.policy_contract import load_policy_contract

    policy_path = tmp_path / "orchestrator-policy.md"
    policy_path.write_text(
        """
# Orchestrator Policy

```yaml
deterministic_control_actions:
  - tool_handler
  - human_review_before_run
planner_fallback_rules:
  - predicate: prefer_rag_db_qa
    action: rag_db_qa
  - predicate: always
    action: qa
recurrence:
  max_stagnation_steps: 6
  max_weak_progress_steps: 5
```
""".strip(),
        encoding="utf-8",
    )

    policy = load_policy_contract(
        policy_path,
        known_actions={
            "tool_handler",
            "human_review_before_run",
            "rag_db_qa",
            "qa",
            "end",
        },
        known_predicates={"prefer_rag_db_qa", "always"},
    )

    assert policy.deterministic_control_actions == ("tool_handler", "human_review_before_run")
    assert policy.planner_fallback_rules[0].predicate == "prefer_rag_db_qa"
    assert policy.planner_fallback_rules[0].action == "rag_db_qa"
    assert policy.recurrence.max_stagnation_steps == 6
    assert policy.recurrence.max_weak_progress_steps == 5


def test_load_policy_contract_rejects_unknown_actions(tmp_path: Path) -> None:
    from graph.nodes.orchestrator.policy_contract import load_policy_contract

    policy_path = tmp_path / "orchestrator-policy.md"
    policy_path.write_text(
        """
```yaml
deterministic_control_actions:
  - does_not_exist
planner_fallback_rules:
  - predicate: always
    action: qa
recurrence:
  max_stagnation_steps: 4
  max_weak_progress_steps: 4
```
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as excinfo:
        load_policy_contract(
            policy_path,
            known_actions={"qa", "end"},
            known_predicates={"always"},
        )

    assert "Unknown action" in str(excinfo.value)
