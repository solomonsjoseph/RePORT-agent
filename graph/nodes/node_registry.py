"""Central registry for all graph action nodes.

Every node is described by a single NodeDefinition.  When you add a new node:

  1. Add a NodeDefinition entry to NODE_REGISTRY below.
  2. Register the callable in builder.py's ``action_nodes`` dict.
  3. That is all — capability text, routing policy, reset behaviour, and LLM
     state-summary context are all derived automatically from the registry.

Risk-1 fix: knowledge that was previously scattered across three locations
  (builder.py, orchestrator.NODE_CAPABILITIES, orchestrator.build_default_policies)
  now lives in one place.

Risk-2 fix: ``validate_registry()`` asserts structural invariants at startup.

Risk-3 fix: ``reset_agent_keys`` on each NodeDefinition drives _reset_for_new_turn
  so new nodes self-register their cleanup needs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..state import AgentState
from .state_helpers import get_agent_state
from .tool_routing import is_tool_requested


MAX_ERROR_ITERATIONS = 5


# ---------------------------------------------------------------------------
# Reusable state predicates
# ---------------------------------------------------------------------------

def _error_iterations(state: AgentState) -> int:
    return int((state.get("meta") or {}).get("error_iterations", 0))


def _has_fresh_run_approval(state: AgentState) -> bool:
    review = get_agent_state(state, "human_review")
    if review.get("before_run_decision") != "approve":
        return False
    approved = review.get("approved_code_hash")
    current = (state.get("meta") or {}).get("current_code_hash")
    return bool(approved and current and approved == current)


# ---------------------------------------------------------------------------
# NodeDefinition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NodeDefinition:
    """Describes a single action node in the agent graph.

    Attributes:
        name:             Node identifier (must match the key in builder.action_nodes).
        capability:       One-sentence description shown to the LLM planner.
        priority:         Deterministic routing priority — lower numbers are
                          evaluated first.  Values must be unique across the
                          registry so ordering is unambiguous.
        is_ready:         Predicate called by the orchestrator to decide whether
                          this node should run given the current state.
        reset_agent_keys: Keys under state["agents"] that should be cleared when
                          a new user turn is detected.  Add any agent-level state
                          your node writes here so it does not bleed across turns.
    """
    name: str
    capability: str
    priority: int
    is_ready: Callable[[AgentState], bool]
    reset_agent_keys: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Registry
# Ordered by priority (ascending) for readability; the orchestrator sorts anyway.
# ---------------------------------------------------------------------------

NODE_REGISTRY: list[NodeDefinition] = [
    NodeDefinition(
        name="error_handler",
        capability=(
            "Revise broken code after execution failures and increment retry state."
        ),
        priority=10,
        is_ready=lambda s: (
            get_agent_state(s, "executor").get("run_status") == "error"
            and _error_iterations(s) < MAX_ERROR_ITERATIONS
        ),
        reset_agent_keys=[],  # meta["error_iterations"] is reset via _reset_for_new_turn
    ),
    NodeDefinition(
        name="human_review_after_error",
        capability="Ask human for guidance after repeated execution failures.",
        priority=20,
        is_ready=lambda s: (
            get_agent_state(s, "executor").get("run_status") == "error"
            and _error_iterations(s) >= MAX_ERROR_ITERATIONS
            and get_agent_state(s, "human_review").get("after_error_decision") is None
        ),
        reset_agent_keys=["human_review"],
    ),
    NodeDefinition(
        name="tool_handler",
        capability=(
            "Execute requested external tools and store tool results back to requesting agents."
        ),
        priority=30,
        is_ready=is_tool_requested,
        reset_agent_keys=[],
    ),
    NodeDefinition(
        name="qa",
        capability=(
            "Answer user questions directly in natural language (optionally using tool "
            "results), without code unless requested."
        ),
        priority=40,
        is_ready=lambda s: (
            (s.get("meta") or {}).get("intent") == "qa"
            and not is_tool_requested(s)
        ),
        reset_agent_keys=[],
    ),
    NodeDefinition(
        name="generate_code",
        capability=(
            "Generate Python analysis code from the user's analytical request and "
            "available context.  Usually when dataset and schema are not null."
        ),
        priority=50,
        is_ready=lambda s: (
            not (s.get("output") or {}).get("generated_code")
            and not is_tool_requested(s)
        ),
        reset_agent_keys=["generate_code"],
    ),
    NodeDefinition(
        name="human_review_before_run",
        capability="Ask human approval before running generated code.",
        priority=60,
        is_ready=lambda s: (
            bool((s.get("output") or {}).get("generated_code"))
            and get_agent_state(s, "executor").get("run_status") in ("idle", "pending")
            and get_agent_state(s, "human_review").get("before_run_decision") is None
        ),
        reset_agent_keys=["human_review"],
    ),
    NodeDefinition(
        name="execute_code",
        capability=(
            "Execute previously generated Python code against the loaded dataframe "
            "and collect outputs/errors."
        ),
        priority=70,
        is_ready=lambda s: (
            bool((s.get("output") or {}).get("generated_code"))
            and get_agent_state(s, "executor").get("run_status") in ("idle", "pending")
            and _has_fresh_run_approval(s)
        ),
        reset_agent_keys=["executor"],
    ),
    NodeDefinition(
        name="human_review_final",
        capability="Ask human approval of final successful output.",
        priority=80,
        is_ready=lambda s: (
            get_agent_state(s, "executor").get("run_status") == "ok"
            and get_agent_state(s, "human_review").get("final_decision") is None
        ),
        reset_agent_keys=["human_review"],
    ),
]

# ---------------------------------------------------------------------------
# Derived lookups — built once at import time, used by orchestrator + builder.
# ---------------------------------------------------------------------------

# Maps node name → one-sentence capability description for the LLM planner prompt.
NODE_CAPABILITIES: dict[str, str] = {nd.name: nd.capability for nd in NODE_REGISTRY}

# Maps node name → NodeDefinition for O(1) lookup.
NODE_REGISTRY_MAP: dict[str, NodeDefinition] = {nd.name: nd for nd in NODE_REGISTRY}


# ---------------------------------------------------------------------------
# Validation  (Risk-2 fix)
# ---------------------------------------------------------------------------

def validate_registry(known_action_names: list[str] | None = None) -> None:
    """Assert structural invariants of NODE_REGISTRY.

    Call this at application startup (inside build_graph) so registration
    mistakes surface immediately rather than as silent routing failures.

    Args:
        known_action_names: When provided, asserts that the registry names match
            this list exactly (order-independent).  Pass the keys of builder.py's
            ``action_nodes`` dict.
    """
    names = [nd.name for nd in NODE_REGISTRY]

    # Unique names
    assert len(names) == len(set(names)), (
        f"Duplicate node names in NODE_REGISTRY: {names}"
    )

    # Unique priorities — ambiguous ordering causes silent routing bugs
    priorities = [nd.priority for nd in NODE_REGISTRY]
    assert len(priorities) == len(set(priorities)), (
        "Duplicate priority values in NODE_REGISTRY — ordering would be ambiguous.\n"
        f"  name→priority: {dict(zip(names, priorities))}"
    )

    # All is_ready fields are callable
    for nd in NODE_REGISTRY:
        assert callable(nd.is_ready), (
            f"NodeDefinition '{nd.name}'.is_ready must be callable, got {type(nd.is_ready)}"
        )
        assert isinstance(nd.reset_agent_keys, list), (
            f"NodeDefinition '{nd.name}'.reset_agent_keys must be a list"
        )

    # Cross-check against builder's action_nodes when provided
    if known_action_names is not None:
        registry_set = set(names)
        builder_set = set(known_action_names)
        missing_from_registry = builder_set - registry_set
        missing_from_builder = registry_set - builder_set
        assert not missing_from_registry, (
            f"Nodes in builder but missing from NODE_REGISTRY: {missing_from_registry}\n"
            "Add a NodeDefinition for each missing node."
        )
        assert not missing_from_builder, (
            f"Nodes in NODE_REGISTRY but missing from builder action_nodes: {missing_from_builder}\n"
            "Either add the node callable to builder.py or remove it from the registry."
        )
