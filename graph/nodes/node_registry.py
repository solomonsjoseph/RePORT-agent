"""Central registry for graph action metadata and deterministic predicates.

Every node is described by a single NodeDefinition.  When you add a new node:

  1. Add a NodeDefinition entry to NODE_REGISTRY below.
  2. Register the callable in builder.py's ``action_nodes`` dict.
  3. That is all for transitional static structure and validation. Planner-
     facing capability text is sourced from the node modules via
     action_metadata.py, and semantic routing now lives outside this registry.

Risk-1 fix: routing metadata that was previously scattered across multiple
  locations now lives in one place.

Risk-2 fix: ``validate_registry()`` asserts structural invariants at startup.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..state import AgentState, MetaKeys
from .action_metadata import ACTION_CAPABILITIES
from .state_helpers import get_agent_state
from .tool_routing import is_tool_requested


MAX_ERROR_ITERATIONS = 5


# ---------------------------------------------------------------------------
# Reusable state predicates
# ---------------------------------------------------------------------------

def _error_iterations(state: AgentState) -> int:
    return int((state.get("meta") or {}).get("error_iterations", 0))


def _execution_error(state: AgentState) -> dict:
    return dict((state.get("output") or {}).get("error") or {})


def _execution_error_category(state: AgentState) -> str | None:
    return _execution_error(state).get("category")


def _has_retryable_execution_error(state: AgentState) -> bool:
    return (
        get_agent_state(state, "executor").get("run_status") == "error"
        and _execution_error_category(state) == "retryable_code"
    )


def _has_terminal_execution_error(state: AgentState) -> bool:
    return (
        get_agent_state(state, "executor").get("run_status") == "error"
        and _execution_error_category(state) in {
            "policy_blocked",
            "unsupported_runtime",
            "infrastructure",
            "timeout",
        }
    )


def _has_exhausted_retryable_execution_error(state: AgentState) -> bool:
    return (
        _has_retryable_execution_error(state)
        and _error_iterations(state) >= MAX_ERROR_ITERATIONS
    )


def _has_fresh_run_approval(state: AgentState) -> bool:
    meta = dict(state.get("meta") or {})
    approved = meta.get(MetaKeys.EXECUTION_TICKET_HASH)
    current = meta.get(MetaKeys.CURRENT_CODE_HASH)
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
        is_ready:         Predicate for deterministic invariants and action
                          masking. It is not a semantic planner fallback.
    """
    name: str
    capability: str
    priority: int
    is_ready: Callable[[AgentState], bool]


# ---------------------------------------------------------------------------
# Registry
# Ordered by priority (ascending) for readability; the orchestrator sorts anyway.
# ---------------------------------------------------------------------------

NODE_REGISTRY: list[NodeDefinition] = [
    NodeDefinition(
        name="error_handler",
        capability=ACTION_CAPABILITIES["error_handler"],
        priority=10,
        is_ready=lambda s: (
            _has_retryable_execution_error(s)
            and _error_iterations(s) < MAX_ERROR_ITERATIONS
        ),
    ),
    NodeDefinition(
        name="terminal_execution_error",
        capability=ACTION_CAPABILITIES["terminal_execution_error"],
        priority=20,
        is_ready=lambda s: (
            _has_terminal_execution_error(s)
        ),
    ),
    NodeDefinition(
        name="human_review_after_error",
        capability=ACTION_CAPABILITIES["human_review_after_error"],
        priority=25,
        is_ready=lambda s: (
            (
                _has_exhausted_retryable_execution_error(s)
            )
            and get_agent_state(s, "human_review").get("after_error_decision") is None
        ),
    ),
    NodeDefinition(
        name="tool_handler",
        capability=ACTION_CAPABILITIES["tool_handler"],
        priority=30,
        is_ready=is_tool_requested,
    ),
    NodeDefinition(
        name="clarification",
        capability=ACTION_CAPABILITIES["clarification"],
        priority=35,
        is_ready=lambda s: bool((s.get("meta") or {}).get(MetaKeys.AWAITING_USER_CLARIFICATION)),
    ),
    NodeDefinition(
        name="qa",
        capability=ACTION_CAPABILITIES["qa"],
        priority=40,
        is_ready=lambda s: (
            not bool((s.get("output") or {}).get("generated_code"))
            and not is_tool_requested(s)
        ),
    ),
    NodeDefinition(
        name="generate_code",
        capability=ACTION_CAPABILITIES["generate_code"],
        priority=50,
        is_ready=lambda s: (
            not (s.get("output") or {}).get("generated_code")
            and not is_tool_requested(s)
        ),
    ),
    NodeDefinition(
        name="human_review_before_run",
        capability=ACTION_CAPABILITIES["human_review_before_run"],
        priority=60,
        is_ready=lambda s: (
            bool((s.get("output") or {}).get("generated_code"))
            and get_agent_state(s, "executor").get("run_status") in ("idle", "pending")
            and not _has_fresh_run_approval(s)
        ),
    ),
    NodeDefinition(
        name="execute_code",
        capability=ACTION_CAPABILITIES["execute_code"],
        priority=70,
        is_ready=lambda s: (
            bool((s.get("output") or {}).get("generated_code"))
            and get_agent_state(s, "executor").get("run_status") in ("idle", "pending")
            and _has_fresh_run_approval(s)
        ),
    ),
    NodeDefinition(
        name="human_review_final",
        capability=ACTION_CAPABILITIES["human_review_final"],
        priority=80,
        is_ready=lambda s: (
            get_agent_state(s, "executor").get("run_status") == "ok"
            and get_agent_state(s, "human_review").get("final_decision") is None
        ),
    ),
]

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
