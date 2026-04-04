"""Central registry for all graph action nodes.

Every node is described by a single NodeDefinition.  When you add a new node:

  1. Add a NodeDefinition entry to NODE_REGISTRY below.
  2. Register the callable in builder.py's ``action_nodes`` dict.
  3. That is all — capability text, routing policy, and LLM
     state-summary context are all derived automatically from the registry.

Risk-1 fix: knowledge that was previously scattered across three locations
  (builder.py, orchestrator.NODE_CAPABILITIES, orchestrator.build_default_policies)
  now lives in one place.

Risk-2 fix: ``validate_registry()`` asserts structural invariants at startup.

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..state import AgentState, MetaKeys
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
    review = get_agent_state(state, "human_review")
    if review.get("before_run_decision") != "approve":
        return False
    approved = review.get("approved_code_hash")
    current = (state.get("meta") or {}).get("current_code_hash")
    return bool(approved and current and approved == current)


def _is_code_clarification_continuation(state: AgentState) -> bool:
    meta = state.get("meta") or {}
    if not meta.get(MetaKeys.AWAITING_USER_CLARIFICATION):
        return False

    return_node = meta.get(MetaKeys.CLARIFICATION_RETURN_NODE)
    if return_node in ("generate_code", "error_handler"):
        return True

    # Backward-compatible handling for older threads that set only the generic
    # clarification flag from generate_code/error_handler without a return node.
    return state.get("last_action") in ("generate_code", "error_handler")


def _has_affirmative_code_readiness(state: AgentState) -> bool:
    meta = state.get("meta") or {}
    if "generate_code" in list(meta.get(MetaKeys.LOOP_GUARD_BYPASS_ACTIONS, [])):
        return True
    if _is_code_clarification_continuation(state):
        return True
    return False


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
        capability=(
            "Revise previously generated code after execution failures and increment retry state."
        ),
        priority=10,
        is_ready=lambda s: (
            _has_retryable_execution_error(s)
            and _error_iterations(s) < MAX_ERROR_ITERATIONS
        ),
    ),
    NodeDefinition(
        name="terminal_execution_error",
        capability="Explain terminal execution failures directly to the user and end the turn without retrying.",
        priority=20,
        is_ready=lambda s: (
            _has_terminal_execution_error(s)
        ),
    ),
    NodeDefinition(
        name="human_review_after_error",
        capability="Ask human for guidance after repeated retryable code-execution failures once retry budget is exhausted.",
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
        capability=(
            "Execute already-requested external tools and store results back to the requesting agent. "
            "Do not use for routing decisions or direct user replies."
        ),
        priority=30,
        is_ready=is_tool_requested,
    ),
    NodeDefinition(
        name="clarification",
        capability=(
            "Resume an active clarification loop by interpreting the user's follow-up and handing "
            "control back to the relevant subworkflow such as QA tool routing or code generation."
        ),
        priority=35,
        is_ready=lambda s: bool((s.get("meta") or {}).get(MetaKeys.AWAITING_USER_CLARIFICATION)),
    ),
    NodeDefinition(
        name="qa",
        capability=(
            "Handle direct user-facing Q&A in natural language, including factual and explanatory "
            "requests, conversation, and tool-assisted information tasks such as search, weather, "
            "and calculator queries. Prefer this over code generation unless the user explicitly "
            "wants code or dataset/programmatic work."
        ),
        priority=40,
        is_ready=lambda s: (
            not bool((s.get("output") or {}).get("generated_code"))
            and not is_tool_requested(s)
        ),
    ),
    NodeDefinition(
        name="generate_code",
        capability=(
            "Generate Python code only for explicit code-writing requests or dataset/programmatic "
            "tasks such as analysis on the user's data, plotting, transformation, or computation "
            "that should be performed in code. Do not use for general Q&A, web search, weather, "
            "or factual lookup."
        ),
        priority=50,
        is_ready=lambda s: (
            not (s.get("output") or {}).get("generated_code")
            and not is_tool_requested(s)
        ),
    ),
    NodeDefinition(
        name="human_review_before_run",
        capability="Ask human approval before running newly generated code.",
        priority=60,
        is_ready=lambda s: (
            bool((s.get("output") or {}).get("generated_code"))
            and get_agent_state(s, "executor").get("run_status") in ("idle", "pending")
            and get_agent_state(s, "human_review").get("before_run_decision") is None
        ),
    ),
    NodeDefinition(
        name="execute_code",
        capability=(
            "Execute previously generated Python code after approval and collect outputs or errors."
        ),
        priority=70,
        is_ready=lambda s: (
            bool((s.get("output") or {}).get("generated_code"))
            and get_agent_state(s, "executor").get("run_status") in ("idle", "pending")
            and _has_fresh_run_approval(s)
        ),
    ),
    NodeDefinition(
        name="human_review_final",
        capability="Ask human approval of the final successful code-execution output before ending the task.",
        priority=80,
        is_ready=lambda s: (
            get_agent_state(s, "executor").get("run_status") == "ok"
            and get_agent_state(s, "human_review").get("final_decision") is None
        ),
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
