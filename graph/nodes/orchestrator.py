"""Orchestrator node and all routing helpers.

Responsibilities
----------------
* Determine the next action on every graph step (hybrid LLM + deterministic policy).
* Detect and break infinite loops (recursion guards).
* Reset ephemeral state when a new user turn begins (context-memory fix).
* Infer user intent from the latest message (prototype vs. data-analysis).

Adding a new node
-----------------
1. Add a NodeDefinition to ``graph/nodes/node_registry.py`` — capability text,
   routing policy, and reset keys live there.
2. Register the callable in ``builder.py``'s ``action_nodes`` dict.
3. No changes needed here unless you need custom summary fields for the LLM.
"""
from __future__ import annotations

import hashlib
import json

from typing import Iterable

from ..state import AgentState, MetaKeys
from .state_helpers import get_agent_state
from .tool_routing import is_tool_requested
from .node_registry import (
    NODE_CAPABILITIES,
    NODE_REGISTRY,
    NODE_REGISTRY_MAP,
    validate_registry,
)
from prompts.planner_prompt import make_planner_prompt


# ---------------------------------------------------------------------------
# Loop-guard constants
# ---------------------------------------------------------------------------

MAX_ACTION_REPEATS = 3    # max same-action appearances in LOOP_GUARD_LOOKBACK window
LOOP_GUARD_LOOKBACK = 8   # trace entries examined by loop-detection helpers


# ---------------------------------------------------------------------------
# Intent-detection cue lists
# Keep cues short and domain-agnostic to reduce overfitting.
# ---------------------------------------------------------------------------

QA_LEADING_PHRASES = (
    "what is",
    "what's",
    "who is",
    "define",
    "explain",
    "tell me about",
    "which",
)

CODE_REQUEST_CUES = (
    "write code",
    "generate code",
    "show code",
    "python",
    "plot",
    "chart",
    "analyze",
    "calculate",
    "compute",
    "run",
)

DATA_OPERATION_CUES = (
    "dataset",
    "dataframe",
    "csv",
    "table",
    "columns",
)

# Risk-6 fix (part 1): prototype/tutorial phrasing that does NOT imply running code.
PROTOTYPE_CUES = (
    "example of",
    "how to",
    "show me how",
    "how do i",
    "how would i",
    "for example",
    "prototype",
    "demo",
)

# Risk-6 fix (part 2): explicit references to the user's own uploaded data.
OWN_DATA_CUES = (
    "my file",
    "my attached",
    "attached file",
    "this file",
    "this data",
    "this dataset",
    "my csv",
    "my excel",
    "uploaded",
    "for my data",
    "on my data",
    "my data",
)

INFO_CODE_CUES = (
    "sample code",
    "example code",
    "code example",
    "template",
    "for reference",
    "without running",
    "do not run",
    "dont run",
)

EXECUTION_CUES = (
    "execute",
    "run it",
    "use my dataset",
    "on my dataset",
    "for my dataset",
    "on this dataset",
    "fit the model",
)

# Generic analysis-action verbs that, combined with OWN_DATA_CUES, signal code-execution.
ANALYSIS_CUES = (
    "perform",
    "conduct",
    "carry out",
    "do a",
    "run a",
    "apply",
)


# ---------------------------------------------------------------------------
# Message helpers
# ---------------------------------------------------------------------------

def _latest_user_message(state: AgentState) -> str:
    messages = list(state.get("messages", []))
    for message in reversed(messages):
        if getattr(message, "type", None) == "human":
            return str(getattr(message, "content", "") or "").strip()
    return ""


def _has_unanswered_human_message(state: AgentState) -> bool:
    """Return True when the latest chat turn is a human message awaiting a reply."""
    messages = list(state.get("messages", []))
    if not messages:
        return False
    last_human_index = -1
    last_ai_index = -1
    for idx, message in enumerate(messages):
        message_type = getattr(message, "type", None)
        if message_type == "human":
            last_human_index = idx
        elif message_type == "ai":
            last_ai_index = idx
    return last_human_index > last_ai_index


# ---------------------------------------------------------------------------
# End-condition helpers
# ---------------------------------------------------------------------------

def _should_end_now(state: AgentState) -> bool:
    if (state.get("meta") or {}).get(MetaKeys.AWAITING_USER_CLARIFICATION):
        return True
    if state.get("last_action") == "qa" and not _has_unanswered_human_message(state):
        return True
    return False


# ---------------------------------------------------------------------------
# New-turn detection and ephemeral-state reset  (Risk-3 + context-memory fix)
# ---------------------------------------------------------------------------

def _user_message_hash(state: AgentState) -> str | None:
    """Return a short SHA-256 hex digest of the latest user message, or None."""
    msg = _latest_user_message(state)
    if not msg:
        return None
    return hashlib.sha256(msg.encode()).hexdigest()[:16]


def _reset_for_new_turn(output: dict, agents: dict, meta: dict) -> tuple[dict, dict, dict]:
    """Clear ephemeral routing/execution state when a new user message is detected.

    Preserves: messages history, workflow_trace, observations, last_user_message_hash.
    Resets:    generated_code, error state, human-review decisions, intent, code hashes.

    The set of agent keys to reset is driven by NodeDefinition.reset_agent_keys so
    new nodes self-register their cleanup needs (Risk-3 fix).
    """
    output = dict(output)
    output.pop("generated_code", None)
    output.pop("error", None)
    output.pop("tool_results", None)

    agents = dict(agents)
    # Collect all agent keys that need resetting from the registry.
    keys_to_reset: set[str] = set()
    for nd in NODE_REGISTRY:
        keys_to_reset.update(nd.reset_agent_keys)

    for key in keys_to_reset:
        if key in agents:
            agents[key] = {}  # wipe; get_agent_state will re-populate defaults on next access

    meta = dict(meta)
    meta.pop(MetaKeys.INTENT, None)
    meta.pop(MetaKeys.CURRENT_CODE_HASH, None)
    meta.pop(MetaKeys.AWAITING_USER_CLARIFICATION, None)
    meta.pop(MetaKeys.TOOL_REQUEST_QUEUE, None)
    meta[MetaKeys.ERROR_ITERATIONS] = 0

    return output, agents, meta


# ---------------------------------------------------------------------------
# Intent inference  (Risk-6 fix)
# ---------------------------------------------------------------------------

def infer_intent_from_latest_user(state: AgentState) -> str | None:
    """Classify the latest user message as 'code', 'qa', or None (unknown).

    Rule priority (highest first):
      1. Explicit no-execute request + code mention → qa
      2. Prototype/tutorial phrasing with no own-data reference → qa
      3. Any reference to the user's own data → code
      4. Score-based fallback using CODE_REQUEST_CUES vs. QA_LEADING_PHRASES
    """
    user_message = _latest_user_message(state).lower()
    if not user_message:
        return None

    # If QA previously asked for a required tool field, treat the next user
    # message as a QA follow-up even when it's a terse value with no keyword.
    qa_state = get_agent_state(state, "qa")
    if qa_state.get("awaiting_tool_clarification"):
        return "qa"

    has_code_request = any(token in user_message for token in CODE_REQUEST_CUES)
    has_info_code_request = any(token in user_message for token in INFO_CODE_CUES)
    has_prototype_request = any(token in user_message for token in PROTOTYPE_CUES)
    has_data_context = any(token in user_message for token in DATA_OPERATION_CUES)
    has_own_data = any(token in user_message for token in OWN_DATA_CUES)
    has_analysis_action = any(token in user_message for token in ANALYSIS_CUES)
    has_execution_request = (
        any(token in user_message for token in EXECUTION_CUES)
        or has_data_context
        or has_own_data
    )

    # Rule 1: "give me sample code / template" — never execute
    if has_code_request and has_info_code_request and not has_execution_request:
        return "qa"

    # Rule 2: Prototype/tutorial request with no reference to own data → QA
    # e.g. "give me example of how to perform survival analysis"
    if has_prototype_request and not has_own_data and not has_data_context:
        return "qa"

    # Rule 3: User references their own uploaded data → always execute
    # e.g. "help me perform survival analysis for my attached file"
    if has_own_data:
        return "code"

    # Rule 4: Analysis verb + data context → execute
    if has_analysis_action and has_data_context:
        return "code"

    code_score = 0
    if has_code_request:
        code_score += 1
    if has_data_context:
        code_score += 1

    qa_score = 0
    if user_message.endswith("?"):
        qa_score += 1
    if user_message.startswith(QA_LEADING_PHRASES):
        qa_score += 1

    if code_score >= 1 and (qa_score == 0 or code_score > qa_score):
        return "code"
    if qa_score >= 1:
        return "qa"
    return None


# ---------------------------------------------------------------------------
# Loop / cycle detection  (recursion-guard helpers)
# ---------------------------------------------------------------------------

def _detect_two_node_cycle(trace: list[str], lookback: int = LOOP_GUARD_LOOKBACK) -> bool:
    """Return True when the recent trace contains an A→B→A→B alternating pattern.

    Uses a sliding 4-element window so cycles that start just before the tail are
    still caught.
    """
    window = trace[-lookback:]
    if len(window) < 4:
        return False
    for i in range(len(window) - 3):
        a, b, c, d = window[i], window[i + 1], window[i + 2], window[i + 3]
        if a == c and b == d and a != b:
            return True
    return False


def _count_action_in_recent_trace(
    action: str,
    trace: list[str],
    lookback: int = LOOP_GUARD_LOOKBACK,
) -> int:
    """Count appearances of ``action`` within the last ``lookback`` trace entries."""
    return trace[-lookback:].count(action)


def _apply_loop_guards(
    next_action: str,
    state: AgentState,
    observations: list[str],
) -> tuple[str, list[str], bool]:
    """Apply cycle-detection and repeat-action guards.

    Returns (next_action, observations, guard_fired).
    Overrides next_action to 'end' when a loop is detected.
    """
    if next_action == "end":
        return next_action, observations, False

    trace = list((state.get("meta") or {}).get(MetaKeys.WORKFLOW_TRACE, []))
    obs = list(observations)

    # Guard 1: strict A→B→A→B cycle
    if _detect_two_node_cycle(trace):
        obs.append(
            f"orchestrator [loop_guard]: two-node cycle detected "
            f"(tail={trace[-LOOP_GUARD_LOOKBACK:]}); overriding '{next_action}' → 'end'"
        )
        return "end", obs, True

    # Guard 2: same action repeated too many times in the lookback window
    repeat_count = _count_action_in_recent_trace(next_action, trace)
    if repeat_count >= MAX_ACTION_REPEATS:
        obs.append(
            f"orchestrator [loop_guard]: '{next_action}' appeared {repeat_count}× "
            f"in the last {LOOP_GUARD_LOOKBACK} steps (max={MAX_ACTION_REPEATS}); "
            "overriding → 'end'"
        )
        return "end", obs, True

    return next_action, obs, False


# ---------------------------------------------------------------------------
# Tool-queue helpers
# ---------------------------------------------------------------------------

def _tool_request_queue(state: AgentState) -> list[str]:
    return list((state.get("meta") or {}).get(MetaKeys.TOOL_REQUEST_QUEUE, []))


def _next_tool_requester(state: AgentState) -> str | None:
    for requester in _tool_request_queue(state):
        agent_state = get_agent_state(state, requester)
        if agent_state.get("tool_results"):
            return requester
    return None


# ---------------------------------------------------------------------------
# Deterministic policy routing  (driven by the registry)
# ---------------------------------------------------------------------------

def choose_next_action(state: AgentState, available_actions: Iterable[str]) -> str:
    """Return the deterministic next action based on registry policies.

    Risk-2 fix: policies are sorted by NodeDefinition.priority, so ordering is
    explicit and documented rather than implicit list position.
    """
    intent = (
        (state.get("meta") or {}).get(MetaKeys.INTENT)
        or infer_intent_from_latest_user(state)
        or ""
    ).strip()
    available = set(available_actions)

    if _should_end_now(state):
        return "end"

    if intent == "qa" and "qa" in available and not is_tool_requested(state):
        return "qa"

    sorted_nodes = sorted(NODE_REGISTRY, key=lambda nd: nd.priority)
    for nd in sorted_nodes:
        if nd.name in available and nd.is_ready(state):
            return nd.name

    if get_agent_state(state, "human_review").get("final_decision") == "approve":
        return "end"

    return "end"


# ---------------------------------------------------------------------------
# LLM-based planning helpers  (Risk-4 fix: state summary includes registry status)
# ---------------------------------------------------------------------------

def _format_tool_results(state: AgentState) -> str:
    agents = state.get("agents", {})
    formatted: list[str] = []
    for agent_name, agent_state in agents.items():
        results = agent_state.get("tool_results", [])
        if not results:
            continue
        for result in results[-3:]:
            formatted.append(
                f"{agent_name}: {json.dumps(result, default=str, ensure_ascii=False)}"
            )
    return "\n".join(formatted) if formatted else "none"


def _format_state_summary(state: AgentState) -> str:
    """Build a compact state summary for the LLM planner prompt.

    Risk-4 fix: the registry-driven section at the bottom automatically surfaces
    the status of any registered node, so new nodes appear in the summary without
    manual edits here.
    """
    executor_state = get_agent_state(state, "executor")
    review_state = get_agent_state(state, "human_review")

    latest_user = _latest_user_message(state)
    inferred_intent = infer_intent_from_latest_user(state)

    trace = list((state.get("meta") or {}).get(MetaKeys.WORKFLOW_TRACE, []))
    trace_tail = trace[-LOOP_GUARD_LOOKBACK:]
    cycle_detected = _detect_two_node_cycle(trace)
    max_repeats = max(
        (_count_action_in_recent_trace(a, trace) for a in set(trace_tail)),
        default=0,
    )

    parts = [
        f"latest_user_message={latest_user}",
        f"inferred_intent={inferred_intent}",
        f"generated_code_present={bool((state.get('output') or {}).get('generated_code'))}",
        f"executor_run_status={executor_state.get('run_status')}",
        f"before_run_decision={review_state.get('before_run_decision')}",
        f"final_decision={review_state.get('final_decision')}",
        f"error_iterations={(state.get('meta') or {}).get(MetaKeys.ERROR_ITERATIONS, 0)}",
        f"tool_requests_pending={is_tool_requested(state)}",
        f"last_action={state.get('last_action')}",
        f"workflow_trace_tail={trace_tail}",
        f"loop_cycle_detected={cycle_detected}",
        f"max_action_repeats_in_window={max_repeats}",
    ]

    # Registry-driven node status — automatically includes any new node's status.
    agents = state.get("agents") or {}
    for nd in sorted(NODE_REGISTRY, key=lambda n: n.priority):
        agent_st = agents.get(nd.name, {})
        status = agent_st.get("status")
        if status and status not in ("idle", None):
            parts.append(f"{nd.name}_status={status}")

    parts.append("tool_results:\n" + _format_tool_results(state))
    return "\n".join(parts)


def _parse_planner_response(
    content: str,
    available_actions: Iterable[str],
) -> tuple[str, str]:
    content = content.strip()
    if not content:
        return "end", ""
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        action = content
        return action if action in set(available_actions) else "end", ""
    if not isinstance(payload, dict):
        return "end", ""
    action = payload.get("action", "")
    thought = payload.get("thought", "")
    if action in set(available_actions):
        return action, str(thought or "")
    return "end", str(thought or "")


def _format_node_capabilities(available_actions: Iterable[str]) -> str:
    lines: list[str] = []
    for action in sorted(set(available_actions)):
        description = NODE_CAPABILITIES.get(action, "No description provided.")
        lines.append(f"- {action}: {description}")
    return "\n".join(lines)


def llm_select_next_action(
    state: AgentState,
    llm,
    available_actions: Iterable[str],
) -> tuple[str, str]:
    actions = ", ".join(sorted(set(available_actions)))
    prompt = make_planner_prompt().format_prompt(
        actions=actions,
        summary=_format_state_summary(state),
        node_capabilities=_format_node_capabilities(available_actions),
    )
    response = llm.invoke(prompt.to_messages())
    return _parse_planner_response(response.content, available_actions)


# ---------------------------------------------------------------------------
# Orchestrator node
# ---------------------------------------------------------------------------

def orchestrator_node(state: AgentState, llm, available_actions: Iterable[str]) -> AgentState:
    orchestrator_state = dict(state.get("orchestrator", {}))
    next_action = orchestrator_state.get("next_action")
    thought = orchestrator_state.get("thought", "")
    meta = dict(state.get("meta") or {})

    # ------------------------------------------------------------------
    # New-turn detection — runs on every new user message.
    #
    # Two distinct cases:
    #   A) User answered a clarification question  → preserve context, route back
    #      to the node that asked the question (no state reset).
    #   B) New independent question                → full ephemeral-state reset.
    # ------------------------------------------------------------------
    output = dict(state.get("output") or {})
    agents = dict(state.get("agents") or {})
    current_hash = _user_message_hash(state)
    was_awaiting_clarification = bool(meta.get(MetaKeys.AWAITING_USER_CLARIFICATION))

    if current_hash and current_hash != meta.get(MetaKeys.LAST_USER_MESSAGE_HASH):
        if was_awaiting_clarification:
            # Case A: clarification answer — preserve everything, only update
            # the hash and route straight back to the node that asked.
            meta.pop(MetaKeys.AWAITING_USER_CLARIFICATION, None)
            meta[MetaKeys.LAST_USER_MESSAGE_HASH] = current_hash
            clarification_return = meta.get(MetaKeys.CLARIFICATION_RETURN_NODE, "qa")
            if clarification_return in set(available_actions):
                next_action = clarification_return
            else:
                next_action = "qa"
            orchestrator_state["next_action"] = next_action
        else:
            # Case B: genuinely new question — full ephemeral-state reset.
            output, agents, meta = _reset_for_new_turn(output, agents, meta)
            meta[MetaKeys.LAST_USER_MESSAGE_HASH] = current_hash
            next_action = None
            orchestrator_state.pop("next_action", None)

    # ------------------------------------------------------------------
    # Set intent once per turn (re-inferred after reset above).
    # Risk-6 fix: LLM choice is used to backfill intent when keyword
    # detection returns None, so the policy fires correctly on the next step.
    # ------------------------------------------------------------------
    inferred_intent = infer_intent_from_latest_user(state)
    if not meta.get(MetaKeys.INTENT) and inferred_intent:
        meta[MetaKeys.INTENT] = inferred_intent

    # ------------------------------------------------------------------
    # Routing decision
    # ------------------------------------------------------------------
    if not next_action:
        # Special case: return to the agent that requested tools after tool_handler ran.
        if state.get("last_action") == "tool_handler":
            requester = _next_tool_requester(state)
            if requester and requester in set(available_actions):
                next_action = requester
                queue = _tool_request_queue(state)
                meta[MetaKeys.TOOL_REQUEST_QUEUE] = [n for n in queue if n != requester]

        if not next_action:
            llm_choice, thought = llm_select_next_action(state, llm, available_actions)
            fallback_action = choose_next_action(state, available_actions)

            if llm_choice != "end" and not _should_end_now(state):
                next_action = llm_choice
                # Risk-6 fix: backfill intent from LLM choice when keyword detection
                # was inconclusive, so deterministic policies work on the next step.
                if not meta.get(MetaKeys.INTENT):
                    if llm_choice == "qa":
                        meta[MetaKeys.INTENT] = "qa"
                    elif llm_choice in ("generate_code", "execute_code", "error_handler"):
                        meta[MetaKeys.INTENT] = "code"
            else:
                next_action = fallback_action

    # ------------------------------------------------------------------
    # Recursion guards — applied AFTER routing decision, BEFORE committing.
    # Reads the trace as it existed before this orchestrator call appends itself.
    # ------------------------------------------------------------------
    observations = list(state.get("observations", []))
    next_action, observations, _guard_fired = _apply_loop_guards(
        next_action or "end", state, observations
    )

    # ------------------------------------------------------------------
    # Commit decision
    # ------------------------------------------------------------------
    orchestrator_state["next_action"] = next_action
    if thought:
        orchestrator_state["thought"] = thought
        thoughts = list(orchestrator_state.get("thoughts", []))
        thoughts.append(thought)
        orchestrator_state["thoughts"] = thoughts

    observations.append(f"orchestrator: next_action={next_action}")

    workflow_trace = list(meta.get(MetaKeys.WORKFLOW_TRACE, []))
    workflow_trace.append("orchestrator")
    meta[MetaKeys.WORKFLOW_TRACE] = workflow_trace[-100:]

    return {
        **state,
        "next_action": next_action,
        "last_action": state.get("last_action"),
        "orchestrator": orchestrator_state,
        "observations": observations,
        "output": output,
        "agents": agents,
        "meta": meta,
    }
