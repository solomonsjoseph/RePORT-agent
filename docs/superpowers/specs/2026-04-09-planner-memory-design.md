# Planner Memory Design

## Goal

Preserve multi-turn user intent for orchestrator routing without turning the planner into a full transcript consumer.

The planner should remain primarily driven by structured routing state. It should gain a small amount of durable conversation memory plus a tightly bounded transcript fallback for short-range disambiguation.

## Problem

Today the planner prompt is built from distilled state, including `latest_user_message`, workflow status, observations, blocked actions, and decision trace. This keeps routing cheap and stable, but it can lose important multi-turn intent when that intent is not explicitly carried forward in state.

Example failure modes:

- The user states a preference or constraint two turns earlier, then follows up with "run it" or "do the second option".
- A clarification flow resolves the meaning of the request, but the planner later only sees the final short reply rather than the clarified intent.
- The latest message is semantically incomplete on its own, while the prior one contains the real routing signal.

## Design Principles

- Keep the planner mostly transcript-free.
- Prefer structured state over natural-language chat history.
- Add only the minimum additional context needed for robust routing.
- Make planner memory auditable, cheap, and stable.
- Keep raw dataset context, code, and long error payloads out of the planner.

## Proposed Changes

### 1. Add persistent planner memory fields

Add a small planner-facing memory object to orchestrator state. The initial version should include:

- `active_user_goal`: one sentence describing the current operative task
- `conversation_intent_summary`: a short rolling summary of the current thread
- `unresolved_user_constraints`: a short list of active user constraints, preferences, or requirements that still affect routing

These fields are routing memory, not general-purpose semantic memory. They should stay compact and only contain information that can change the next action selection.

### 2. Update planner memory on user and control transitions

Planner memory should be refreshed when one of the following occurs:

- a new human message arrives
- a clarification exchange changes the interpreted task
- a review checkpoint produces a routing-relevant decision such as approve or regenerate
- a tool request or tool result materially changes what the user is asking for next

Planner memory should not be rewritten on every internal step. Execution status, retries, and review states already live elsewhere in orchestrator state and should remain separate.

### 3. Add a tiny transcript fallback only when needed

The planner should receive a small raw conversation window only in narrow cases:

- there is an unanswered human message
- the system is in a clarification flow

The window should be capped at the last 2 turns, with the intended shape:

- most recent AI message, if present
- most recent human message
- previous paired turn when available

In practice this means up to 4 recent messages, but only enough to resolve short references such as:

- "that"
- "same as before"
- "the second option"
- "run it"

This transcript snippet is a disambiguation aid, not the primary routing input.

### 4. Explicit precedence rules

When structured planner memory and the transcript snippet differ:

- structured planner memory wins
- workflow status and action masks still act as hard control constraints
- the transcript snippet is used only to interpret recent references that are not yet fully encoded in state

This prevents the planner from being swayed by incidental language in recent turns when durable routing state already captures the intended behavior.

## Planner Prompt Changes

The planner prompt should be extended with a dedicated planner memory section and an optional recent-turns section.

The prompt should present:

- environment summary
- planner memory
- recent observations
- planner decision trace
- blocked actions
- recent conversation turns, only when transcript fallback is enabled

The prompt text should explicitly instruct the planner:

- use planner memory as the primary source of user intent
- use recent conversation turns only to resolve short-range ambiguity
- do not ignore action masks or workflow status because of transcript wording

## State Model

### Planner memory shape

The planner state should grow to include a memory payload similar to:

```python
planner = {
    "decision_trace": [...],
    "last_decision": {...},
    "memory": {
        "active_user_goal": str,
        "conversation_intent_summary": str,
        "unresolved_user_constraints": list[str],
    },
}
```

### Transcript fallback shape

The planner context builder should optionally include:

```python
recent_turns_for_planner = [
    {"role": "ai", "content": "..."},
    {"role": "human", "content": "..."},
]
```

This field should be omitted or set to an explicit sentinel when fallback is not active.

## Update Logic

### Active user goal

`active_user_goal` should capture the operative task the orchestrator is routing toward right now.

Examples:

- "Answer the user's conceptual question about the orchestrator design"
- "Generate analysis code for the requested survival comparison"
- "Resume clarification for the user's unresolved tool request"

This field should change when the user meaningfully changes tasks or when clarification resolves an ambiguous task into a concrete one.

### Conversation intent summary

`conversation_intent_summary` should be a compact summary of the current thread and recent progression. It should preserve context that may matter for the next routing decision but does not fit cleanly into single-state flags.

Examples:

- "User is discussing orchestrator design and comparing planner context strategies."
- "User requested code-based dataset analysis and approved moving from planning to execution review."

### Unresolved user constraints

`unresolved_user_constraints` should store active constraints that still matter for routing. These are not generic preferences unless they can affect which node should run next.

Examples:

- "Prefer planner to remain transcript-light"
- "Do not add fallback solution branches"
- "Need human approval before execution"

Resolved or obsolete constraints should be removed.

## Implementation Boundaries

### In scope

- extend planner state with memory fields
- add helper logic to derive and update planner memory
- extend planner prompt and context builder
- add optional recent-turn snippet for planner disambiguation
- add tests for memory persistence, transcript gating, and prompt formatting

### Out of scope

- giving the planner full raw conversation history
- changing specialist node prompts beyond what is needed to preserve current behavior
- turning planner memory into a generalized long-term memory system
- moving dataset/schema context into the planner

## Suggested Code Changes

Likely implementation points:

- `graph/nodes/orchestrator/context_builder.py`
  - include planner memory in the planner context
  - add conditional recent-turn extraction
- `prompts/planner_prompt.py`
  - add planner memory and optional recent-turn sections
- `graph/nodes/orchestrator/node.py`
  - update planner memory when new user turns or routing-relevant control transitions occur
- `graph/nodes/orchestrator/state_logic.py`
  - add helpers for extracting recent planner turns and updating memory fields
- tests
  - prompt formatting
  - planner memory update behavior
  - transcript fallback activation conditions

## Testing Strategy

Add targeted unit tests for:

- planner prompt includes memory fields
- transcript fallback is absent during normal stable routing
- transcript fallback is present when there is an unanswered human message
- transcript fallback is present during clarification flow
- planner memory updates on new user turns
- planner memory persists across internal node transitions
- structured memory takes precedence over transcript snippet in routing behavior where applicable

## Risks

### Risk: memory becomes stale

If planner memory is not updated on task shifts, the planner may route based on old intent.

Mitigation:

- update memory on each new human message
- refresh memory on clarification resolution and key review decisions
- keep fields short and specific

### Risk: transcript fallback expands over time

If the fallback grows beyond a tiny bounded window, planner behavior will drift toward transcript-driven reasoning.

Mitigation:

- cap the window strictly
- only enable it under explicit conditions
- test for omission outside those conditions

### Risk: duplicated or conflicting sources of truth

If planner memory duplicates workflow status or review state inconsistently, routing becomes harder to reason about.

Mitigation:

- keep memory focused on user intent
- keep control-state facts in existing meta and agent state
- define precedence rules explicitly in code and prompt text

## Recommended Implementation Approach

Use a hybrid design:

- structured planner memory as the durable routing source of truth
- tiny transcript fallback as a narrow disambiguation tool

This is the smallest change that addresses the current failure mode without sacrificing the planner-centric design.
