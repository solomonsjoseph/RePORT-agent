# Agent Paradigm Comparison Slide Design

Date: 2026-04-08
Audience: Researchers, end users, and technically curious reviewers
Goal: Add one new opening slide before the existing orchestrator deck to compare three classic agent paradigms with this app in a balanced way.

## Scope

This design covers one new slide only. It is an addition to the existing slide deck, not a replacement for the current 5-slide structure.

The new slide will appear before the current Slide 1.

## Core Narrative

The slide should answer one framing question early:

What kind of agent system is this app?

The answer should be:

The app uses ideas that overlap with ReAct, Plan-and-Solve, and Reflection where those ideas are useful, but it is not a pure version of any one of them. It is best described as an orchestrator-driven multi-agent system with specialist nodes, safety gates, clarification, and human review.

## Tone and Positioning

- Be accurate rather than promotional
- Use simple language even when naming agent paradigms
- Avoid claiming that the app is completely novel
- Avoid forcing the system into one classic category when the design is clearly mixed
- Make the comparison useful for understanding, not just labeling

## Slide Role

This slide is a framing slide. It should prepare the audience for the rest of the deck by locating the app relative to familiar agent patterns.

It should help the audience understand two things:

1. Which classic ideas appear in the app
2. Why the app should still be described differently overall

## Slide Layout

The slide uses a two-part layout.

### Top Half: Small Paradigm Diagrams

Three compact diagrams placed side by side:

- **ReAct**
  `Think -> Act -> Observe -> Repeat`

- **Plan-and-Solve**
  `Plan -> Execute Step 1 -> Step 2 -> Final Answer`

- **Reflection**
  `Attempt -> Error or Feedback -> Revise -> Retry`

Design intent:
- Keep the diagrams simple and recognizable
- Use them as conceptual summaries, not technical detail
- Make the three paradigms visually comparable at a glance

### Bottom Half: Summary Comparison Table

A compact comparison table below the diagrams.

Recommended columns:
- Paradigm
- What it means
- What this app uses
- How this app differs

Recommended rows:
- ReAct
- Plan-and-Solve
- Reflection
- This app

## Content Direction

### ReAct Row

Meaning:
Step-by-step reasoning and acting with observations between steps.

What this app uses:
- Iterative action selection
- Observation-driven next-step routing
- Tool and execution results feeding subsequent decisions

How this app differs:
- Not one single agent doing all reasoning and acting in one prompt loop
- Uses an orchestrator plus specialist nodes instead of one monolithic ReAct loop

### Plan-and-Solve Row

Meaning:
Plan first, then execute the solution steps.

What this app uses:
- Planning logic at the orchestrator level
- Deliberate next-step selection from workflow context

How this app differs:
- Does not rely on one fixed full plan created at the start
- Re-plans step by step as the workflow state changes

### Reflection Row

Meaning:
Try something, inspect failure or feedback, revise, and retry.

What this app uses:
- Error handling and code revision
- Regeneration after review feedback
- Recovery loops after failed execution

How this app differs:
- Reflection is only one subsystem, not the full architecture
- The app also includes routing, clarification, approval gates, and final review

### This App Row

Meaning:
An orchestrator-driven workflow coordinating multiple specialist nodes.

What this app uses:
- ReAct-style iteration where useful
- Planning where needed for next-step choice
- Reflection where failure recovery is needed

How this app differs:
- Central orchestrator
- Specialist nodes for code generation, execution, QA, clarification, tool use, and review
- Human-in-the-loop approvals
- Safety gates before execution

## Main Takeaway

The slide should end with one clear takeaway line:

`This app borrows useful ideas from classic agent paradigms, but is best described as an orchestrator-driven multi-agent system.`

## Content Constraints

- Do not say the app is “better than” every classic paradigm
- Do not say the app is a pure ReAct, pure Plan-and-Solve, or pure Reflection system
- Do not overload the slide with too much text
- Do not introduce low-level implementation terms such as internal file names or module names
- Keep the bottom table readable in presentation format

## Success Criteria

The slide succeeds if the audience can answer these questions:

1. What are the three classic paradigms being compared?
2. Which parts of those paradigms appear in the app?
3. Why is the app not a pure example of any one of them?
4. What is the best high-level label for the app?
