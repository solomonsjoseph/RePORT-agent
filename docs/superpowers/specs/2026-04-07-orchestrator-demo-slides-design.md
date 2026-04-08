# Orchestrator Demo Slides Design

Date: 2026-04-07
Audience: Researchers and end users
Goal: Give a short demo-oriented introduction to the orchestrator and highlight its capabilities with enough system detail to feel credible without becoming overly technical.

## Scope

This design covers a 5-slide presentation deck. The deck is not a full product overview. It is focused on the orchestrator, clarification behavior, and human review checkpoints.

## Core Narrative

The presentation should make one argument:

The app works because the orchestrator turns an open-ended research request into a controlled multi-step workflow, asks for clarification when needed, and keeps the researcher in control through review checkpoints.

## Tone and Level of Detail

- Use plain language first, then add a small amount of system detail.
- Avoid deep implementation details such as specific class names or internal files.
- Keep the message credible by naming the major specialist roles and review stages.
- Frame the app as a research assistant, not as a fully autonomous replacement for researcher judgment.

## Slide Structure

### Slide 1: Orchestrator Capabilities

Purpose:
Introduce the orchestrator as the control layer of the app.

Key message:
The orchestrator decides what should happen next instead of forcing every request through a rigid pipeline.

Content:
- Interprets the user request
- Selects the next specialist node
- Tracks workflow progress
- Responds to observations, errors, and feedback
- Keeps the analysis moving toward a result

Suggested visual:
A central "Orchestrator" box with surrounding specialist roles:
- Code Generator
- Executor
- Error Handler
- Q&A
- Tool Handler
- Human Review

Speaker direction:
Emphasize flexibility. The main idea is that different research questions do not always need the same next step, so the orchestrator makes that decision dynamically.

### Slide 2: Orchestrator Context Recognition

Purpose:
Show that the orchestrator acts on context, not just the latest user message.

Key message:
Before choosing the next step, the orchestrator reads the current working environment of the analysis.

Content:
- User question or task
- Uploaded dataset and schema
- Current workflow state
- Recent outputs from previous nodes
- Execution status
- Error messages or human feedback

Suggested visual:
A layered input diagram flowing into the orchestrator:
"Request", "Data", "Schema", "State", "Outputs", "Feedback" -> "Orchestrator decision"

Speaker direction:
Explain that this is why the app can make reasonable next-step decisions instead of repeating the same behavior every turn.

### Slide 3: Clarification Before Action

Purpose:
Show that the app does not guess when the research intent is unclear.

Key message:
When the request is ambiguous, the orchestrator pauses the workflow and asks a clarification question before analysis begins.

Content:
- Detects unclear or underspecified research requests
- Requests clarification from the user
- Prevents wasted analysis and misleading outputs
- Resumes workflow once the objective is clear

Suggested visual:
A simple branch:
"Ambiguous request?" -> "Yes" -> "Ask clarification"
"Ambiguous request?" -> "No" -> "Proceed with analysis"

Speaker direction:
This slide should communicate trust and research usefulness. The value is not just automation; it is avoiding confident but irrelevant work.

### Slide 4: Human Review in the Loop

Purpose:
Show that researchers keep control at critical stages.

Key message:
The orchestrator coordinates the workflow, but the human decides whether execution should continue, be corrected, or be accepted.

Content:
- Review before execution
- Review after an error and proposed fix
- Final review of code, outputs, and figures
- Option to approve or request regeneration

Suggested visual:
A three-checkpoint timeline:
"Before Run" -> "After Error" -> "Final Review"

Speaker direction:
Present this as guided autonomy. The app speeds up analysis, but sensitive decisions remain visible and reviewable.

### Slide 5: Demo Proof

Purpose:
Close with evidence that the app can complete a realistic research task.

Key message:
The orchestrator can coordinate an end-to-end analysis loop and return inspectable results.

Content:
- Research question enters the system
- Orchestrator routes work across specialist nodes
- Approved code is executed
- Result includes generated code, textual output, and figures when applicable

Suggested visual:
A compact end-to-end workflow strip plus one example output panel.

Speaker direction:
This slide should feel concrete. Use one realistic example question and show that the system produces outputs a researcher can inspect and refine.

## Recommended Demo Positioning

The presenter should describe the app as:

"An orchestrator-driven research assistant that can interpret a request, decide the next action, ask for clarification when needed, and keep the researcher in control through review checkpoints."

## Content Constraints

- Do not oversell full autonomy.
- Do not claim the system always succeeds without correction.
- Do not focus on model-provider configuration or low-level infrastructure unless asked.
- Keep each slide to one main idea and three to five supporting bullets.

## Success Criteria

The deck succeeds if the audience can answer these questions after the presentation:

1. What is the orchestrator?
2. What information does it use to make decisions?
3. How does the system handle unclear requests?
4. Where does the human review the workflow?
5. What evidence shows the app can complete real work?
