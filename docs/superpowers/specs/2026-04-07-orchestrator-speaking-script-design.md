# Orchestrator Speaking Script Design

Date: 2026-04-07
Audience: Researchers and end users
Goal: Create a presentation-ready speaking script for the existing 5-slide orchestrator deck, using simple and easy English in a formal but accessible tone.

## Scope

This design covers the spoken script only. It does not change slide order, slide visuals, or slide text. The script must match the existing deck in `slides/orchestrator-demo-2026-04-07.md`.

## Core Narrative

The script should reinforce one consistent message:

The orchestrator makes the app useful for research because it can understand context, ask for clarification when needed, coordinate specialist steps, and keep the researcher involved at key decision points.

## Tone and Delivery Style

- Formal enough for a presentation
- Simple and easy English
- Clear enough for non-technical researchers
- No dense jargon or internal implementation terms
- Confident but not exaggerated

## Script Format

The script should be written as:

- One full paragraph per slide
- Approximately one minute of speaking per slide
- Smooth transitions between slides
- Easy to memorize and deliver aloud

The script should sound like spoken presentation language, not like bullet points read out loud.

## Slide-by-Slide Design

### Slide 1: Orchestrator Capabilities

Purpose:
Introduce the orchestrator as the control layer of the app.

Script focus:
- Explain that the orchestrator decides the next step dynamically
- Show that it connects user requests to the right specialist behavior
- Emphasize flexibility over a fixed pipeline

Delivery note:
This opening should define the orchestrator in plain language and give the audience a simple mental model for the rest of the talk.

### Slide 2: Orchestrator Context Recognition

Purpose:
Explain that the orchestrator makes decisions from the full working context, not only from the latest user message.

Script focus:
- Mention the request, dataset, schema, workflow state, outputs, and feedback
- Explain why context matters for decision quality
- Keep the explanation high-level and audience-friendly

Delivery note:
This slide should make the orchestrator sound thoughtful and grounded, not magical.

### Slide 3: Clarification Before Action

Purpose:
Show that the app does not guess when the request is unclear.

Script focus:
- Explain ambiguity in simple research terms
- Show that the system pauses to ask clarification questions
- Connect clarification to better quality and less wasted work

Delivery note:
This should build trust by showing restraint, not just automation.

### Slide 4: Human Review in the Loop

Purpose:
Show that the system supports researcher control at key stages.

Script focus:
- Explain the review points before execution, after errors, and at the end
- Emphasize that the human can approve or request changes
- Position this as guided autonomy

Delivery note:
This slide should reassure the audience that the system does not hide important actions.

### Slide 5: Demo Proof

Purpose:
Close with a credible summary that the app can complete a real workflow.

Script focus:
- Describe an end-to-end path from question to result
- Mention outputs such as code, text, and figures
- End with a simple statement of practical value for researchers

Delivery note:
The close should feel concrete and convincing, while still modest in its claims.

## Content Constraints

- Do not use internal codebase terms such as file names, module names, or workflow implementation details
- Do not claim full autonomy or guaranteed success
- Do not make the language too technical for a mixed research audience
- Do not make the script so short that it feels like presenter notes rather than a full paragraph

## Success Criteria

The script succeeds if:

1. Each paragraph matches its slide exactly
2. The language is simple enough to speak naturally
3. The audience can understand the orchestrator without technical background
4. The talk feels coherent from slide 1 through slide 5
5. The tone is professional without sounding robotic or overhyped
