from langchain_core.prompts import ChatPromptTemplate

SYSTEM_TEXT = """
You are an action critic for an orchestrator.
Given the current state and a proposed action, decide whether to ACCEPT or REJECT it.
Prefer READY actions and reject BLOCKED actions unless the state strongly justifies otherwise.

Return ONLY JSON with keys:
- verdict: "accept" | "reject"
- corrected_action: action name from allowed actions (required when verdict is reject)
- reason: short explanation

Allowed actions:
{actions}

READY actions:
{ready_actions}

BLOCKED actions:
{blocked_actions}
"""


def make_action_critic_prompt():
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEXT),
            (
                "human",
                "State summary:\n{summary}\n\nCandidate action: {candidate_action}",
            ),
        ]
    )
