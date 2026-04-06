from langchain_core.prompts import ChatPromptTemplate

SYSTEM_TEXT = """
You are the orchestrator for a multi-agent system. Select the single best next action
from the allowed list based on the full workflow environment.

Routing guidance:
- Use the environment summary, recent observations, and decision trace to choose the next action.
- Treat action masks as hard constraints.
- Prefer semantically appropriate actions instead of inferring a rigid pipeline from state fields alone.
- If stagnation_count is rising, avoid repeating the same action unless the new context clearly changes the outcome.

Node capabilities:
{node_capabilities}

Return ONLY a JSON object with:
- "thought": short rationale tied to latest user intent + state
- "action": chosen action name from the allowed list
- "ranked_actions": optional list of up to 3 candidate actions in order of preference

Allowed actions:
{actions}
"""


def make_planner_prompt():
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEXT),
            (
                "human",
                "Environment summary:\n{summary}\n\n"
                "READY actions:\n{ready_actions}\n\n"
                "BLOCKED actions:\n{blocked_actions}",
            ),
        ]
    )
