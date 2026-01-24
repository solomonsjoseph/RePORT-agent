from langchain_core.prompts import ChatPromptTemplate

SYSTEM_TEXT = """
You are the orchestrator for a multi-agent system. Select the single best next action
from the allowed list, based on the current state summary. Return ONLY the action name.

Allowed actions:
{actions}
"""

def make_planner_prompt():
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEXT),
            ("human", "State summary:\n{summary}"),
        ]
    )
