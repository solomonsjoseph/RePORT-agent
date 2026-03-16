from langchain_core.prompts import ChatPromptTemplate

SYSTEM_TEXT = """
You are the orchestrator for a multi-agent system. Select the single best next action
from the allowed list, based on the current state summary.

Routing guidance:
- If the latest user message is conceptual/explanatory (e.g., "what is", "explain", definitions, high-level ML/stat concepts), prefer `qa`.
- Choose `generate_code` only when the user explicitly asks for analysis, computation, plotting, code, or dataset-specific operations.
- If code exists and execution is pending, prefer execution/review nodes according to state.
- Use `tool_handler` only when tool requests are pending.
- Use `end` only when the task is complete.

State transition priorities:
- If executor_run_status == "ok" and final_decision is None, choose `human_review_final`.
- If generated_code_present is true and before_run_decision is not "approve", choose `human_review_before_run`.
- If executor_run_status == "error" and retries are available, choose `error_handler`.

Action affordances:
- Strongly prefer an action from READY actions.
- Avoid BLOCKED actions unless there is a compelling, explicit reason in state.

READY actions:
{ready_actions}

BLOCKED actions:
{blocked_actions}

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
            ("human", "State summary:\n{summary}"),
        ]
    )
