"""Central tuning constants for workflow control and prompt context windows.

Adjust these values when changing retry behavior or how much recent context a
node receives. Counts named ``*_TURNS`` use the turn grouping implemented by
``utils.message_window.window_messages`` unless explicitly noted otherwise.
"""

# Retry loop
MAX_ERROR_ITERATIONS = 5

# Conversation windows
PLANNER_RECENT_TURNS = 3
TOOL_ROUTER_RECENT_TURNS = 3
QA_RECENT_TURNS = 10
DB_RAG_RECENT_TURNS = 10
CODEGEN_RECENT_TURNS = 10
ERROR_HANDLER_RECENT_TURNS = 10
CLARIFICATION_RECENT_TURNS = 3
CLARIFICATION_WITH_PENDING_RECENT_TURNS = 5

# Planner diagnostic context limits
WORKFLOW_TRACE_TAIL = 8
RECENT_OBSERVATIONS_LIMIT = 6
PLANNER_DECISION_TRACE_LIMIT = 5
PLANNER_RAW_RESPONSE_PREVIEW_CHARS = 300
