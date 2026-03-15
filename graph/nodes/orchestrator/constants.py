"""Constants for orchestrator routing and intent detection."""

# Loop-guard constants
MAX_ACTION_REPEATS = 4
LOOP_GUARD_LOOKBACK = 8
ACTION_LOOKBACK_ACTIONS = 8

# Intent-detection cue lists
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
    "analysis",
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
    "data",
)

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
    "attached data",
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

ANALYSIS_CUES = (
    "perform",
    "conduct",
    "carry out",
    "do a",
    "run a",
    "apply",
    "analyze",
    "analysis",
)
