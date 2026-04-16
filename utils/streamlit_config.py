"""Central defaults and option ranges for Streamlit UI configuration."""

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_PROVIDER = "openai"
DEFAULT_API_KEY = ""
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini-2025-04-14"
DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet-20240620"
ANTHROPIC_API_VERSION = "2023-06-01"

PROVIDER_OPTIONS = ("openai", "anthropic", "vllm")
PROVIDER_LABELS = {
    "openai": "OpenAI (ChatGPT)",
    "anthropic": "Anthropic (Claude)",
    "vllm": "Local",
}
EXECUTION_MODE_OPTIONS = ("docker", "trusted_local")
DEFAULT_ENVIRONMENT_MODE = "project_default"
WORKING_DIRECTORY_RUNS_SUBDIR = "runs"

DEFAULT_TEMPERATURE = 0.1
TEMPERATURE_RANGE = (0.0, 1.0)
TEMPERATURE_STEP = 0.05

DEFAULT_TOP_P = 0.9
TOP_P_RANGE = (0.5, 1.0)
TOP_P_STEP = 0.05

DEFAULT_ALLOW_TRUSTED_LOCAL_POLICY_BLOCKED = True

DEFAULT_MAX_AUTO_STEPS = 4
MAX_AUTO_STEPS_RANGE = (1, 8)

DEFAULT_EXECUTION_TIMEOUT_SEC = 60
EXECUTION_TIMEOUT_RANGE = (5, 120)
EXECUTION_TIMEOUT_STEP = 5


def provider_display_label(provider: str) -> str:
    return PROVIDER_LABELS.get(provider, provider)
