from utils import streamlit_config


def test_streamlit_config_exposes_ui_defaults_in_one_place() -> None:
    assert streamlit_config.DEFAULT_PROVIDER == "openai"
    assert streamlit_config.DEFAULT_TEMPERATURE == 0.1
    assert streamlit_config.DEFAULT_TOP_P == 0.9
    assert streamlit_config.DEFAULT_MAX_AUTO_STEPS == 4
    assert streamlit_config.DEFAULT_EXECUTION_TIMEOUT_SEC == 60
    assert streamlit_config.DEFAULT_BASE_URL == "http://localhost:8000/v1"


def test_streamlit_config_exposes_supported_options_in_one_place() -> None:
    assert streamlit_config.PROVIDER_OPTIONS == ("openai", "anthropic", "vllm")
    assert streamlit_config.PROVIDER_LABELS == {
        "openai": "OpenAI (ChatGPT)",
        "anthropic": "Anthropic (Claude)",
        "vllm": "Local",
    }
    assert streamlit_config.EXECUTION_MODE_OPTIONS == ("docker", "trusted_local")
    assert streamlit_config.DEFAULT_ENVIRONMENT_MODE == "project_default"
    assert streamlit_config.WORKING_DIRECTORY_RUNS_SUBDIR == "runs"
    assert streamlit_config.DEFAULT_ALLOW_TRUSTED_LOCAL_POLICY_BLOCKED is True
    assert streamlit_config.MAX_AUTO_STEPS_RANGE == (1, 8)
    assert streamlit_config.EXECUTION_TIMEOUT_RANGE == (5, 120)
    assert streamlit_config.EXECUTION_TIMEOUT_STEP == 5
    assert streamlit_config.TEMPERATURE_RANGE == (0.0, 1.0)
    assert streamlit_config.TEMPERATURE_STEP == 0.05
    assert streamlit_config.TOP_P_RANGE == (0.5, 1.0)
    assert streamlit_config.TOP_P_STEP == 0.05


def test_streamlit_config_exposes_model_env_defaults_in_one_place() -> None:
    assert streamlit_config.DEFAULT_OPENAI_MODEL == "gpt-4.1-mini-2025-04-14"
    assert streamlit_config.DEFAULT_ANTHROPIC_MODEL == "claude-3-5-sonnet-20240620"
    assert streamlit_config.DEFAULT_API_KEY == ""


def test_provider_labels_are_resolved_without_changing_provider_ids() -> None:
    assert streamlit_config.provider_display_label("openai") == "OpenAI (ChatGPT)"
    assert streamlit_config.provider_display_label("anthropic") == "Anthropic (Claude)"
    assert streamlit_config.provider_display_label("vllm") == "Local"
    assert streamlit_config.provider_display_label("custom-provider") == "custom-provider"
