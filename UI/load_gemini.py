from UI.load_provider import load_provider


def load_gemini(default_api_key, gemini_env_key, default_gemini_model, load_gemini_models_fn):
    return load_provider(
        provider_label="Gemini",
        session_state_key="gemini_api_key",
        input_label="Gemini API Key",
        default_api_key=default_api_key,
        env_api_key=gemini_env_key,
        default_model=default_gemini_model,
        load_models_fn=load_gemini_models_fn,
        model_help="Choose the Gemini model to use.",
    )
