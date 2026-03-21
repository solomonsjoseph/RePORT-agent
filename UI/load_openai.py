from UI.load_provider import load_provider


def load_openai(default_api_key, openai_env_key, default_openai_model, load_openai_models_fn):
    return load_provider(
        provider_label="OpenAI",
        session_state_key="openai_api_key",
        input_label="OpenAI API Key",
        default_api_key=default_api_key,
        env_api_key=openai_env_key,
        default_model=default_openai_model,
        load_models_fn=load_openai_models_fn,
        model_help="Choose the OpenAI model to use.",
    )
