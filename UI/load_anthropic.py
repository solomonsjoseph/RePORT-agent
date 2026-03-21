from UI.load_provider import load_provider


def load_anthropic(default_api_key, anthropic_env_key, default_anthropic_model, load_anthropic_models_fn):
    return load_provider(
        provider_label="Anthropic",
        session_state_key="anthropic_api_key",
        input_label="Anthropic API Key",
        default_api_key=default_api_key,
        env_api_key=anthropic_env_key,
        default_model=default_anthropic_model,
        load_models_fn=load_anthropic_models_fn,
        model_help="Choose the Anthropic model to use.",
    )
