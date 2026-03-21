import streamlit as st


def load_provider(
    provider_label,
    session_state_key,
    input_label,
    default_api_key,
    env_api_key,
    default_model,
    load_models_fn,
    model_help,
):
    if session_state_key not in st.session_state:
        st.session_state[session_state_key] = None

    form_key = f"{session_state_key}_form"
    with st.sidebar.form(form_key, clear_on_submit=False):
        api_key = st.text_input(
            input_label,
            value=default_api_key,
            type="password",
            help="Press Enter or click Submit to confirm.",
        )
        submitted = st.form_submit_button("🔑 Submit")

    if submitted:
        st.session_state[session_state_key] = (api_key or env_api_key or "").strip()

    if not st.session_state[session_state_key]:
        st.sidebar.info("Enter API key and press Enter or click Submit to continue.")
        st.stop()

    effective_api_key = st.session_state[session_state_key]
    if not effective_api_key:
        st.sidebar.warning(f"{provider_label} API key is required.")
        st.info(f"Please enter a valid {provider_label} API key in the sidebar to continue.")
        st.stop()

    try:
        models = load_models_fn(effective_api_key)
        if not models:
            st.sidebar.error("API key was accepted but no models were returned.")
            st.stop()
    except Exception as e:
        st.sidebar.error(f"Invalid {provider_label} API key (or unable to reach {provider_label}).")
        st.sidebar.caption(f"Details: {e}")
        st.info(f"Please enter a valid {provider_label} API key in the sidebar to continue.")
        st.stop()

    default_index = models.index(default_model) if default_model in models else 0
    model_name = st.sidebar.selectbox(
        "Model name",
        models,
        index=default_index,
        help=model_help,
    )
    st.sidebar.markdown("**🧠 Model:**")
    st.sidebar.markdown(f"### {model_name}")
    st.sidebar.caption(provider_label)

    return effective_api_key, model_name
