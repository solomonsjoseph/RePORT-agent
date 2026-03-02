import streamlit as st

def load_openai(default_api_key, openai_env_key, default_openai_model, load_openai_models_fn):
    if "openai_api_key" not in st.session_state:
        st.session_state.openai_api_key = None

    with st.sidebar.form("openai_api_form", clear_on_submit=False):
        api_key = st.text_input(
            "OpenAI API Key",
            value=default_api_key,
            type="password",
            help="Press Enter or click Submit to confirm."
        )
        submitted = st.form_submit_button("🔑 Submit")

    if submitted:
        st.session_state.openai_api_key = (api_key or openai_env_key or "").strip()

    if not st.session_state.openai_api_key:
        st.sidebar.info("Enter API key and press Enter or click Submit to continue.")
        st.stop()

    effective_api_key = st.session_state.openai_api_key
    if not effective_api_key:
        st.sidebar.warning("OpenAI API key is required.")
        st.info("Please enter a valid OpenAI API key in the sidebar to continue.")
        st.stop()

    try:
        openai_models = load_openai_models_fn(effective_api_key)
        if not openai_models:
            st.sidebar.error("API key was accepted but no models were returned.")
            st.stop()
    except Exception as e:
        st.sidebar.error("Invalid OpenAI API key (or unable to reach OpenAI).")
        st.sidebar.caption(f"Details: {e}")
        st.info("Please enter a valid OpenAI API key in the sidebar to continue.")
        st.stop()

    default_index = openai_models.index(default_openai_model) if default_openai_model in openai_models else 0
    model_name = st.sidebar.selectbox(
        "Model name",
        openai_models,
        index=default_index,
        help="Choose the OpenAI model to use.",
    )
    st.sidebar.markdown("**🧠 Model:**")
    st.sidebar.markdown(f"### {model_name}")
    st.sidebar.caption("OpenAI")

    return effective_api_key, model_name
