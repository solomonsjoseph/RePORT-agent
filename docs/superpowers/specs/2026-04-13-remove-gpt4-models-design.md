# Remove GPT-4 Series Models Design

**Goal**

Remove all `gpt-4*` OpenAI models from the Streamlit app model selector and switch the default OpenAI model to `gpt-5`.

**Decision**

The app will stop treating `gpt-4*` model IDs as selectable OpenAI chat candidates. The filtering change will live in the shared OpenAI model loader so the Streamlit UI and any future callers consume the same GPT-5-only model list. The default OpenAI model constant will change from `gpt-4o-mini` to `gpt-5`.

**Behavior Changes**

1. `utils/openai_models.py` will only admit `gpt-5*` model IDs as OpenAI chat candidates.
2. Existing unsupported GPT-5 variant filtering will remain in place for models such as audio, realtime, search, transcribe, tts, moderation, image, codex, and pro.
3. `list_supported_openai_chat_models(...)` will continue probing candidate models before exposing them in the UI.
4. `utils/streamlit_config.py` will set `DEFAULT_OPENAI_MODEL = "gpt-5"`.
5. Tests will be updated to prove `gpt-4*` models are excluded and the default moved to `gpt-5`.

**Expected Outcome**

When the provider is OpenAI, the Streamlit sidebar model selector will no longer show any GPT-4-series entries, including `gpt-4o` and `gpt-4o-mini`. If `OPENAI_MODEL` is unset, the app will default to `gpt-5`.
