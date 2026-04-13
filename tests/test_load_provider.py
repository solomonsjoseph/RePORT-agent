from __future__ import annotations

from types import SimpleNamespace

import pytest

from UI import load_provider as load_provider_module
from utils.openai_models import OpenAIModelProbeError


class _FakeForm:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeSidebar:
    def __init__(self):
        self.info_messages = []
        self.warning_messages = []
        self.error_messages = []
        self.caption_messages = []
        self.markdown_messages = []
        self.selectbox_calls = []
        self._submitted = False

    def form(self, *_args, **_kwargs):
        return _FakeForm()

    def text_input(self, *_args, **_kwargs):
        return ""

    def form_submit_button(self, *_args, **_kwargs):
        return self._submitted

    def info(self, message):
        self.info_messages.append(message)

    def warning(self, message):
        self.warning_messages.append(message)

    def error(self, message):
        self.error_messages.append(message)

    def caption(self, message):
        self.caption_messages.append(message)

    def markdown(self, message):
        self.markdown_messages.append(message)

    def selectbox(self, label, options, index=0, help=None):
        self.selectbox_calls.append((label, options, index, help))
        return options[index]


class _FakeStreamlit:
    def __init__(self):
        self.session_state = {}
        self.sidebar = _FakeSidebar()
        self.info_messages = []
        self.runtime = SimpleNamespace(exists=lambda: False)

    def info(self, message):
        self.info_messages.append(message)

    def text_input(self, *args, **kwargs):
        return self.sidebar.text_input(*args, **kwargs)

    def form_submit_button(self, *args, **kwargs):
        return self.sidebar.form_submit_button(*args, **kwargs)

    def stop(self):
        raise RuntimeError("st.stop called")


def test_load_provider_uses_env_key_without_submit(monkeypatch):
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(load_provider_module, "st", fake_st)

    model_calls = []

    def load_models_fn(api_key):
        model_calls.append(api_key)
        return ["gpt-4o-mini", "gpt-4.1"]

    api_key, model_name = load_provider_module.load_provider(
        provider_label="OpenAI",
        session_state_key="openai_api_key",
        input_label="OpenAI API Key",
        default_api_key="",
        env_api_key="env-key",
        default_model="gpt-4o-mini",
        load_models_fn=load_models_fn,
        model_help="Choose model.",
    )

    assert api_key == "env-key"
    assert model_name == "gpt-4o-mini"
    assert fake_st.session_state["openai_api_key"] == "env-key"
    assert model_calls == ["env-key"]
    assert fake_st.sidebar.info_messages == []


def test_load_provider_stops_when_no_key_available(monkeypatch):
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(load_provider_module, "st", fake_st)

    with pytest.raises(RuntimeError, match="Streamlit input is required here"):
        load_provider_module.load_provider(
            provider_label="OpenAI",
            session_state_key="openai_api_key",
            input_label="OpenAI API Key",
            default_api_key="",
            env_api_key="",
            default_model="gpt-4o-mini",
            load_models_fn=lambda _api_key: ["gpt-4o-mini"],
            model_help="Choose model.",
        )

    assert fake_st.sidebar.info_messages == [
        "Enter API key and press Enter or click Submit to continue."
    ]


def test_load_provider_surfaces_loader_error_details(monkeypatch):
    fake_st = _FakeStreamlit()
    fake_st.session_state["openai_api_key"] = "env-key"
    monkeypatch.setattr(load_provider_module, "st", fake_st)

    def load_models_fn(_api_key):
        raise OpenAIModelProbeError(
            "No supported OpenAI chat models passed the availability probe: gpt-5: Quota exceeded"
        )

    with pytest.raises(RuntimeError, match="Failed to load OpenAI models"):
        load_provider_module.load_provider(
            provider_label="OpenAI",
            session_state_key="openai_api_key",
            input_label="OpenAI API Key",
            default_api_key="",
            env_api_key="env-key",
            default_model="gpt-4o-mini",
            load_models_fn=load_models_fn,
            model_help="Choose model.",
        )

    assert fake_st.sidebar.error_messages == [
        "No supported OpenAI chat models passed the availability probe: gpt-5: Quota exceeded"
    ]
    assert fake_st.sidebar.caption_messages == []
