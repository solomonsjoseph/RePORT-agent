from __future__ import annotations

from utils.openai_models import (
    is_openai_chat_candidate,
    is_openai_gpt5_family,
    list_supported_openai_chat_models,
    openai_chat_probe_payload,
)


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self):
        return self._payload

    def raise_for_status(self) -> None:
        if not self.ok:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeHTTPClient:
    def __init__(self) -> None:
        self.post_payloads: dict[str, dict] = {}

    def get(self, _url, headers=None, timeout=None):
        assert headers is not None
        assert timeout == 10
        return _FakeResponse(
            200,
            {
                "data": [
                    {"id": "gpt-4o"},
                    {"id": "gpt-5"},
                    {"id": "gpt-5-codex"},
                    {"id": "gpt-4o-audio-preview"},
                ]
            },
        )

    def post(self, _url, headers=None, json=None, timeout=None):
        assert headers is not None
        assert timeout == 15
        assert json is not None
        self.post_payloads[json["model"]] = json
        if json["model"] == "gpt-4o":
            return _FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]})
        if json["model"] == "gpt-5":
            return _FakeResponse(200, {"choices": [{"message": {"content": "ok"}}]})
        return _FakeResponse(400, {"error": {"message": "unsupported"}})


def test_is_openai_gpt5_family_detects_gpt5_models() -> None:
    assert is_openai_gpt5_family("gpt-5")
    assert is_openai_gpt5_family("gpt-5.4-mini")
    assert not is_openai_gpt5_family("gpt-4o")


def test_openai_chat_probe_payload_uses_gpt5_token_field() -> None:
    assert openai_chat_probe_payload("gpt-4o")["max_tokens"] == 8
    assert "max_completion_tokens" not in openai_chat_probe_payload("gpt-4o")

    assert openai_chat_probe_payload("gpt-5")["max_completion_tokens"] == 8
    assert "max_tokens" not in openai_chat_probe_payload("gpt-5")


def test_is_openai_chat_candidate_filters_non_chat_variants() -> None:
    assert is_openai_chat_candidate("gpt-4o")
    assert is_openai_chat_candidate("gpt-5")
    assert not is_openai_chat_candidate("gpt-5-codex")
    assert not is_openai_chat_candidate("gpt-4o-audio-preview")


def test_list_supported_openai_chat_models_probes_and_filters() -> None:
    http_client = _FakeHTTPClient()

    models = list_supported_openai_chat_models("secret", http_client=http_client)

    assert models == ["gpt-4o", "gpt-5"]
    assert http_client.post_payloads["gpt-4o"]["max_tokens"] == 8
    assert http_client.post_payloads["gpt-5"]["max_completion_tokens"] == 8
