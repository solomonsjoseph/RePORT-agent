from __future__ import annotations

from utils import execution_mode


def test_current_execution_mode_defaults_to_docker(monkeypatch) -> None:
    monkeypatch.delenv("EXECUTION_MODE", raising=False)

    assert execution_mode.current_execution_mode() == "docker"


def test_current_execution_mode_normalizes_invalid_value(monkeypatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "weird")

    assert execution_mode.current_execution_mode() == "docker"


def test_apply_execution_mode_sets_mode(monkeypatch) -> None:
    monkeypatch.delenv("EXECUTION_MODE", raising=False)

    execution_mode.apply_execution_mode("trusted_local")

    assert execution_mode.current_execution_mode() == "trusted_local"


def test_apply_execution_mode_clears_stale_fallback_env(monkeypatch) -> None:
    monkeypatch.setenv("ALLOW_TRUSTED_LOCAL_FALLBACK", "1")

    execution_mode.apply_execution_mode("docker")

    assert execution_mode.current_execution_mode() == "docker"
    assert "ALLOW_TRUSTED_LOCAL_FALLBACK" not in execution_mode.os.environ
