from pathlib import Path

import pytest

from utils import streamlit_config
from utils.working_directory import (
    WORKING_DIRECTORY_ENV_VAR,
    build_execution_profile,
    normalize_working_directory,
)


def test_streamlit_config_exposes_working_directory_defaults() -> None:
    assert WORKING_DIRECTORY_ENV_VAR == "REPORT_AGENT_WORKING_DIRECTORY"
    assert streamlit_config.DEFAULT_ENVIRONMENT_MODE == "project_default"
    assert streamlit_config.WORKING_DIRECTORY_RUNS_SUBDIR == "runs"


def test_normalize_working_directory_returns_resolved_absolute_path(tmp_path: Path) -> None:
    nested = tmp_path / "workspace"
    nested.mkdir()

    resolved = normalize_working_directory(str(nested))

    assert resolved == nested.resolve()


def test_normalize_working_directory_rejects_blank_values() -> None:
    with pytest.raises(ValueError, match="Working directory is required"):
        normalize_working_directory("   ")


def test_build_execution_profile_uses_project_default_mode(tmp_path: Path) -> None:
    profile = build_execution_profile(tmp_path)

    assert profile == {
        "working_directory": str(tmp_path.resolve()),
        "environment_mode": "project_default",
    }
