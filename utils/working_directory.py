from __future__ import annotations

from pathlib import Path


WORKING_DIRECTORY_ENV_VAR = "REPORT_AGENT_WORKING_DIRECTORY"


def normalize_working_directory(raw_path: str | Path) -> Path:
    text = str(raw_path).strip()
    if not text:
        raise ValueError("Working directory is required.")
    return Path(text).expanduser().resolve()


def build_execution_profile(path: str | Path) -> dict[str, str]:
    normalized = normalize_working_directory(path)
    return {
        "working_directory": str(normalized),
        "environment_mode": "project_default",
    }
