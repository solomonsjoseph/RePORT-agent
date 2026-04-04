from __future__ import annotations

import os
import shutil


EXECUTION_MODE_OPTIONS = ("docker", "trusted_local")


def current_execution_mode() -> str:
    mode = os.getenv("EXECUTION_MODE", "docker").strip().lower()
    if mode in EXECUTION_MODE_OPTIONS:
        return mode
    return "docker"


def docker_available() -> bool:
    return shutil.which("docker") is not None


def apply_execution_mode(mode: str) -> None:
    normalized = mode.strip().lower()
    if normalized not in EXECUTION_MODE_OPTIONS:
        normalized = "docker"

    os.environ["EXECUTION_MODE"] = normalized
    os.environ.pop("ALLOW_TRUSTED_LOCAL_FALLBACK", None)
