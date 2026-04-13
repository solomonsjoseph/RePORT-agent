from __future__ import annotations

import os
import shutil


EXECUTION_MODE_OPTIONS = ("docker", "trusted_local")
DEFAULT_EXECUTION_MODE = "trusted_local"
TRUSTED_LOCAL_POLICY_BLOCK_ENV = "ALLOW_TRUSTED_LOCAL_POLICY_BLOCKED"


def current_execution_mode() -> str:
    mode = os.getenv("EXECUTION_MODE", DEFAULT_EXECUTION_MODE).strip().lower()
    if mode in EXECUTION_MODE_OPTIONS:
        return mode
    return DEFAULT_EXECUTION_MODE


def allow_trusted_local_policy_blocked() -> bool:
    raw = os.getenv(TRUSTED_LOCAL_POLICY_BLOCK_ENV, "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def docker_available() -> bool:
    return shutil.which("docker") is not None


def apply_execution_mode(mode: str) -> None:
    normalized = mode.strip().lower()
    if normalized not in EXECUTION_MODE_OPTIONS:
        normalized = "docker"

    os.environ["EXECUTION_MODE"] = normalized
    os.environ.pop("ALLOW_TRUSTED_LOCAL_FALLBACK", None)
