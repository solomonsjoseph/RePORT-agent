from __future__ import annotations

import ast
import io
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from utils.execution_mode import (
    allow_trusted_local_policy_blocked,
    current_execution_mode,
)


DISALLOWED_IMPORT_ROOTS = {
    "socket",
    "subprocess",
}

DISALLOWED_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "open",
    "input",
}

DISALLOWED_ATTR_CALLS = {
    ("os", "system"),
    ("os", "popen"),
    ("os", "spawnl"),
    ("os", "spawnle"),
    ("os", "spawnlp"),
    ("os", "spawnlpe"),
    ("os", "spawnv"),
    ("os", "spawnve"),
    ("os", "spawnvp"),
    ("os", "spawnvpe"),
    ("socket", "socket"),
    ("subprocess", "Popen"),
    ("subprocess", "call"),
    ("subprocess", "check_call"),
    ("subprocess", "check_output"),
    ("subprocess", "run"),
}

DISALLOWED_FS_FUNCTION_CALLS = {
    ("os", "remove"),
    ("os", "unlink"),
    ("os", "rename"),
    ("os", "replace"),
    ("os", "mkdir"),
    ("os", "makedirs"),
    ("os", "rmdir"),
    ("os", "removedirs"),
    ("shutil", "rmtree"),
    ("shutil", "move"),
    ("shutil", "copy"),
    ("shutil", "copy2"),
    ("shutil", "copyfile"),
    ("shutil", "copytree"),
}

DISALLOWED_FS_METHOD_CALLS = {
    "mkdir",
    "rmdir",
    "rename",
    "replace",
    "unlink",
    "write_bytes",
    "write_text",
    "touch",
    "symlink_to",
    "hardlink_to",
    "savefig",
    "to_csv",
    "to_excel",
    "to_feather",
    "to_hdf",
    "to_json",
    "to_parquet",
    "to_pickle",
    "to_sql",
}

ERROR_CATEGORY_RETRYABLE_CODE = "retryable_code"
ERROR_CATEGORY_POLICY_BLOCKED = "policy_blocked"
ERROR_CATEGORY_UNSUPPORTED_RUNTIME = "unsupported_runtime"
ERROR_CATEGORY_INFRASTRUCTURE = "infrastructure"
ERROR_CATEGORY_TIMEOUT = "timeout"

_MODULE_NOT_FOUND_RE = re.compile(r"No module named ['\"]([^'\"]+)['\"]")


def _build_exec_globals(df: pd.DataFrame) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    import numpy as np
    from lifelines import CoxPHFitter, KaplanMeierFitter
    from scipy.stats import chi2_contingency, fisher_exact

    return {
        "pd": pd,
        "np": np,
        "KaplanMeierFitter": KaplanMeierFitter,
        "CoxPHFitter": CoxPHFitter,
        "df": df,
        "chi2_contingency": chi2_contingency,
        "fisher_exact": fisher_exact,
        "plt": plt,
        "__name__": "__main__",
    }


def _execute_user_code(code: str, df: pd.DataFrame):
    """Execute user code quietly and return structured errors."""

    global_env = _build_exec_globals(df)
    local_env: dict[str, Any] = {}

    stdout_buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout_buf

    figure_png = b""
    plt = global_env["plt"]

    try:
        exec(code, global_env, local_env)
        result = local_env.get("result")
        error = None

        fig = plt.gcf()
        if fig and fig.axes:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            figure_png = buf.getvalue()

        plt.close("all")

    except Exception as e:
        result = None
        error = _normalize_runner_error(
            {
                "type": "PythonRuntimeError",
                "message": str(e),
            }
        )

    finally:
        sys.stdout = old_stdout

    stdout = stdout_buf.getvalue()

    return result, stdout, figure_png, error


def _attribute_name(node: ast.AST) -> tuple[str, ...]:
    parts: list[str] = []
    current: ast.AST | None = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return tuple(reversed(parts))


def _format_fs_call_name(
    node: ast.Call,
    name_parts: tuple[str, ...],
    fallback_attr: str | None,
) -> str:
    if isinstance(node.func, ast.Attribute):
        value = node.func.value
        if isinstance(value, ast.Call):
            callee = _attribute_name(value.func)
            if callee and callee[-1] == "Path":
                return f"Path.{node.func.attr}"
        if isinstance(value, ast.Name) and value.id == "Path":
            return f"Path.{node.func.attr}"
    if name_parts:
        if len(name_parts) >= 2 and name_parts[0] in {"os", "shutil", "Path"}:
            return ".".join(name_parts[:2])
        if len(name_parts) == 1:
            return name_parts[0]
    return fallback_attr or "filesystem call"


def _collect_import_aliases(tree: ast.AST) -> dict[str, tuple[str, ...]]:
    aliases: dict[str, tuple[str, ...]] = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                aliases[alias.asname or root] = tuple(alias.name.split("."))
        elif isinstance(node, ast.ImportFrom) and node.module:
            module_parts = tuple(node.module.split("."))
            for alias in node.names:
                aliases[alias.asname or alias.name] = module_parts + (alias.name,)

    return aliases


def _policy_error(message: str):
    return None, "", b"", _make_error(
        "PolicyBlockedError",
        message,
        ERROR_CATEGORY_POLICY_BLOCKED,
    )


def _make_error(error_type: str, message: str, category: str) -> dict[str, str]:
    return {
        "type": error_type,
        "message": message,
        "category": category,
    }


def _allow_trusted_local_fs_mutations() -> bool:
    return (
        current_execution_mode() == "trusted_local"
        and allow_trusted_local_policy_blocked()
    )


def _preflight_policy_check(code: str) -> dict[str, str] | None:
    try:
        tree = ast.parse((code or "").strip())
    except SyntaxError as exc:
        return {
            "type": "PolicyBlockedError",
            "message": f"Code failed policy parsing: {exc.msg}",
            "category": ERROR_CATEGORY_POLICY_BLOCKED,
        }

    import_aliases = _collect_import_aliases(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in DISALLOWED_IMPORT_ROOTS:
                    return {
                        "type": "PolicyBlockedError",
                        "message": f"Disallowed import: {root}",
                        "category": ERROR_CATEGORY_POLICY_BLOCKED,
                    }
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root = node.module.split(".", 1)[0]
                if root in DISALLOWED_IMPORT_ROOTS:
                    return {
                        "type": "PolicyBlockedError",
                        "message": f"Disallowed import: {root}",
                        "category": ERROR_CATEGORY_POLICY_BLOCKED,
                    }
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in DISALLOWED_CALLS:
                return {
                    "type": "PolicyBlockedError",
                    "message": f"Disallowed call: {node.func.id}",
                    "category": ERROR_CATEGORY_POLICY_BLOCKED,
                }

            if isinstance(node.func, ast.Name):
                imported_name = import_aliases.get(node.func.id)
                if imported_name in DISALLOWED_FS_FUNCTION_CALLS and not _allow_trusted_local_fs_mutations():
                    return {
                        "type": "PolicyBlockedError",
                        "message": f"Disallowed filesystem mutation: {'.'.join(imported_name[:2])}",
                        "category": ERROR_CATEGORY_POLICY_BLOCKED,
                    }

            func_name = _attribute_name(node.func)
            if len(func_name) >= 2 and tuple(func_name[:2]) in DISALLOWED_ATTR_CALLS:
                return {
                    "type": "PolicyBlockedError",
                    "message": f"Disallowed call: {'.'.join(func_name[:2])}",
                    "category": ERROR_CATEGORY_POLICY_BLOCKED,
                }
            if (
                len(func_name) >= 2
                and tuple(func_name[:2]) in DISALLOWED_FS_FUNCTION_CALLS
                and not _allow_trusted_local_fs_mutations()
            ):
                return {
                    "type": "PolicyBlockedError",
                    "message": f"Disallowed filesystem mutation: {'.'.join(func_name[:2])}",
                    "category": ERROR_CATEGORY_POLICY_BLOCKED,
                }
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in DISALLOWED_FS_METHOD_CALLS
                and not _allow_trusted_local_fs_mutations()
            ):
                return {
                    "type": "PolicyBlockedError",
                    "message": (
                        "Disallowed filesystem mutation: "
                        f"{_format_fs_call_name(node, func_name, node.func.attr)}"
                    ),
                    "category": ERROR_CATEGORY_POLICY_BLOCKED,
                }
    return None


def _normalize_runner_error(error: dict[str, Any] | None) -> dict[str, str] | None:
    if not error:
        return None

    error_type = str(error.get("type") or "SandboxExecutionError")
    message = str(error.get("message") or "")
    category = str(error.get("category") or "")

    if error_type == "PythonRuntimeError":
        match = _MODULE_NOT_FOUND_RE.search(message)
        if match:
            return _make_error(
                "DependencyNotAvailableError",
                f"Sandbox image does not include package: {match.group(1)}",
                ERROR_CATEGORY_UNSUPPORTED_RUNTIME,
            )
        return _make_error(error_type, message, ERROR_CATEGORY_RETRYABLE_CODE)

    if error_type == "PolicyBlockedError":
        return _make_error(error_type, message, ERROR_CATEGORY_POLICY_BLOCKED)
    if error_type == "TimeoutError":
        return _make_error(error_type, message, ERROR_CATEGORY_TIMEOUT)
    if error_type == "DependencyNotAvailableError":
        return _make_error(error_type, message, ERROR_CATEGORY_UNSUPPORTED_RUNTIME)
    if error_type == "SandboxExecutionError":
        return _make_error(error_type, message, ERROR_CATEGORY_INFRASTRUCTURE)
    if category:
        return _make_error(error_type, message, category)
    return _make_error(error_type, message, ERROR_CATEGORY_INFRASTRUCTURE)


def _safe_json_value(value: Any) -> Any:
    try:
        json.dumps(value)
    except TypeError:
        return None
    return value


def _docker_command(input_dir: Path, output_dir: Path) -> list[str]:
    sandbox_image = os.getenv("SANDBOX_IMAGE", "report-agent-sandbox:latest")
    memory_mb = os.getenv("SANDBOX_MEMORY_MB", "512")
    cpu_limit = os.getenv("SANDBOX_CPU_LIMIT", "1.0")
    return [
        "docker",
        "run",
        "--rm",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit",
        "64",
        "--memory",
        f"{memory_mb}m",
        "--cpus",
        cpu_limit,
        "--user",
        "65534:65534",
        "--mount",
        f"type=bind,src={input_dir},dst=/sandbox/input,readonly",
        "--mount",
        f"type=bind,src={output_dir},dst=/sandbox/output",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "-e",
        "MPLCONFIGDIR=/tmp/matplotlib",
        sandbox_image,
        "python",
        "/opt/report_agent/runner.py",
        "--input-dir",
        "/sandbox/input",
        "--output-dir",
        "/sandbox/output",
    ]


def _run_docker_sandbox(code: str, df: pd.DataFrame):
    timeout_seconds = float(os.getenv("EXECUTION_TIMEOUT_SEC", "20"))
    with tempfile.TemporaryDirectory(prefix="report-agent-exec-") as tmpdir:
        tmp_path = Path(tmpdir)
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        (input_dir / "code.py").write_text(code, encoding="utf-8")
        df.to_csv(input_dir / "dataset.csv", index=False)

        command = _docker_command(input_dir, output_dir)
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None, "", b"", _make_error(
                "TimeoutError",
                f"Execution exceeded {timeout_seconds:.0f}s sandbox timeout.",
                ERROR_CATEGORY_TIMEOUT,
            )
        except FileNotFoundError:
            return None, "", b"", _make_error(
                "SandboxExecutionError",
                "Docker is not installed or not available on PATH.",
                ERROR_CATEGORY_INFRASTRUCTURE,
            )

        if completed.returncode != 0:
            stderr = (completed.stderr or "").strip()
            return None, "", b"", _make_error(
                "SandboxExecutionError",
                stderr or "Sandboxed execution failed.",
                ERROR_CATEGORY_INFRASTRUCTURE,
            )

        result_path = output_dir / "result.json"
        if not result_path.exists():
            return None, "", b"", _make_error(
                "SandboxExecutionError",
                "Sandboxed execution failed to return result.json.",
                ERROR_CATEGORY_INFRASTRUCTURE,
            )

        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None, "", b"", _make_error(
                "SandboxExecutionError",
                "Sandboxed execution returned malformed result.json.",
                ERROR_CATEGORY_INFRASTRUCTURE,
            )

        stdout_path = output_dir / "stdout.txt"
        stdout = stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else payload.get("text", "")

        figure_path = output_dir / "figure.png"
        figure_png = figure_path.read_bytes() if figure_path.exists() else b""

        error = _normalize_runner_error(payload.get("error"))
        result = payload.get("result")
        if payload.get("status") != "ok" and not error:
            error = _make_error(
                "SandboxExecutionError",
                "Sandboxed execution returned an unknown failure.",
                ERROR_CATEGORY_INFRASTRUCTURE,
            )
        return result, stdout, figure_png, error


def run_python_user(code: str, df: pd.DataFrame):
    """Execute user code and return result, stdout, figure, and structured error."""

    policy_error = _preflight_policy_check(code)
    if policy_error is not None:
        return _policy_error(policy_error["message"])

    execution_mode = current_execution_mode()
    if execution_mode in {"inline", "trusted_local"}:
        return _execute_user_code(code, df)
    if execution_mode == "docker":
        return _run_docker_sandbox(code, df)
    if execution_mode == "subprocess":
        return _run_docker_sandbox(code, df)

    return None, "", b"", _make_error(
        "SandboxExecutionError",
        f"Unsupported execution mode: {execution_mode}",
        ERROR_CATEGORY_INFRASTRUCTURE,
    )
