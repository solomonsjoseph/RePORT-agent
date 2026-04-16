from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd


def _load_execution_module():
    lifelines_mod = ModuleType("lifelines")
    lifelines_mod.KaplanMeierFitter = object
    lifelines_mod.CoxPHFitter = object
    sys.modules["lifelines"] = lifelines_mod
    sys.modules.pop("tools.execution", None)
    return importlib.import_module("tools.execution")


class _TmpDirCtx:
    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> str:
        return str(self.path)

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_run_python_user_blocks_disallowed_imports() -> None:
    execution = _load_execution_module()
    result, stdout, figure_png, error = execution.run_python_user(
        "import subprocess\nprint('x')",
        pd.DataFrame({"a": [1]}),
    )

    assert result is None
    assert stdout == ""
    assert figure_png == b""
    assert error == {
        "category": "policy_blocked",
        "type": "PolicyBlockedError",
        "message": "Disallowed import: subprocess",
    }


def test_run_python_user_allows_pathlib_usage_in_trusted_local_mode(monkeypatch) -> None:
    execution = _load_execution_module()

    monkeypatch.setenv("EXECUTION_MODE", "trusted_local")

    result, stdout, figure_png, error = execution.run_python_user(
        "from pathlib import Path\nresult = Path('data').name",
        pd.DataFrame({"a": [1]}),
    )

    assert result == "data"
    assert stdout == ""
    assert figure_png == b""
    assert error is None


def test_run_python_user_allows_shutil_import_without_blocking(monkeypatch) -> None:
    execution = _load_execution_module()

    monkeypatch.setenv("EXECUTION_MODE", "trusted_local")

    result, stdout, figure_png, error = execution.run_python_user(
        "import shutil\nresult = shutil.get_unpack_formats()[0][0]",
        pd.DataFrame({"a": [1]}),
    )

    assert isinstance(result, str)
    assert stdout == ""
    assert figure_png == b""
    assert error is None


def test_run_python_user_blocks_pathlib_file_writes(monkeypatch) -> None:
    execution = _load_execution_module()

    monkeypatch.setenv("EXECUTION_MODE", "trusted_local")
    monkeypatch.setenv("ALLOW_TRUSTED_LOCAL_POLICY_BLOCKED", "0")

    result, stdout, figure_png, error = execution.run_python_user(
        "from pathlib import Path\nPath('x.txt').write_text('data')",
        pd.DataFrame({"a": [1]}),
    )

    assert result is None
    assert stdout == ""
    assert figure_png == b""
    assert error == {
        "category": "policy_blocked",
        "type": "PolicyBlockedError",
        "message": "Disallowed filesystem mutation: Path.write_text",
    }


def test_run_python_user_blocks_pathlib_file_deletes(monkeypatch) -> None:
    execution = _load_execution_module()

    monkeypatch.setenv("EXECUTION_MODE", "trusted_local")
    monkeypatch.setenv("ALLOW_TRUSTED_LOCAL_POLICY_BLOCKED", "0")

    result, stdout, figure_png, error = execution.run_python_user(
        "from pathlib import Path\nPath('x.txt').unlink()",
        pd.DataFrame({"a": [1]}),
    )

    assert result is None
    assert stdout == ""
    assert figure_png == b""
    assert error == {
        "category": "policy_blocked",
        "type": "PolicyBlockedError",
        "message": "Disallowed filesystem mutation: Path.unlink",
    }


def test_run_python_user_blocks_os_file_delete_calls(monkeypatch) -> None:
    execution = _load_execution_module()

    monkeypatch.setenv("EXECUTION_MODE", "trusted_local")
    monkeypatch.setenv("ALLOW_TRUSTED_LOCAL_POLICY_BLOCKED", "0")

    result, stdout, figure_png, error = execution.run_python_user(
        "import os\nos.remove('x.txt')",
        pd.DataFrame({"a": [1]}),
    )

    assert result is None
    assert stdout == ""
    assert figure_png == b""
    assert error == {
        "category": "policy_blocked",
        "type": "PolicyBlockedError",
        "message": "Disallowed filesystem mutation: os.remove",
    }


def test_run_python_user_blocks_shutil_tree_delete_calls(monkeypatch) -> None:
    execution = _load_execution_module()

    monkeypatch.setenv("EXECUTION_MODE", "trusted_local")
    monkeypatch.setenv("ALLOW_TRUSTED_LOCAL_POLICY_BLOCKED", "0")

    result, stdout, figure_png, error = execution.run_python_user(
        "import shutil\nshutil.rmtree('tmpdir')",
        pd.DataFrame({"a": [1]}),
    )

    assert result is None
    assert stdout == ""
    assert figure_png == b""
    assert error == {
        "category": "policy_blocked",
        "type": "PolicyBlockedError",
        "message": "Disallowed filesystem mutation: shutil.rmtree",
    }


def test_run_python_user_blocks_dataframe_export_calls(monkeypatch) -> None:
    execution = _load_execution_module()

    monkeypatch.setenv("EXECUTION_MODE", "trusted_local")
    monkeypatch.setenv("ALLOW_TRUSTED_LOCAL_POLICY_BLOCKED", "0")

    result, stdout, figure_png, error = execution.run_python_user(
        "df.to_csv('out.csv', index=False)",
        pd.DataFrame({"a": [1]}),
    )

    assert result is None
    assert stdout == ""
    assert figure_png == b""
    assert error == {
        "category": "policy_blocked",
        "type": "PolicyBlockedError",
        "message": "Disallowed filesystem mutation: to_csv",
    }


def test_run_python_user_blocks_plot_file_exports(monkeypatch) -> None:
    execution = _load_execution_module()

    monkeypatch.setenv("EXECUTION_MODE", "trusted_local")
    monkeypatch.setenv("ALLOW_TRUSTED_LOCAL_POLICY_BLOCKED", "0")

    result, stdout, figure_png, error = execution.run_python_user(
        "plt.plot([1, 2], [3, 4])\nplt.savefig('plot.png')",
        pd.DataFrame({"a": [1]}),
    )

    assert result is None
    assert stdout == ""
    assert figure_png == b""
    assert error == {
        "category": "policy_blocked",
        "type": "PolicyBlockedError",
        "message": "Disallowed filesystem mutation: savefig",
    }


def test_run_python_user_allows_pathlib_file_writes_when_trusted_local_toggle_enabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    execution = _load_execution_module()

    monkeypatch.setenv("EXECUTION_MODE", "trusted_local")
    monkeypatch.setenv("ALLOW_TRUSTED_LOCAL_POLICY_BLOCKED", "1")
    monkeypatch.chdir(tmp_path)

    result, stdout, figure_png, error = execution.run_python_user(
        "from pathlib import Path\nPath('x.txt').write_text('data')\nresult = Path('x.txt').read_text()",
        pd.DataFrame({"a": [1]}),
    )

    assert result == "data"
    assert stdout == ""
    assert figure_png == b""
    assert error is None
    assert (tmp_path / "x.txt").read_text(encoding="utf-8") == "data"


def test_run_python_user_reads_structured_docker_outputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    execution = _load_execution_module()
    written: dict[str, list[str]] = {}

    def fake_run(command, **kwargs):
        written["command"] = command

        mount_arg = next(arg for arg in command if "dst=/sandbox/output" in arg)
        output_dir = Path(
            mount_arg.split("src=", 1)[1].split(",dst=/sandbox/output", 1)[0]
        )
        (output_dir / "result.json").write_text(
            json.dumps({"status": "ok", "text": "analysis done", "error": None}),
            encoding="utf-8",
        )
        (output_dir / "stdout.txt").write_text("analysis done\n", encoding="utf-8")
        (output_dir / "figure.png").write_bytes(b"png")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setenv("EXECUTION_MODE", "docker")
    monkeypatch.setenv("SANDBOX_IMAGE", "report-agent-sandbox:test")
    monkeypatch.setattr(execution.tempfile, "TemporaryDirectory", lambda prefix="": _TmpDirCtx(tmp_path))
    monkeypatch.setattr(execution.subprocess, "run", fake_run)

    result, stdout, figure_png, error = execution.run_python_user(
        "print('analysis done')",
        pd.DataFrame({"a": [1]}),
    )

    assert result is None
    assert stdout == "analysis done\n"
    assert figure_png == b"png"
    assert error is None
    assert written["command"][:3] == ["docker", "run", "--rm"]
    assert "--network=none" in written["command"]


def test_run_python_user_returns_timeout_error_for_docker_timeout(monkeypatch) -> None:
    execution = _load_execution_module()
    def fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=["docker"], timeout=20)

    monkeypatch.setenv("EXECUTION_MODE", "docker")
    monkeypatch.setenv("SANDBOX_IMAGE", "report-agent-sandbox:test")
    monkeypatch.setattr(execution.subprocess, "run", fake_run)

    result, stdout, figure_png, error = execution.run_python_user(
        "print('x')",
        pd.DataFrame({"a": [1]}),
    )

    assert result is None
    assert stdout == ""
    assert figure_png == b""
    assert error == {
        "category": "timeout",
        "type": "TimeoutError",
        "message": "Execution exceeded 20s sandbox timeout.",
    }


def test_run_python_user_returns_sandbox_error_when_result_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    execution = _load_execution_module()
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setenv("EXECUTION_MODE", "docker")
    monkeypatch.setenv("SANDBOX_IMAGE", "report-agent-sandbox:test")
    monkeypatch.setattr(execution.tempfile, "TemporaryDirectory", lambda prefix="": _TmpDirCtx(tmp_path))
    monkeypatch.setattr(execution.subprocess, "run", fake_run)

    result, stdout, figure_png, error = execution.run_python_user(
        "print('x')",
        pd.DataFrame({"a": [1]}),
    )

    assert result is None
    assert stdout == ""
    assert figure_png == b""
    assert error == {
        "category": "infrastructure",
        "type": "SandboxExecutionError",
        "message": "Sandboxed execution failed to return result.json.",
    }


def test_run_python_user_maps_missing_package_to_unsupported_runtime(
    monkeypatch,
    tmp_path: Path,
) -> None:
    execution = _load_execution_module()

    def fake_run(command, **kwargs):
        mount_arg = next(arg for arg in command if "dst=/sandbox/output" in arg)
        output_dir = Path(
            mount_arg.split("src=", 1)[1].split(",dst=/sandbox/output", 1)[0]
        )
        (output_dir / "result.json").write_text(
            json.dumps(
                {
                    "status": "error",
                    "text": "",
                    "error": {
                        "type": "PythonRuntimeError",
                        "message": "No module named 'statsmodels'",
                    },
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setenv("EXECUTION_MODE", "docker")
    monkeypatch.setenv("SANDBOX_IMAGE", "report-agent-sandbox:test")
    monkeypatch.setattr(execution.tempfile, "TemporaryDirectory", lambda prefix="": _TmpDirCtx(tmp_path))
    monkeypatch.setattr(execution.subprocess, "run", fake_run)

    result, stdout, figure_png, error = execution.run_python_user(
        "import statsmodels",
        pd.DataFrame({"a": [1]}),
    )

    assert result is None
    assert stdout == ""
    assert figure_png == b""
    assert error == {
        "category": "unsupported_runtime",
        "type": "DependencyNotAvailableError",
        "message": "Sandbox image does not include package: statsmodels",
    }


def test_run_python_user_trusted_local_mode_executes_inline(monkeypatch) -> None:
    execution = _load_execution_module()

    monkeypatch.setenv("EXECUTION_MODE", "trusted_local")

    result, stdout, figure_png, error = execution.run_python_user(
        "result = 42\nprint('ok')",
        pd.DataFrame({"a": [1]}),
    )

    assert result == 42
    assert stdout == "ok\n"
    assert figure_png == b""
    assert error is None


def test_run_python_user_defaults_to_trusted_local_mode(monkeypatch) -> None:
    execution = _load_execution_module()

    monkeypatch.delenv("EXECUTION_MODE", raising=False)

    result, stdout, figure_png, error = execution.run_python_user(
        "result = 3 + 4\nprint('default local')",
        pd.DataFrame({"a": [1]}),
    )

    assert result == 7
    assert stdout == "default local\n"
    assert figure_png == b""
    assert error is None


def test_run_python_user_trusted_local_mode_maps_missing_package_to_unsupported_runtime(
    monkeypatch,
) -> None:
    execution = _load_execution_module()

    monkeypatch.setenv("EXECUTION_MODE", "trusted_local")

    result, stdout, figure_png, error = execution.run_python_user(
        "import definitely_missing_package_xyz",
        pd.DataFrame({"a": [1]}),
    )

    assert result is None
    assert stdout == ""
    assert figure_png == b""
    assert error == {
        "category": "unsupported_runtime",
        "type": "DependencyNotAvailableError",
        "message": "Sandbox image does not include package: definitely_missing_package_xyz",
    }


def test_run_python_user_docker_mode_without_docker_returns_infrastructure_error(
    monkeypatch,
) -> None:
    execution = _load_execution_module()

    def fake_run(*_args, **_kwargs):
        raise FileNotFoundError("docker not found")

    monkeypatch.setenv("EXECUTION_MODE", "docker")
    monkeypatch.setattr(execution.subprocess, "run", fake_run)

    result, stdout, figure_png, error = execution.run_python_user(
        "print('docker fallback')",
        pd.DataFrame({"a": [1]}),
    )

    assert result is None
    assert stdout == ""
    assert figure_png == b""
    assert error == {
        "category": "infrastructure",
        "type": "SandboxExecutionError",
        "message": "Docker is not installed or not available on PATH.",
    }
