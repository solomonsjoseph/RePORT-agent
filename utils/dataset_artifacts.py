from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_ROOT = PROJECT_ROOT / "runtime" / "datasets"


def persist_dataset_artifact(
    *,
    runtime_root: str | Path | None,
    thread_id: str,
    dataset_id: str,
    kind: str,
    dataframe: pd.DataFrame,
    schema: dict[str, Any] | None,
    provenance: dict[str, Any] | None,
) -> dict[str, Any]:
    root = Path(runtime_root or DEFAULT_RUNTIME_ROOT) / thread_id
    root.mkdir(parents=True, exist_ok=True)

    data_path = root / f"{dataset_id}.parquet"
    schema_path = root / f"{dataset_id}.schema.json"
    metadata_path = root / f"{dataset_id}.metadata.json"

    dataframe.to_parquet(data_path, index=False)
    schema_path.write_text(json.dumps(schema or {}, indent=2, ensure_ascii=False), encoding="utf-8")

    artifact = {
        "id": dataset_id,
        "kind": kind,
        "path": str(data_path),
        "schema_path": str(schema_path),
        "metadata_path": str(metadata_path),
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "columns": list(dataframe.columns),
        "created_at": datetime.now(UTC).isoformat(),
        "provenance": dict(provenance or {}),
    }
    metadata_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    return artifact


def register_dataset_artifact(
    state: dict[str, Any],
    artifact: dict[str, Any],
    *,
    make_active: bool,
) -> dict[str, Any]:
    artifacts = dict(state.get("artifacts") or {})
    datasets = dict(artifacts.get("datasets") or {})
    datasets[artifact["id"]] = artifact
    artifacts["datasets"] = datasets
    if make_active:
        artifacts["active_dataset_id"] = artifact["id"]
    return {
        **state,
        "artifacts": artifacts,
    }


def build_dataset_artifacts_patch(
    current_artifacts: dict[str, Any] | None,
    artifact: dict[str, Any] | None,
    *,
    make_active: bool = True,
) -> dict[str, Any]:
    artifacts = dict(current_artifacts or {})
    if not artifact:
        return artifacts

    datasets = dict(artifacts.get("datasets") or {})
    datasets[artifact["id"]] = artifact
    artifacts["datasets"] = datasets
    if make_active:
        artifacts["active_dataset_id"] = artifact["id"]
    return artifacts


def load_dataset_artifact(artifact: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_parquet(artifact["path"])
    schema_path = artifact.get("schema_path")
    schema = {}
    if schema_path and Path(schema_path).exists():
        schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    return df, schema


def build_dataset_context(artifact: dict[str, Any] | None) -> str:
    if not artifact:
        return "No dataset or schema provided."

    df, schema = load_dataset_artifact(artifact)
    cols = df.columns.tolist()
    col_section = "\n".join(f"- {c}" for c in cols)

    schema_lines = []
    for col, meta in dict(schema or {}).items():
        if not isinstance(meta, dict):
            continue
        schema_lines.append(
            f"{col}:\n"
            f"  • Description: {meta.get('description', 'N/A')}\n"
            f"  • Type: {meta.get('dataType', 'N/A')}\n"
            f"  • Notes: {meta.get('notes', '')}\n"
        )

    schema_text = "\n".join(schema_lines) if schema_lines else "No schema metadata available."
    return (
        "Available columns:\n"
        f"{col_section}\n\n"
        "Column metadata:\n"
        f"{schema_text}"
    )


def get_registered_datasets(state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return dict((state.get("artifacts") or {}).get("datasets") or {})


def get_active_dataset_artifact(state: dict[str, Any]) -> dict[str, Any] | None:
    artifacts = dict(state.get("artifacts") or {})
    dataset_id = artifacts.get("active_dataset_id")
    datasets = get_registered_datasets(state)
    if dataset_id:
        return datasets.get(dataset_id)
    if len(datasets) == 1:
        return next(iter(datasets.values()))
    return None


def get_analysis_dataset_candidates(state: dict[str, Any]) -> list[dict[str, Any]]:
    return list(get_registered_datasets(state).values())


def build_active_dataset_artifacts_patch(
    current_artifacts: dict[str, Any] | None,
    dataset_id: str,
) -> dict[str, Any]:
    artifacts = dict(current_artifacts or {})
    datasets = dict(artifacts.get("datasets") or {})
    if dataset_id not in datasets:
        raise KeyError(f"Unknown dataset id: {dataset_id}")
    artifacts["datasets"] = datasets
    artifacts["active_dataset_id"] = dataset_id
    return artifacts


def choose_analysis_dataset(
    state: dict[str, Any],
    *,
    latest_user_message: str,
) -> tuple[dict[str, Any] | None, str]:
    del latest_user_message
    candidates = get_analysis_dataset_candidates(state)
    if not candidates:
        return None, "missing"

    meta = dict(state.get("meta") or {})
    explicit_id = meta.get("analysis_dataset_id")
    if explicit_id:
        for artifact in candidates:
            if artifact.get("id") == explicit_id:
                return artifact, "explicit"

    if len(candidates) == 1:
        return candidates[0], "single"

    if len(candidates) > 1:
        return None, "ambiguous"
    return None, "missing"
