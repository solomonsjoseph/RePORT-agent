from __future__ import annotations

from pathlib import Path

import pandas as pd


def test_persist_dataset_artifact_writes_parquet_and_metadata(tmp_path: Path) -> None:
    from utils.dataset_artifacts import persist_dataset_artifact

    df = pd.DataFrame({"sex": ["Male", "Female"]})
    artifact = persist_dataset_artifact(
        runtime_root=tmp_path,
        thread_id="thread-1",
        dataset_id="uploaded-1",
        kind="uploaded",
        dataframe=df,
        schema={"sex": {"description": "Biological sex"}},
        provenance={"source": "upload"},
    )

    assert Path(artifact["path"]).exists()
    assert Path(artifact["schema_path"]).exists()
    assert artifact["row_count"] == 2
    assert artifact["kind"] == "uploaded"


def test_register_dataset_artifact_sets_active_pointer_for_new_subset() -> None:
    from utils.dataset_artifacts import register_dataset_artifact

    state = {
        "artifacts": {
            "datasets": {
                "uploaded-1": {"id": "uploaded-1", "kind": "uploaded"},
            },
            "active_dataset_id": "uploaded-1",
        }
    }

    updated = register_dataset_artifact(
        state,
        {
            "id": "subset-1",
            "kind": "subset",
            "path": "/tmp/subset.parquet",
            "schema_path": "/tmp/subset.schema.json",
            "provenance": {"sql": "SELECT 1"},
        },
        make_active=True,
    )

    assert updated["artifacts"]["active_dataset_id"] == "subset-1"
    assert updated["artifacts"]["datasets"]["subset-1"]["kind"] == "subset"


def test_choose_analysis_dataset_requires_explicit_choice_when_uploaded_and_subset_exist() -> None:
    from utils.dataset_artifacts import choose_analysis_dataset

    state = {
        "artifacts": {
            "datasets": {
                "uploaded-1": {"id": "uploaded-1", "kind": "uploaded"},
                "subset-1": {"id": "subset-1", "kind": "subset"},
            },
            "active_dataset_id": "subset-1",
        },
        "meta": {},
    }

    artifact, reason = choose_analysis_dataset(state, latest_user_message="Run a Cox model")

    assert artifact is None
    assert reason == "ambiguous"
