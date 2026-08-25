from __future__ import annotations

import json

from fraud_ml_engineering.run_artifacts import build_run_manifest, create_run_artifacts, write_json


def test_create_run_artifacts_creates_a_unique_run_directory(tmp_path) -> None:
    artifacts = create_run_artifacts(tmp_path, prefix="smoke")

    assert artifacts.run_root.is_dir()
    assert artifacts.run_root.name.startswith("smoke_")
    assert artifacts.manifest_path.parent == artifacts.run_root


def test_manifest_captures_reproducibility_context_without_training_dependencies(tmp_path) -> None:
    manifest = build_run_manifest(
        dataset="elliptic",
        seed=42,
        command=["python", "-m", "fraud_ml_engineering", "--dataset", "elliptic"],
        config_path="configs/experiments/onchain_main_selection.yaml",
        notes="dataset revision: provider snapshot 2026-07-31",
        extra={"label_fraction": 0.1},
    )
    destination = tmp_path / "manifest.json"
    write_json(destination, manifest)

    loaded = json.loads(destination.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == 1
    assert loaded["experiment"]["dataset"] == "elliptic"
    assert loaded["experiment"]["seed"] == 42
    assert loaded["experiment"]["extra"]["label_fraction"] == 0.1
    assert loaded["runtime"]["python_version"]
    assert not destination.with_suffix(".json.tmp").exists()
