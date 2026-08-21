import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraud_ml_engineering.run_artifacts import build_run_manifest, create_run_artifacts, write_json


def test_build_run_manifest_is_serializable(tmp_path):
    manifest = build_run_manifest(dataset="dummy", seed=42, repo_root=tmp_path)
    dumped = json.dumps(manifest)
    assert "dataset" in dumped
    assert "recorded_at" in dumped


def test_create_and_round_trip(tmp_path):
    artifacts = create_run_artifacts(tmp_path)
    assert artifacts.run_root.exists()
    assert artifacts.manifest_path.name == "manifest.json"
    assert artifacts.summary_path.name == "summary.json"


def test_write_json_round_trip(tmp_path):
    target = tmp_path / "nested" / "payload.json"
    write_json(target, {"a": 1, "b": [1, 2, 3]})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "b": [1, 2, 3]}