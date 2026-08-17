from __future__ import annotations

import json
from pathlib import Path

import pytest

from rivermark_benchmark.external_sources import (
    EXTERNAL_SOURCE_SNAPSHOT_SCHEMA,
    ExternalSourceError,
    scan_external_source_snapshots,
    write_external_source_manifest,
)


def _write(root: Path, relative: str, content: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _materialize_lerobot(root: Path) -> None:
    base = root / "lerobot-main"
    for relative in (
        "README.md",
        "pyproject.toml",
        "src/lerobot/datasets/dataset_writer.py",
        "src/lerobot/datasets/lerobot_dataset.py",
    ):
        _write(base, relative, relative)


def test_external_source_snapshot_is_path_free_and_hash_bound(tmp_path: Path) -> None:
    source_root = tmp_path / "external"
    _materialize_lerobot(source_root)

    manifest = scan_external_source_snapshots(source_root, source_ids=["lerobot"])

    assert manifest["schema"] == EXTERNAL_SOURCE_SNAPSHOT_SCHEMA
    assert manifest["status"] == "complete"
    assert manifest["source_count"] == 1
    record = manifest["records"][0]
    assert record["status"] == "complete"
    assert record["file_count"] == 4
    assert record["total_bytes"] > 0
    assert len(record["key_files"]) == 4
    encoded = json.dumps(manifest, sort_keys=True)
    assert str(source_root) not in encoded
    assert str(tmp_path) not in encoded


def test_external_source_snapshot_preserves_missing_source_denominator(tmp_path: Path) -> None:
    manifest = scan_external_source_snapshots(tmp_path, source_ids=["rlds"])

    assert manifest["status"] == "incomplete"
    assert manifest["complete_source_count"] == 0
    record = manifest["records"][0]
    assert record["status"] == "missing"
    assert record["file_count"] == 0
    assert record["total_bytes"] == 0


def test_external_source_snapshot_rejects_unknown_or_repository_local_sources(tmp_path: Path) -> None:
    with pytest.raises(ExternalSourceError, match="unknown"):
        scan_external_source_snapshots(tmp_path, source_ids=["unknown"])
    with pytest.raises(ExternalSourceError, match="outside"):
        scan_external_source_snapshots(tmp_path, source_ids=["rlds"], repository_root=tmp_path)


def test_write_external_source_manifest_is_atomic_and_hash_checked(tmp_path: Path) -> None:
    source_root = tmp_path / "external"
    _materialize_lerobot(source_root)
    manifest = scan_external_source_snapshots(source_root, source_ids=["lerobot"])
    output = write_external_source_manifest(tmp_path / "manifest.json", manifest)

    written = json.loads(output.read_text(encoding="utf-8"))
    assert written == manifest
    tampered = dict(manifest)
    tampered["status"] = "incomplete"
    with pytest.raises(ExternalSourceError, match="hash"):
        write_external_source_manifest(tmp_path / "tampered.json", tampered)
