from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "build_hm3d_asset_split_evidence.py"


def _module():
    spec = importlib.util.spec_from_file_location("hm3d_asset_split_evidence", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scene(root: Path, scene_id: str) -> Path:
    path = root / scene_id / f"{scene_id.split('-', 1)[1]}.glb"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(scene_id.encode("ascii"))
    return path


def test_asset_split_preserves_scene_disjointness_and_quarantines_minival(tmp_path: Path):
    module = _module()
    train_root = tmp_path / "train"
    val_root = tmp_path / "val"
    for index in range(10):
        _scene(train_root, f"{index:05d}-train{index:06d}")
    for index in range(800, 812):
        _scene(val_root, f"{index:05d}-val{index:08d}")
    _scene(val_root, "00821-eF36g7L6Z9M")
    tool = tmp_path / "converter.py"
    tool.write_text("converter", encoding="utf-8")
    license_record = tmp_path / "license.md"
    license_record.write_text("licensed", encoding="utf-8")

    p01, p05, partial = module.build_payloads(
        train_root=train_root,
        val_root=val_root,
        conversion_tool=tool,
        license_record_path=license_record,
        train_validation_count=2,
        split_seed="test-seed",
    )

    scenes = p01["payload"]["scenes"]
    assignments = p05["payload"]["scene_assignments"]
    assert len(scenes) == len(assignments) == 23
    assert {row["split"] for row in scenes} == {"train", "validation", "test"}
    assert all(
        row["split"] == "validation"
        for row in scenes
        if row["scene_id"].startswith(("00800-", "00801-", "00802-", "00803-"))
    )
    assert all(
        row["split"] == "test" for row in scenes if row["scene_id"].startswith(("00810-", "00811-"))
    )
    assert (
        next(row for row in scenes if row["scene_id"] == "00821-eF36g7L6Z9M")["split"]
        == "validation"
    )
    assert partial["artifacts"] == []
    assert p05["payload"]["split_manifest_sha256"] == module.canonical_sha256(assignments)
