from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aerocity_method.adapters.hm3d_runtime import (
    ENGINEERING_EXAMPLE_STATUS,
    FORMAL_SPLIT_TIER,
    OFFICIAL_EXAMPLE_TIER,
    HM3DAssetRecord,
    audit_asset_scope,
    lock_asset,
    reachable_component_mask,
    summarize_official_metadata,
)


def test_official_example_is_engineering_only(tmp_path: Path) -> None:
    asset = tmp_path / "scene.glb"
    asset.write_bytes(b"official-example")
    record = lock_asset(
        asset,
        scene_id="00337-CFVBbU9Rsyb",
        split="example",
        asset_tier=OFFICIAL_EXAMPLE_TIER,
        asset_kind="render_glb",
    )
    audit = audit_asset_scope([record])
    assert audit["status"] == ENGINEERING_EXAMPLE_STATUS
    assert audit["engineering_runtime_authorized"] is True
    assert audit["formal_training_authorized"] is False
    assert audit["p09_freeze_authorized"] is False
    assert audit["formal_results_authorized"] is False


def test_example_cannot_impersonate_formal_split() -> None:
    with pytest.raises(ValueError, match="cannot be represented"):
        HM3DAssetRecord(
            scene_id="00337-CFVBbU9Rsyb",
            split="train",
            asset_tier=OFFICIAL_EXAMPLE_TIER,
            asset_kind="render_glb",
            path="scene.glb",
            sha256="a" * 64,
            bytes=1,
        )


def test_formal_scope_needs_all_three_splits() -> None:
    rows = [
        HM3DAssetRecord(
            scene_id=f"scene-{split}",
            split=split,
            asset_tier=FORMAL_SPLIT_TIER,
            asset_kind="render_glb",
            path=f"{split}.glb",
            sha256=character * 64,
            bytes=1,
        )
        for split, character in (("train", "a"), ("validation", "b"))
    ]
    assert audit_asset_scope(rows)["formal_training_authorized"] is False
    rows.append(
        HM3DAssetRecord(
            scene_id="scene-test",
            split="test",
            asset_tier=FORMAL_SPLIT_TIER,
            asset_kind="render_glb",
            path="test.glb",
            sha256="c" * 64,
            bytes=1,
        )
    )
    assert audit_asset_scope(rows)["formal_training_authorized"] is True
    assert audit_asset_scope(rows)["p09_freeze_authorized"] is False


def test_metadata_summary_keeps_area_units_explicit() -> None:
    rows = [
        {
            "scene": "scene-a",
            "split": "train",
            "num_floors": "2",
            "num_rooms": "10",
            "navigable_area": "100.0",
            "floor_space": "300.0",
        },
        {
            "scene": "scene-b",
            "split": "validation",
            "num_floors": "4",
            "num_rooms": "20",
            "navigable_area": "400.0",
            "floor_space": "900.0",
        },
    ]
    report = summarize_official_metadata(rows)
    assert report["scene_rows"] == 2
    assert report["metrics"]["navigable_area"]["max"] == 400.0
    assert "square metres" in report["note"]
    assert report["largest_by_navigable_area"][0]["scene_id"] == "scene-b"


def test_reachable_denominator_uses_the_union_of_start_components_only() -> None:
    arrays = {
        "free_mask": np.asarray([[[True]], [[True]], [[False]], [[True]]]),
        "component_labels": np.asarray([[[1]], [[1]], [[0]], [[2]]], dtype=np.int32),
        "origin_center_m": np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
        "resolution_m": np.asarray(1.0, dtype=np.float64),
    }

    mask, metadata = reachable_component_mask(
        arrays,
        start_positions_m=((0.0, 0.0, 0.0), (3.0, 0.0, 0.0)),
    )

    assert mask[:, 0, 0].tolist() == [True, True, False, True]
    assert metadata["component_ids"] == [1, 2]
    assert metadata["start_component_ids"] == [1, 2]
    assert metadata["component_voxel_counts"] == {"1": 2, "2": 1}
    assert metadata["reachable_voxel_count"] == 3
    assert metadata["reachable_volume_m3"] == pytest.approx(3.0)
    assert len(metadata["mask_sha256"]) == 64
    assert len(metadata["metadata_sha256"]) == 64


def test_reachable_denominator_rejects_a_start_outside_retained_free_space() -> None:
    arrays = {
        "free_mask": np.asarray([[[True]], [[False]]]),
        "component_labels": np.asarray([[[1]], [[0]]], dtype=np.int32),
        "origin_center_m": np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
        "resolution_m": np.asarray(1.0, dtype=np.float64),
    }

    with pytest.raises(ValueError, match="not in retained free flight space"):
        reachable_component_mask(arrays, start_positions_m=((1.0, 0.0, 0.0),))
