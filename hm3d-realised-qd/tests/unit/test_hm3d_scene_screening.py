from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "screen_hm3d_p07_scene_candidates.py"
SPEC = importlib.util.spec_from_file_location("hm3d_scene_screening", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _metadata(scene: str, floors: int, rooms: int, area: float) -> dict[str, str]:
    return {
        "scene": scene,
        "split": "train",
        "num_floors": str(floors),
        "num_rooms": str(rooms),
        "navigable_area": str(area),
        "floor_space": str(area * 2.0),
        "navigation_complexity": str(area / 10.0),
        "scene_clutter": "3.0",
        "overall_scene_quality": "2",
        "diversity": "1",
    }


def _glb(root: Path, scene: str) -> None:
    token = scene.split("-", 1)[1]
    path = root / scene / f"{token}.glb"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"official-scene-placeholder")


def test_screen_uses_only_installed_official_train_rows_and_keeps_floor_claims_limited(
    tmp_path: Path,
) -> None:
    root = tmp_path / "train"
    rows = [
        _metadata("00001-single", 1, 14, 300.0),
        _metadata("00002-multi", 2, 20, 430.0),
        _metadata("00003-scale", 4, 40, 2000.0),
        _metadata("00004-missing", 2, 30, 500.0),
        {**_metadata("00005-val", 3, 30, 500.0), "split": "val"},
    ]
    for scene in ("00001-single", "00002-multi", "00003-scale"):
        _glb(root, scene)

    report = MODULE.screen_train_candidates(
        rows,
        root,
        minimum_room_count=12,
        first_admission_max_navigable_area_m2=750.0,
        shortlist_size=10,
    )

    assert report["installed_train_scene_count"] == 3
    admission = report["next_runtime_admission_order"]
    assert admission["single_floor_controller_calibration"]["scene_id"] == "00001-single"
    assert admission["multi_floor_primary"]["scene_id"] == "00002-multi"
    assert admission["largest_multi_floor_scale_stress"]["scene_id"] == "00003-scale"
    forbidden = report["selection_contract"]["forbidden_inference"]
    assert "floor count is not vertical free-flight evidence" in forbidden


def test_screen_fails_when_no_practical_multifloor_runtime_candidate_exists(tmp_path: Path) -> None:
    root = tmp_path / "train"
    rows = [_metadata("00001-single", 1, 14, 300.0), _metadata("00002-huge", 2, 20, 800.0)]
    for scene in ("00001-single", "00002-huge"):
        _glb(root, scene)
    with pytest.raises(ValueError, match="no practical first runtime candidate"):
        MODULE.screen_train_candidates(
            rows,
            root,
            minimum_room_count=12,
            first_admission_max_navigable_area_m2=750.0,
            shortlist_size=10,
        )


def test_training_queue_is_deterministic_and_covers_floor_area_complexity_strata(
    tmp_path: Path,
) -> None:
    root = tmp_path / "train"
    rows: list[dict[str, str]] = []
    for floors in range(1, 5):
        for index in range(12):
            scene = f"{floors}{index:04d}-scene{floors}{index:02d}"
            row = _metadata(scene, floors, 12 + index, 40.0 + 10.0 * index)
            row["navigation_complexity"] = str(2.0 + ((index * 5) % 12))
            rows.append(row)
            _glb(root, scene)

    kwargs = {
        "minimum_room_count": 1,
        "first_admission_max_navigable_area_m2": 1000.0,
        "shortlist_size": 4,
        "training_cohort_size": 20,
        "selection_seed": "frozen-seed",
        "minimum_per_floor_stratum": 3,
    }
    first = MODULE.screen_train_candidates(rows, root, **kwargs)
    second = MODULE.screen_train_candidates(rows, root, **kwargs)
    first_queue = first["metadata_training_admission_queue"]
    second_queue = second["metadata_training_admission_queue"]

    first_ids = [row["scene_id"] for row in first_queue["primary_queue"]]
    second_ids = [row["scene_id"] for row in second_queue["primary_queue"]]
    assert first_ids == second_ids
    assert len(first_ids) == 20
    assert set(first_queue["floor_selection_quotas"]) == {"1", "2", "3", "4_plus"}
    assert all(value >= 3 for value in first_queue["floor_selection_quotas"].values())
    assert len({row["area_quantile_within_floor"] for row in first_queue["primary_queue"]}) == 3
    assert (
        len({row["complexity_quantile_within_floor"] for row in first_queue["primary_queue"]}) == 3
    )
    assert (
        len(
            {
                (row["area_quantile_within_floor"], row["complexity_quantile_within_floor"])
                for row in first_queue["primary_queue"]
            }
        )
        >= 6
    )
    assert not set(first_ids).intersection(
        row["scene_id"] for row in first_queue["same_protocol_reserves"]
    )
    audit = first_queue["metadata_diversity_audit"]
    assert audit["exact_duplicate_vector_pairs"] == 0
    assert audit["minimum_within_floor_pairwise_distance"] > 0.0
    assert set(audit["feature_axes"]) == {
        "num_floors",
        "num_rooms",
        "navigable_area_m2",
        "floor_space_m2",
        "navigation_complexity",
        "scene_clutter",
        "floor_space_per_floor_m2",
        "room_density_per_100_navigable_m2",
        "navigable_to_floor_space_ratio",
    }
    assert all(
        set(row["metadata_diversity_vector"]) == set(audit["feature_axes"])
        for row in first_queue["primary_queue"]
    )
