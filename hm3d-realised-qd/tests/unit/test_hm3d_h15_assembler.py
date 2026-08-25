from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from aerocity_method.runtime.sensors import FORMAL_H15_SENSOR_PILOT_MODES, SensorProfile

ROOT = Path(__file__).resolve().parents[2]
ASSEMBLER_PATH = ROOT / "scripts" / "assemble_hm3d_h15_sensor_pilot.py"


def _load_assembler():
    spec = importlib.util.spec_from_file_location("hm3d_h15_assembler", ASSEMBLER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _profile(mode: str) -> SensorProfile:
    if mode == "physics_only":
        return SensorProfile("physics-only-baseline", mode, 0.0, (), ())
    if mode == "sparse_range_3d":
        return SensorProfile(
            "sparse-range-3d-vfov90",
            mode,
            10.0,
            ("transit", "observe", "dwell", "map_update"),
            ("range_points", "source_observation_id"),
            range_enabled=True,
        )
    raise ValueError(f"unsupported formal H15 mode: {mode}")


def _worker_row(fleet_size: int, mode: str) -> dict[str, object]:
    profile = _profile(mode)
    observations = [0] * fleet_size if mode == "physics_only" else [3] * fleet_size
    return {
        "schema_version": "hm3d-h15-sensor-row-v3",
        "status": "H15_REAL_SENSOR_ROW_COMPLETE",
        "synthetic": False,
        "evidence_class": "real_runtime_measurement",
        "runtime_run_id": f"isaac-hm3d-h15-{fleet_size}-{mode}",
        "runtime_command_sha256": "a" * 64,
        "runner_version": "hm3d-h15-real-sensor-row-v4",
        "source_observation_binding": True,
        "selection_partition": "train",
        "scene_id": "00244-E64sjs3Dyfd",
        "collision_usd_sha256": "c" * 64,
        "receiver_position_source_sha256": "d" * 64,
        "pilot_claim_limit": "Throughput only",
        "fleet_size": fleet_size,
        "mode": mode,
        "record": {
            "comparison_id": "h15-hm3d-train-00244-vfov90-v1",
            "scene_id": "00244-E64sjs3Dyfd",
            "episode_id": "h15-throughput-episode",
            "fleet_size": fleet_size,
            "profile": profile.to_dict(),
            "physics_dt_s": 1.0 / 120.0,
            "planned_episodes": 1,
            "executed_episodes": 1,
            "failed_episodes": 0,
            "physics_real_time_factor": 1.0,
            "environment_steps_per_s": 120.0,
            "sensor_frames_per_s": 0.0 if mode == "physics_only" else 10.0,
            "render_time_s": 0.0,
            "transfer_time_s": 0.0,
            "gpu_memory_mb": 1.0,
            "cpu_memory_mb": 1.0,
            "dropped_frames": 0,
            "observations_per_agent": observations,
            "measurement_scope": "throughput_only",
            "wall_clock_s": 1.0,
        },
    }


def _write_complete_matrix(rows_dir: Path) -> None:
    rows_dir.mkdir()
    for mode in FORMAL_H15_SENSOR_PILOT_MODES:
        (rows_dir / f"row_N4_{mode}_v3.json").write_text(
            json.dumps(_worker_row(4, mode)), encoding="utf-8"
        )


def test_assembler_returns_exact_p06_payload_and_full_provenance(tmp_path: Path):
    module = _load_assembler()
    rows_dir = tmp_path / "rows"
    _write_complete_matrix(rows_dir)

    payload, audit = module.assemble(rows_dir, "sparse_range_3d")

    assert set(payload) == {
        "evidence_class",
        "runtime_run_id",
        "runtime_command_sha256",
        "source_observation_binding",
        "selection_partition",
        "records",
        "selected_profile",
        "entitlements",
    }
    assert len(payload["records"]) == 2
    assert payload["selected_profile"]["mode"] == "sparse_range_3d"
    assert {row["method_id"] for row in payload["entitlements"]} == {
        "random",
        "frontier_3d",
        "auction",
        "gvp_mrep_port",
        "single_rl",
        "realised_qd_ogfr_rb_sf_sac",
        "no_qd",
        "planned_qd",
        "realised_qd",
        "no_ogfr",
        "ogfr",
        "rb_sf_sac_reference",
        "rb_sf_sac_selected",
    }
    assert audit["matrix"]["status"] == "PASS"
    assert len(audit["row_files"]) == 2


def test_assembler_rejects_mixed_source_hashes(tmp_path: Path):
    module = _load_assembler()
    rows_dir = tmp_path / "rows"
    _write_complete_matrix(rows_dir)
    changed = rows_dir / "row_N4_sparse_range_3d_v3.json"
    payload = json.loads(changed.read_text(encoding="utf-8"))
    payload["collision_usd_sha256"] = "e" * 64
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="source-consistent"):
        module.assemble(rows_dir, "sparse_range_3d")


def test_assembler_ignores_smoke_rows_and_requires_full_formal_matrix(tmp_path: Path):
    module = _load_assembler()
    rows_dir = tmp_path / "rows"
    rows_dir.mkdir()
    (rows_dir / "smoke_n4_range.json").write_text(
        json.dumps(_worker_row(4, "sparse_range_3d")), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="no formal H15 rows"):
        module.assemble(rows_dir, "sparse_range_3d")


def test_assembler_rejects_legacy_task_quality_row_schema(tmp_path: Path):
    module = _load_assembler()
    rows_dir = tmp_path / "rows"
    _write_complete_matrix(rows_dir)
    changed = rows_dir / "row_N4_sparse_range_3d_v3.json"
    payload = json.loads(changed.read_text(encoding="utf-8"))
    payload["record"]["explored_free_flight_volume_auc_time"] = 0.0
    payload["record"].pop("measurement_scope")
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="fields mismatch"):
        module.assemble(rows_dir, "sparse_range_3d")


def test_assembler_rejects_non_clean_worker_exit_ledger(tmp_path: Path):
    module = _load_assembler()
    rows_dir = tmp_path / "rows"
    _write_complete_matrix(rows_dir)
    ledger = {
        "schema_version": "hm3d-h15-matrix-ledger-v3",
        "status": "H15_MATRIX_COMPLETE",
        "completed_rows": [
            {"fleet_size": fleet, "mode": mode, "status": "completed"}
            for fleet in (4,)
            for mode in FORMAL_H15_SENSOR_PILOT_MODES
        ],
    }
    ledger["completed_rows"][-1]["status"] = "failed"
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    with pytest.raises(ValueError, match="missing or non-clean"):
        module.assemble(
            rows_dir,
            "sparse_range_3d",
            matrix_ledger=ledger,
            matrix_ledger_path=ledger_path,
        )
