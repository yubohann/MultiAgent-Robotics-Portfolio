"""Bind measured HM3D cohort evidence into fail-closed P03--P05 artifacts.

This assembler does not create measurements.  It validates the immutable
Isaac/PhysX JSON files already produced for the stratified train/validation
cohort, derives only aggregate fields required by the frozen preflight schema,
and refuses to overwrite an artifact or evidence manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerocity_method.contracts.io import canonical_sha256  # noqa: E402
from aerocity_method.evaluation.hm3d_exploration_contract import (  # noqa: E402
    DEFAULT_PATH as DEFAULT_EXPLORATION_CONTRACT,
)
from aerocity_method.evaluation.hm3d_exploration_contract import (  # noqa: E402
    load_exploration_observation_contract,
)
from aerocity_method.evaluation.hm3d_exploration_metrics import (  # noqa: E402
    evaluation_denominator_sha256,
)
from aerocity_method.evaluation.hm3d_preflight import (  # noqa: E402
    METHOD_CORE,
    PREFLIGHT_ARTIFACT_SCHEMA_VERSION,
    PREFLIGHT_EVIDENCE_SCHEMA_VERSION,
    PREFLIGHT_PROTOCOL_SCHEMA_VERSION,
    load_preflight_protocol,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _command_sha256(argv: list[str]) -> str:
    """Return a persisted digest for an assembly command, never a hash object."""

    return hashlib.sha256("\0".join(argv).encode("utf-8")).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _write_new(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite measured artifact: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return _sha256(path)


def _envelope(*, phase: str, kind: str, origin: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PREFLIGHT_ARTIFACT_SCHEMA_VERSION,
        "phase_id": phase,
        "kind": kind,
        "origin": origin,
        "measured": True,
        "synthetic": False,
        "denominator_complete": True,
        "payload": payload,
    }


def _artifact_ref(path: Path, *, phase: str, kind: str, origin: str) -> dict[str, str]:
    return {
        "phase_id": phase,
        "kind": kind,
        "origin": origin,
        "path": str(path.resolve()),
        "sha256": _sha256(path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "external" / "hm3d_formal_preflight_protocol.json",
    )
    parser.add_argument(
        "--exploration-contract",
        type=Path,
        default=DEFAULT_EXPLORATION_CONTRACT,
        help="Frozen public sensor and exploration-metric contract required by P04.",
    )
    parser.add_argument("--p01", type=Path, required=True)
    parser.add_argument("--p02", type=Path, required=True)
    parser.add_argument("--train-observation-ledger", type=Path, required=True)
    parser.add_argument("--validation-observation-ledger", type=Path, required=True)
    parser.add_argument("--train-flight-space", type=Path, required=True)
    parser.add_argument("--validation-flight-space", type=Path, required=True)
    parser.add_argument("--train-vertical-counterfactual", type=Path, required=True)
    parser.add_argument("--validation-vertical-counterfactual", type=Path, required=True)
    parser.add_argument("--train-collision-replay", type=Path, required=True)
    parser.add_argument("--validation-collision-replay", type=Path, required=True)
    parser.add_argument("--p03-output", type=Path, required=True)
    parser.add_argument("--p04-output", type=Path, required=True)
    parser.add_argument("--p05-output", type=Path, required=True)
    parser.add_argument("--evidence-output", type=Path, required=True)
    return parser.parse_args()


def build_p03_scene_row(
    *,
    locked: dict[str, Any],
    flight: dict[str, Any],
    vertical_counterfactual: dict[str, Any],
    replay: dict[str, Any],
) -> dict[str, Any]:
    """Validate one real 3D scene without requiring a P04 observation run.

    P03 establishes evaluator geometry before P04 can collect observations.
    Keeping this construction independent of P04 avoids a circular admission
    dependency.  The operational sensor is sparse range, so P03 proves the
    collision-derived flight geometry, vertical pressure, and PhysX replay;
    P04 separately proves the public range-outcome contract.
    """

    scene_id = flight.get("scene_id")
    if not isinstance(scene_id, str) or scene_id not in locked:
        raise ValueError("flight-space scene is absent from P01 lock")
    if replay.get("scene_id") != scene_id:
        raise ValueError("cohort evidence scene IDs disagree")
    if flight.get("source_glb_sha256") != locked[scene_id]["sha256"]:
        raise ValueError("flight-space source geometry does not match P01")
    if replay.get("collision_usd_sha256") != flight.get("collision_usd_sha256"):
        raise ValueError("collision replay and flight-space collision hashes disagree")
    if replay.get("source_glb_sha256") != locked[scene_id]["sha256"]:
        raise ValueError("collision replay source geometry does not match P01")
    if replay.get("synthetic") is not False or replay.get("measured") is not True:
        raise ValueError("collision replay is not measured runtime evidence")
    if replay.get("status") != "COLLISION_REPLAY_PASSED" or replay.get("passed") is not True:
        raise ValueError("CF2X collision replay did not pass")
    if vertical_counterfactual.get("status") != "P03_VERTICAL_COUNTERFACTUAL_COMPLETE":
        raise ValueError("target-free vertical counterfactual did not complete")
    if (
        vertical_counterfactual.get("synthetic") is not False
        or vertical_counterfactual.get("formal_result") is not False
        or vertical_counterfactual.get("evidence_class") != "real_runtime"
    ):
        raise ValueError("vertical counterfactual is not measured runtime admission evidence")
    if vertical_counterfactual.get("scene_id") != scene_id:
        raise ValueError("vertical counterfactual scene mismatch")
    if vertical_counterfactual.get("source_glb_sha256") != locked[scene_id]["sha256"]:
        raise ValueError("vertical counterfactual source geometry mismatch")
    if vertical_counterfactual.get("collision_usd_sha256") != flight.get("collision_usd_sha256"):
        raise ValueError("vertical counterfactual collision geometry mismatch")
    if vertical_counterfactual.get("flight_space_manifest_hash") != flight.get(
        "flight_space_manifest_hash"
    ):
        raise ValueError("vertical counterfactual flight-space mismatch")
    if vertical_counterfactual.get("sensor_profile") != "sparse-range-3d-vfov90":
        raise ValueError("vertical counterfactual sensor profile drifted")
    if vertical_counterfactual.get("ray_pattern") != "six-axis-range-rays":
        raise ValueError("vertical counterfactual ray pattern drifted")
    raw_counterfactual = vertical_counterfactual.get("fixed_altitude_counterfactual")
    vertical = vertical_counterfactual.get("vertical_geometry_probe")
    if not isinstance(raw_counterfactual, dict) or not isinstance(vertical, dict):
        raise ValueError("vertical counterfactual payload is incomplete")
    if raw_counterfactual.get("run") is not True:
        raise ValueError("target-free fixed-altitude counterfactual was not run")
    for route_name in ("free_height", "fixed_height"):
        route = vertical_counterfactual.get(route_name)
        if not isinstance(route, dict):
            raise ValueError("vertical counterfactual route evidence is missing")
        if int(route.get("route_pose_count", 0)) < 2:
            raise ValueError("vertical counterfactual route has too few poses")
        if int(route.get("accepted_range_outcomes_total", 0)) < 1:
            raise ValueError("vertical counterfactual has no accepted PhysX outcomes")
        metric = route.get("metric")
        if (
            not isinstance(metric, dict)
            or not 0.0 <= float(metric.get("explored_free_flight_volume_auc_time", -1.0)) <= 1.0
        ):
            raise ValueError("vertical counterfactual metric is invalid")
    if (
        vertical_counterfactual["free_height"]["route_pose_count"]
        != vertical_counterfactual["fixed_height"]["route_pose_count"]
    ):
        raise ValueError("vertical counterfactual routes have unequal pose budgets")
    if not 0.0 <= float(vertical.get("vertical_opportunity_fraction", -1.0)) <= 1.0:
        raise ValueError("vertical opportunity fraction is invalid")
    expected_counterfactual_hash = canonical_sha256(
        {
            key: value
            for key, value in vertical_counterfactual.items()
            if key != "counterfactual_sha256"
        }
    )
    if vertical_counterfactual.get("counterfactual_sha256") != expected_counterfactual_hash:
        raise ValueError("vertical counterfactual hash is invalid")
    flight_derivative = flight.get("collision_derivative_provenance")
    replay_derivative = replay.get("collision_derivative_provenance")
    if not isinstance(flight_derivative, dict) or not isinstance(replay_derivative, dict):
        raise ValueError("collision derivative provenance is missing")
    derivative_sha256 = flight_derivative.get("manifest_sha256")
    if derivative_sha256 != replay_derivative.get("manifest_sha256"):
        raise ValueError("flight-space and replay collision derivatives differ")
    if not isinstance(derivative_sha256, str) or len(derivative_sha256) != 64:
        raise ValueError("collision derivative provenance hash is invalid")
    return {
        "scene_id": scene_id,
        "source_geometry_sha256": locked[scene_id]["sha256"],
        "flight_space_manifest_hash": flight["flight_space_manifest_hash"],
        "representation": flight["representation"],
        "dimension": flight["dimension"],
        "resolution_m": flight["resolution_m"],
        "collision_geometry_sha256": flight["collision_usd_sha256"],
        "free_flight_validated": True,
        "generator_version": flight["generator_version"],
        "vehicle_clearance_m": flight["vehicle_clearance_m"],
        "vertical_span_m": flight["flight_space"]["vertical_span_m"],
        "free_flight_volume_m3": flight["flight_space"]["free_flight_volume_m3"],
        "connected_height_band_count": flight["flight_space"]["connected_height_band_count"],
        "vertical_opportunity_fraction": vertical["vertical_opportunity_fraction"],
        "fixed_altitude_control_run": raw_counterfactual["run"],
        "fixed_altitude_control_delta": raw_counterfactual["explored_free_volume_auc_delta"],
        "fixed_altitude_control_relative_gain": (
            raw_counterfactual["explored_free_volume_auc_delta"]
            / max(
                float(
                    vertical_counterfactual["fixed_height"]["metric"][
                        "explored_free_flight_volume_auc_time"
                    ]
                ),
                1.0e-12,
            )
        ),
        "vertical_counterfactual_sha256": _sha256_from_payload(vertical_counterfactual),
        "collision_replay_passed": replay["passed"],
        "flight_space_evidence_sha256": canonical_sha256(flight),
        "collision_replay_evidence_sha256": canonical_sha256(replay),
        "collision_derivative_sha256": derivative_sha256,
    }


def _sha256_from_payload(payload: dict[str, Any]) -> str:
    """Use the persisted runner digest, after independently validating it above."""

    value = payload.get("counterfactual_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("vertical counterfactual digest is missing")
    return value


def _p04_episode(*, p03_row: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": observation["episode_id"],
        "scene_id": p03_row["scene_id"],
        "source_geometry_sha256": p03_row["source_geometry_sha256"],
        "flight_space_manifest_hash": p03_row["flight_space_manifest_hash"],
        "source_observation_ids_total": observation["source_observation_ids_total"],
        "observed_free_voxels_total": observation["observed_free_voxels_total"],
        "observation_voxel_resolution_m": observation["observation_voxel_resolution_m"],
        "source_observation_binding": observation["source_observation_binding"],
        "method_private_truth_fields": observation["method_private_truth_fields"],
    }


def _validate_observation_against_p03(observation: dict[str, Any], p03_row: dict[str, Any]) -> None:
    """Bind a measured P04 ledger to the previously admitted P03 scene."""

    if (
        observation.get("synthetic") is not False
        or observation.get("evidence_class") != "real_runtime"
    ):
        raise ValueError("observation ledger is not measured runtime evidence")
    if observation.get("scene_id") != p03_row["scene_id"]:
        raise ValueError("observation and P03 scene IDs disagree")
    if observation.get("source_geometry_sha256") != p03_row["source_geometry_sha256"]:
        raise ValueError("observation source geometry does not match P03")
    if observation.get("flight_space_manifest_hash") != p03_row["flight_space_manifest_hash"]:
        raise ValueError("observation and P03 flight-space hashes disagree")
    if observation.get("collision_usd_sha256") != p03_row["collision_geometry_sha256"]:
        raise ValueError("observation and P03 collision hashes disagree")
    if observation.get("method_private_truth_fields") != []:
        raise ValueError("observation ledger exposes evaluator truth")


def main() -> int:
    args = parse_args()
    paths = {
        name: getattr(args, name).expanduser().resolve()
        for name in vars(args)
        if name not in {"evidence_root"}
    }
    for name, path in paths.items():
        if name.endswith("output"):
            continue
        if not path.is_file():
            raise FileNotFoundError(f"{name}: {path}")
    protocol = load_preflight_protocol(paths["protocol"])
    exploration_contract = load_exploration_observation_contract(paths["exploration_contract"])
    if protocol.schema_version != PREFLIGHT_PROTOCOL_SCHEMA_VERSION:
        raise ValueError("unexpected preflight protocol schema")
    p01 = _read(paths["p01"])
    p02 = _read(paths["p02"])
    if p01.get("phase_id") != "P01" or p02.get("phase_id") != "P02":
        raise ValueError("P01/P02 artifact identity mismatch")
    p01_payload = p01.get("payload", {})
    scenes = p01_payload.get("scenes")
    if not isinstance(scenes, list) or len(scenes) < 3:
        raise ValueError("P01 scene lock is incomplete")
    locked = {row["scene_id"]: row for row in scenes}
    if len(locked) != len(scenes):
        raise ValueError("P01 contains duplicate scene IDs")
    observations = [
        _read(paths["train_observation_ledger"]),
        _read(paths["validation_observation_ledger"]),
    ]
    flights = [_read(paths["train_flight_space"]), _read(paths["validation_flight_space"])]
    vertical_counterfactuals = [
        _read(paths["train_vertical_counterfactual"]),
        _read(paths["validation_vertical_counterfactual"]),
    ]
    replays = [_read(paths["train_collision_replay"]), _read(paths["validation_collision_replay"])]
    p03_rows = [
        build_p03_scene_row(
            locked=locked,
            flight=flight,
            vertical_counterfactual=vertical_counterfactual,
            replay=replay,
        )
        for flight, vertical_counterfactual, replay in zip(
            flights, vertical_counterfactuals, replays, strict=True
        )
    ]
    p03_rows.sort(key=lambda row: row["scene_id"])
    by_scene = {row["scene_id"]: row for row in p03_rows}
    for observation in observations:
        scene_id = observation.get("scene_id")
        if not isinstance(scene_id, str) or scene_id not in by_scene:
            raise ValueError("observation scene is absent from the P03 cohort")
        _validate_observation_against_p03(observation, by_scene[scene_id])
    splits = {locked[row["scene_id"]]["split"] for row in p03_rows}
    if splits != {"train", "validation"}:
        raise ValueError("cohort must contain exactly one train and one validation scene")
    public_hashes = {row.get("public_contract_sha256") for row in observations}
    denominator_hashes = {row.get("evaluation_denominator_sha256") for row in observations}
    if len(public_hashes) != 1 or not isinstance(next(iter(public_hashes)), str):
        raise ValueError("cohort public observation contract hashes do not match")
    if next(iter(public_hashes)) != exploration_contract.digest:
        raise ValueError("observation ledger public contract differs from frozen P04 contract")
    if len(denominator_hashes) != 1 or not isinstance(next(iter(denominator_hashes)), str):
        raise ValueError("cohort evaluator denominator hashes do not match")
    expected_denominator_hash = evaluation_denominator_sha256(tuple(p03_rows))
    if next(iter(denominator_hashes)) != expected_denominator_hash:
        raise ValueError("observation ledgers do not bind the P03 evaluator denominator")
    # Earlier independent calibration/replay probes have immutable data hashes but
    # no command field.  Do not invent historical commands: this records the
    # actual assembly command, while each new counterfactual carries its own
    # runtime command and content digest.
    p03_command_hash = _command_sha256([str(value) for value in sys.argv])
    p04_command_hash = canonical_sha256(
        sorted(row["runtime_command_sha256"] for row in observations)
    )
    p03_payload = {
        "evidence_class": "real_runtime_measurement",
        "runtime_run_id": "isaac-hm3d-stratified-cohort-flight-space-vfov90-20260803",
        "runtime_command_sha256": p03_command_hash,
        "navmesh_authorizes_flight": False,
        "admission_scope": "stratified_development_cohort",
        "scenes": p03_rows,
    }
    assignments = [
        {"scene_id": row["scene_id"], "split": row["split"], "asset_sha256": row["sha256"]}
        for row in sorted(scenes, key=lambda row: row["scene_id"])
    ]
    split_hash = canonical_sha256(assignments)
    p04_payload = {
        "evidence_class": "real_runtime_measurement",
        "runtime_run_id": "isaac-hm3d-public-observation-cohort-vfov90-20260803",
        "runtime_command_sha256": p04_command_hash,
        "public_contract_sha256": next(iter(public_hashes)),
        "evaluation_denominator_sha256": next(iter(denominator_hashes)),
        "split_manifest_sha256": split_hash,
        "episodes": [
            _p04_episode(
                p03_row=row,
                observation=next(
                    observation
                    for observation in observations
                    if observation["scene_id"] == row["scene_id"]
                ),
            )
            for row in p03_rows
        ],
    }
    episode_manifest = {
        "episode_id_by_scene": {
            row["scene_id"]: episode["episode_id"]
            for row, episode in zip(p03_rows, p04_payload["episodes"], strict=True)
        },
        "public_observation_contract": p04_payload["public_contract_sha256"],
    }
    difficulty_distribution = {
        "observation_voxel_resolution_m": p04_payload["episodes"][0][
            "observation_voxel_resolution_m"
        ],
        "sensor_contract": "sparse_range_3d_with_source_observation_id",
        "height_balancing": "geometric-height-bands-frozen-before-validation",
    }
    p05_payload = {
        "evidence_class": "source_license_audit",
        "official_split_provenance": (
            "hm3d-v0.2-official-asset-layout-locked-by-P01-no-test-development-access-v1"
        ),
        "dataset_version": p01_payload["dataset_version"],
        "scene_assignments": assignments,
        "split_manifest_sha256": split_hash,
        "public_contract_sha256": next(iter(public_hashes)),
        "evaluation_denominator_sha256": next(iter(denominator_hashes)),
        "episode_seed_manifest_sha256": canonical_sha256(episode_manifest),
        "difficulty_distribution_sha256": canonical_sha256(difficulty_distribution),
        "run_partition": "development",
        "test_used_for_development": False,
        "test_access_count_before_freeze": 0,
    }
    p03_hash = _write_new(
        paths["p03_output"],
        _envelope(phase="P03", kind="flight_space_3d", origin="real_runtime", payload=p03_payload),
    )
    p04_hash = _write_new(
        paths["p04_output"],
        _envelope(
            phase="P04",
            kind="public_observation_contract",
            origin="real_runtime",
            payload=p04_payload,
        ),
    )
    p05_hash = _write_new(
        paths["p05_output"],
        _envelope(
            phase="P05",
            kind="scene_split_freeze",
            origin="source_license_audit",
            payload=p05_payload,
        ),
    )
    artifact_rows = [
        _artifact_ref(paths["p01"], phase="P01", kind="asset_lock", origin="source_license_audit"),
        _artifact_ref(paths["p02"], phase="P02", kind="runtime_aba_reset", origin="real_runtime"),
        {
            "phase_id": "P03",
            "kind": "flight_space_3d",
            "origin": "real_runtime",
            "path": str(paths["p03_output"]),
            "sha256": p03_hash,
        },
        {
            "phase_id": "P04",
            "kind": "public_observation_contract",
            "origin": "real_runtime",
            "path": str(paths["p04_output"]),
            "sha256": p04_hash,
        },
        {
            "phase_id": "P05",
            "kind": "scene_split_freeze",
            "origin": "source_license_audit",
            "path": str(paths["p05_output"]),
            "sha256": p05_hash,
        },
    ]
    evidence_payload = {
        "schema_version": PREFLIGHT_EVIDENCE_SCHEMA_VERSION,
        "protocol_hash": protocol.protocol_hash,
        "requested_gate": "formal_experiment_start",
        "method_core": METHOD_CORE,
        "aerocity_bench_accesses": [],
        "artifacts": artifact_rows,
    }
    _write_new(paths["evidence_output"], evidence_payload)
    print(
        json.dumps(
            {
                "status": "P03_P04_P05_ASSEMBLED",
                "p03": str(paths["p03_output"]),
                "p04": str(paths["p04_output"]),
                "p05": str(paths["p05_output"]),
                "evidence": str(paths["evidence_output"]),
                "protocol_hash": protocol.protocol_hash,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
