"""Run train-only public replay calibration for the realised-QD descriptor.

This launcher is deliberately outside P07/P08 ranking.  It selects six public
intent extremes, runs every case twice from the exact same public reset, and
then verifies that the selected manifest really repeated before the resulting
outcomes may be offered as QD train history.  It does not invent a descriptor,
score an episode, or turn calibration output into a formal result.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerocity_method.contracts import FORMAL_FLEET_SIZE  # noqa: E402
from aerocity_method.contracts.io import canonical_sha256  # noqa: E402
from aerocity_method.runtime.hm3d_realised_qd import (  # noqa: E402
    HM3D_QD_CALIBRATION_INTENT_MODES,
    RealisedQDDescriptor,
    OutcomeQDFeatureVector,
    audit_pre_registered_qd_descriptor_families,
    audit_realised_qd_calibration_mode_contrasts,
    audit_realised_qd_reproducibility,
)

CALIBRATION_INTENT_MODES = HM3D_QD_CALIBRATION_INTENT_MODES
_FORBIDDEN_BASE_ARGUMENTS = {
    "--output",
    "--strategy",
    "--split",
    "--random-key",
    "--qd-calibration-mode",
}


@dataclass(frozen=True, slots=True)
class CalibrationCase:
    """One fixed train reset shared by all six public intent modes."""

    case_id: str
    scene_id: str
    random_key: int
    runner_arguments: tuple[str, ...]

    @classmethod
    def from_dict(cls, payload: object) -> CalibrationCase:
        if not isinstance(payload, dict):
            raise ValueError("QD calibration case must be an object")
        if set(payload) != {"case_id", "scene_id", "random_key", "runner_arguments"}:
            raise ValueError("QD calibration case fields are invalid")
        case_id = payload["case_id"]
        scene_id = payload["scene_id"]
        random_key = payload["random_key"]
        raw_arguments = payload["runner_arguments"]
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("QD calibration case_id is invalid")
        if not isinstance(scene_id, str) or not scene_id:
            raise ValueError("QD calibration scene_id is invalid")
        if not isinstance(random_key, int) or isinstance(random_key, bool) or random_key < 0:
            raise ValueError("QD calibration random_key is invalid")
        if not isinstance(raw_arguments, list) or not all(
            isinstance(value, str) for value in raw_arguments
        ):
            raise ValueError("QD calibration runner_arguments must be a string list")
        arguments = tuple(raw_arguments)
        if any(value in _FORBIDDEN_BASE_ARGUMENTS for value in arguments):
            raise ValueError("QD calibration runner_arguments may not override fixed replay fields")
        if arguments.count("--scene-id") != 1:
            raise ValueError("QD calibration runner_arguments must declare scene_id once")
        scene_index = arguments.index("--scene-id")
        if scene_index + 1 >= len(arguments) or arguments[scene_index + 1] != scene_id:
            raise ValueError("QD calibration case scene_id disagrees with runner_arguments")
        return cls(case_id, scene_id, random_key, arguments)


@dataclass(frozen=True, slots=True)
class CalibrationPlan:
    case: CalibrationCase
    intent_mode: str
    repetition: int
    output_path: Path

    def command(self, *, python: Path, runner: Path) -> tuple[str, ...]:
        return (
            str(python),
            str(runner),
            *self.case.runner_arguments,
            "--split",
            "train",
            "--strategy",
            "qd_calibration",
            "--qd-calibration-mode",
            self.intent_mode,
            "--random-key",
            str(self.case.random_key),
            "--output",
            str(self.output_path),
        )


def _load_cases(path: Path) -> tuple[CalibrationCase, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("QD replay calibration case file must be a JSON array")
    cases = tuple(CalibrationCase.from_dict(row) for row in payload)
    if len(cases) < 2:
        raise ValueError("QD replay calibration needs at least two train scenes")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("QD replay calibration case_id values must be unique")
    if len({case.scene_id for case in cases}) < 2:
        raise ValueError("QD replay calibration must cover at least two train scenes")
    return cases


def build_calibration_plan(
    *,
    cases: Sequence[CalibrationCase],
    output_dir: Path,
    repetitions: int = 2,
    modes: Sequence[str] = CALIBRATION_INTENT_MODES,
) -> tuple[CalibrationPlan, ...]:
    """Build two independent executions for each public mode in every train case."""

    if repetitions < 2:
        raise ValueError("QD replay calibration requires at least two independent repetitions")
    selected_modes = tuple(modes)
    if set(selected_modes) != set(CALIBRATION_INTENT_MODES) or len(selected_modes) != len(
        CALIBRATION_INTENT_MODES
    ):
        raise ValueError("QD replay calibration must retain all six frozen public intent modes")
    plans = tuple(
        CalibrationPlan(
            case=case,
            intent_mode=mode,
            repetition=repetition,
            output_path=output_dir / f"{case.case_id}__{mode}__replay{repetition + 1:02d}.json",
        )
        for case in cases
        for mode in selected_modes
        for repetition in range(repetitions)
    )
    if len(plans) < 24:
        raise ValueError(
            "QD replay calibration needs 24 real executions for two scenes and six modes"
        )
    return plans


def _record_manifests(payload: dict[str, Any]) -> tuple[str, ...]:
    if payload.get("schema_version") != "hm3d-p07-exploration-execution-v1":
        raise ValueError("QD replay calibration output is not a current P07 worker record")
    if payload.get("synthetic") is not False or payload.get("selection_partition") != "train":
        raise ValueError("QD replay calibration requires real train-partition worker records")
    if payload.get("fleet_size") != FORMAL_FLEET_SIZE:
        raise ValueError("QD replay calibration worker violates the formal four-CF2X contract")
    if payload.get("strategy") != "qd_calibration":
        raise ValueError("QD replay calibration output has the wrong strategy")
    qd = payload.get("realised_qd")
    if not isinstance(qd, dict):
        raise ValueError("QD replay calibration output lacks realised-QD outcomes")
    raw_admissions = qd.get("admissions")
    if not isinstance(raw_admissions, list) or not raw_admissions:
        raise ValueError("QD replay calibration output has no executed descriptor admissions")
    manifests: list[str] = []
    for row in raw_admissions:
        if not isinstance(row, dict) or row.get("executed") is not True:
            raise ValueError("QD replay calibration has an incomplete execution outcome")
        if row.get("feasible") is not True:
            raise ValueError("QD replay calibration cannot admit unsafe or empty exploration")
        footprint = row.get("public_new_free_voxel_keys")
        if not isinstance(footprint, list) or not footprint:
            raise ValueError("QD replay calibration requires a non-empty public new-free footprint")
        value = row.get("candidate_manifest_sha256")
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("QD replay calibration candidate manifest hash is invalid")
        int(value, 16)
        manifests.append(value)
    return tuple(manifests)


def _record_descriptors(payload: dict[str, Any]) -> tuple[tuple[str, RealisedQDDescriptor], ...]:
    qd = payload["realised_qd"]
    assert isinstance(qd, dict)
    rows = qd["admissions"]
    assert isinstance(rows, list)
    descriptors: list[tuple[str, RealisedQDDescriptor]] = []
    for row in rows:
        assert isinstance(row, dict)
        descriptor = row.get("descriptor")
        if not isinstance(descriptor, dict):
            raise ValueError("QD replay calibration outcome lacks a realised descriptor")
        descriptors.append(
            (
                str(row["candidate_manifest_sha256"]),
                RealisedQDDescriptor(
                    vertical_motion_ratio=descriptor.get("vertical_motion_ratio"),
                    team_spatial_dispersion=descriptor.get("team_spatial_dispersion"),
                    public_observation_complementarity=descriptor.get(
                        "public_observation_complementarity"
                    ),
                    schema_version=descriptor.get("schema_version", ""),
                ),
            )
        )
    return tuple(descriptors)


def _record_candidate_descriptor_features(
    payload: dict[str, Any],
) -> tuple[OutcomeQDFeatureVector, ...]:
    """Read the complete pre-registered outcome-only feature vector.

    A v4 descriptor by itself cannot tell whether spatial dispersion is
    redundant with complementarity.  Each worker record must therefore bind
    all candidate features into its execution outcome before train-only
    calibration can decide whether v4 is still defensible.
    """

    qd = payload["realised_qd"]
    assert isinstance(qd, dict)
    rows = qd["admissions"]
    assert isinstance(rows, list)
    features: list[OutcomeQDFeatureVector] = []
    for row in rows:
        assert isinstance(row, dict)
        raw_features = row.get("candidate_descriptor_features")
        if not isinstance(raw_features, dict):
            raise ValueError("QD replay calibration outcome lacks descriptor-family features")
        features.append(OutcomeQDFeatureVector.from_dict(raw_features))
    return tuple(features)


def _record_public_footprints(
    payload: dict[str, Any],
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    qd = payload["realised_qd"]
    assert isinstance(qd, dict)
    rows = qd["admissions"]
    assert isinstance(rows, list)
    footprints: list[tuple[tuple[int, int, int], ...]] = []
    for row in rows:
        assert isinstance(row, dict)
        raw_footprint = row.get("public_new_free_voxel_keys")
        if not isinstance(raw_footprint, list):
            raise ValueError("QD replay calibration outcome lacks a public new-free footprint")
        footprint: list[tuple[int, int, int]] = []
        for raw_key in raw_footprint:
            if not isinstance(raw_key, list) or len(raw_key) != 3:
                raise ValueError("QD replay calibration footprint key is invalid")
            if not all(isinstance(value, int) and not isinstance(value, bool) for value in raw_key):
                raise ValueError("QD replay calibration footprint coordinate is invalid")
            footprint.append((raw_key[0], raw_key[1], raw_key[2]))
        footprints.append(tuple(footprint))
    return tuple(footprints)


def _read_record(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"QD replay calibration record must be an object: {path}")
    return payload


def _write_new(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite QD replay calibration summary: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _calibration_admitted(
    replay_status: str, mode_contrast_status: str, descriptor_family_status: str
) -> bool:
    """Return whether both independent descriptor-calibration claims hold.

    Stable replays alone do not show that the emitter controls three distinct
    behaviour axes.  Conversely, a controlled contrast is not an archive if
    the same manifest lands in unrelated cells on a replay.  Keeping this
    conjunction in one named function prevents the summary status from
    silently dropping either condition.  The descriptor-family screen is a
    third requirement: the current v4 axes must not be retained merely
    because they themselves pass while a pre-registered alternative is less
    redundant on the same train outcomes.
    """

    return (
        replay_status == "QD_DESCRIPTOR_REPRODUCIBILITY_ADMITTED"
        and mode_contrast_status == "QD_CALIBRATION_MODE_CONTRAST_ADMITTED"
        and descriptor_family_status == "QD_DESCRIPTOR_FAMILY_CURRENT_ADMITTED"
    )


def run_calibration(
    plans: Sequence[CalibrationPlan], *, python: Path, runner: Path, dry_run: bool
) -> dict[str, object]:
    """Run plans, prove same-manifest replays, and report replay stability."""

    if dry_run:
        return {
            "schema_version": "hm3d-qd-replay-calibration-plan-v1",
            "status": "QD_REPLAY_CALIBRATION_PLAN_ONLY",
            "formal_result": False,
            "commands": [list(plan.command(python=python, runner=runner)) for plan in plans],
        }
    outputs: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for plan in plans:
        command = plan.command(python=python, runner=runner)
        completed = subprocess.run(command, cwd=ROOT, text=True, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"QD replay calibration worker failed: case={plan.case.case_id}, "
                f"mode={plan.intent_mode}, replay={plan.repetition + 1}"
            )
        payload = _read_record(plan.output_path)
        outputs.setdefault((plan.case.case_id, plan.intent_mode), []).append(payload)
    descriptors_by_manifest: dict[str, list[RealisedQDDescriptor]] = {}
    mode_labels: list[str] = []
    descriptors: list[RealisedQDDescriptor] = []
    candidate_features: list[OutcomeQDFeatureVector] = []
    candidate_feature_scenes: list[str] = []
    public_footprints: list[tuple[tuple[int, int, int], ...]] = []
    scene_ids: list[str] = []
    worker_hashes: list[str] = []
    for key, records in outputs.items():
        first_sequence = _record_manifests(records[0])
        if any(_record_manifests(record) != first_sequence for record in records[1:]):
            raise RuntimeError(
                "QD replay calibration did not reproduce the public candidate manifest sequence: "
                f"case={key[0]}, mode={key[1]}"
            )
        for record in records:
            recorded_hash = record.get("runtime_record_sha256")
            if not isinstance(recorded_hash, str) or len(recorded_hash) != 64:
                raise ValueError("QD replay calibration worker record hash is invalid")
            worker_hashes.append(recorded_hash)
            for manifest, descriptor in _record_descriptors(record):
                descriptors_by_manifest.setdefault(manifest, []).append(descriptor)
                mode_labels.append(key[1])
                descriptors.append(descriptor)
                scene_ids.append(key[0])
            record_features = _record_candidate_descriptor_features(record)
            candidate_features.extend(record_features)
            candidate_feature_scenes.extend(key[0] for _ in record_features)
            public_footprints.extend(_record_public_footprints(record))
    replay = audit_realised_qd_reproducibility(descriptors_by_manifest)
    mode_contrast = audit_realised_qd_calibration_mode_contrasts(
        mode_labels,
        descriptors,
        scene_ids,
    )
    descriptor_family_screen = audit_pre_registered_qd_descriptor_families(
        candidate_features, public_footprints, candidate_feature_scenes
    )
    admitted = _calibration_admitted(
        replay.status,
        mode_contrast.status,
        descriptor_family_screen.status,
    )
    return {
        "schema_version": "hm3d-qd-replay-calibration-v1",
        "status": "QD_REPLAY_CALIBRATION_COMPLETE"
        if admitted
        else "QD_REPLAY_CALIBRATION_NOT_ADMITTED",
        "formal_result": False,
        "claim_limit": (
            "Train-only calibration. Replay stability, independent axis control, and "
            "a non-redundant pre-registered current descriptor family are required; "
            "it establishes neither a QD task gain nor a P08 or P09 result."
        ),
        "worker_record_sha256s": sorted(worker_hashes),
        "replay_reproducibility_audit": replay.to_dict(),
        "mode_contrast_audit": mode_contrast.to_dict(),
        "descriptor_family_screen": descriptor_family_screen.to_dict(),
        "plan_sha256": canonical_sha256(
            [
                {
                    "case_id": plan.case.case_id,
                    "scene_id": plan.case.scene_id,
                    "random_key": plan.case.random_key,
                    "intent_mode": plan.intent_mode,
                    "repetition": plan.repetition,
                }
                for plan in plans
            ]
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--runner", type=Path, default=ROOT / "scripts" / "run_hm3d_p07_exploration_episode.py"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    cases = _load_cases(args.cases_json.expanduser().resolve())
    plans = build_calibration_plan(cases=cases, output_dir=output_dir)
    if args.dry_run:
        payload = run_calibration(plans, python=args.python, runner=args.runner, dry_run=True)
        print(json.dumps(payload))
        return 0
    if output_dir.exists():
        raise FileExistsError("QD replay calibration output directory must not already exist")
    output_dir.mkdir(parents=True)
    payload = run_calibration(plans, python=args.python, runner=args.runner, dry_run=False)
    _write_new(output_dir / "qd_replay_calibration_summary.json", payload)
    print(json.dumps({"status": payload["status"], "output_dir": str(output_dir)}, sort_keys=True))
    return 0 if payload["status"] == "QD_REPLAY_CALIBRATION_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
