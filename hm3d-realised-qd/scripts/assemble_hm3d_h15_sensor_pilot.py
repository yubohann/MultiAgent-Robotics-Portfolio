"""Assemble a complete, source-consistent H15 pilot from isolated Isaac rows.

Each H15 matrix cell must be measured in its own Isaac process.  This command
does not manufacture measurements: it rejects incomplete, duplicate, mixed-
source, synthetic, or smoke rows, then serializes the exact P06 payload used
by the formal preflight auditor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerocity_method.contracts import FORMAL_FLEET_SIZE  # noqa: E402
from aerocity_method.contracts.io import (  # noqa: E402
    canonical_sha256,
    read_json_object,
    write_json_atomic,
)
from aerocity_method.evaluation.hm3d_preflight import (  # noqa: E402
    FORMAL_MATRIX_METHODS,
    MECHANISM_VARIANTS,
)
from aerocity_method.runtime.sensors import (  # noqa: E402
    FORMAL_H15_SENSOR_PILOT_MODES,
    SensorThroughputRecord,
    audit_sensor_throughput_pilot,
)

ROW_SCHEMA_VERSION = "hm3d-h15-sensor-row-v3"
ROW_STATUS = "H15_REAL_SENSOR_ROW_COMPLETE"
ASSEMBLER_VERSION = "hm3d-h15-assembler-v3"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _exact(payload: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            f"{label} fields mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _load_row(path: Path) -> tuple[dict[str, Any], SensorThroughputRecord]:
    payload = read_json_object(path)
    _exact(
        payload,
        {
            "schema_version",
            "status",
            "synthetic",
            "evidence_class",
            "runtime_run_id",
            "runtime_command_sha256",
            "runner_version",
            "source_observation_binding",
            "selection_partition",
            "scene_id",
            "collision_usd_sha256",
            "receiver_position_source_sha256",
            "pilot_claim_limit",
            "fleet_size",
            "mode",
            "record",
        },
        f"H15 row {path.name}",
    )
    if payload["schema_version"] != ROW_SCHEMA_VERSION or payload["status"] != ROW_STATUS:
        raise ValueError(f"{path.name} is not a completed H15 worker row")
    if payload["synthetic"] is not False or payload["evidence_class"] != "real_runtime_measurement":
        raise ValueError(f"{path.name} is not real runtime evidence")
    if payload["source_observation_binding"] is not True:
        raise ValueError(f"{path.name} does not preserve source-observation binding")
    if payload["selection_partition"] not in {"train", "validation"}:
        raise ValueError(f"{path.name} uses an invalid selection partition")
    for field in (
        "runtime_run_id",
        "runtime_command_sha256",
        "runner_version",
        "scene_id",
        "collision_usd_sha256",
        "receiver_position_source_sha256",
    ):
        if not isinstance(payload[field], str) or not payload[field]:
            raise ValueError(f"{path.name} has an invalid {field}")
    record_payload = payload["record"]
    if not isinstance(record_payload, dict):
        raise ValueError(f"{path.name} has no sensor record")
    record = SensorThroughputRecord.from_dict(record_payload)
    if record.scene_id != payload["scene_id"]:
        raise ValueError(f"{path.name} scene identity differs from its record")
    if record.fleet_size != payload["fleet_size"] or record.profile.mode != payload["mode"]:
        raise ValueError(f"{path.name} fleet/mode differs from its record")
    return payload, record


def _validate_matrix_ledger(ledger: dict[str, Any]) -> dict[str, Any]:
    """Bind P06 to the parent-observed exit status of every worker process."""

    if ledger.get("schema_version") != "hm3d-h15-matrix-ledger-v3":
        raise ValueError("H15 matrix ledger schema mismatch")
    if ledger.get("status") != "H15_MATRIX_COMPLETE":
        raise ValueError("H15 matrix ledger is not complete")
    rows = ledger.get("completed_rows")
    expected = {(FORMAL_FLEET_SIZE, mode) for mode in FORMAL_H15_SENSOR_PILOT_MODES}
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("H15 matrix ledger lacks the complete camera-free matrix")
    actual = {
        (row.get("fleet_size"), row.get("mode"))
        for row in rows
        if isinstance(row, dict) and row.get("status") == "completed"
    }
    if actual != expected:
        raise ValueError("H15 matrix ledger contains a missing or non-clean worker exit")
    return ledger


def assemble(
    rows_dir: Path,
    selected_mode: str,
    *,
    matrix_ledger: dict[str, Any] | None = None,
    matrix_ledger_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate every isolated worker row and return the P06 payload + audit."""

    if selected_mode not in FORMAL_H15_SENSOR_PILOT_MODES or selected_mode == "physics_only":
        raise ValueError("selected mode must be a measured non-physics sensor profile")
    row_paths = tuple(sorted(rows_dir.glob("row_*_v3.json")))
    if not row_paths:
        raise ValueError("no formal H15 rows found; smoke rows are intentionally ignored")
    loaded = [(*_load_row(path), path) for path in row_paths]
    metadata = [payload for payload, _, _ in loaded]
    records = tuple(record for _, record, _ in loaded)
    pilot = audit_sensor_throughput_pilot(records, modes=FORMAL_H15_SENSOR_PILOT_MODES)
    if pilot["status"] != "PASS":
        raise ValueError(
            f"H15 pilot does not form the complete camera-free matrix: {pilot['reasons']}"
        )

    shared = {
        "selection_partition": {str(row["selection_partition"]) for row in metadata},
        "scene_id": {str(row["scene_id"]) for row in metadata},
        "collision_usd_sha256": {str(row["collision_usd_sha256"]) for row in metadata},
        "receiver_position_source_sha256": {
            str(row["receiver_position_source_sha256"]) for row in metadata
        },
        "comparison_id": {record.comparison_id for record in records},
        "episode_id": {record.episode_id for record in records},
        "physics_dt_s": {record.physics_dt_s for record in records},
    }
    mismatched = sorted(name for name, values in shared.items() if len(values) != 1)
    if mismatched:
        raise ValueError(f"H15 worker rows are not source-consistent: {mismatched}")
    by_mode = [record for record in records if record.profile.mode == selected_mode]
    if {record.fleet_size for record in by_mode} != {FORMAL_FLEET_SIZE}:
        raise ValueError("selected profile was not measured for the formal four-CF2X fleet")
    selected_profiles = {record.profile.entitlement_hash: record.profile for record in by_mode}
    if len(selected_profiles) != 1:
        raise ValueError("selected profile changed across formal-fleet records")
    selected = next(iter(selected_profiles.values()))
    expected_methods = set(FORMAL_MATRIX_METHODS) | set(MECHANISM_VARIANTS)
    entitlements = [
        {"method_id": method_id, "profile_hash": selected.entitlement_hash}
        for method_id in sorted(expected_methods)
    ]
    source_hashes = {
        key: next(iter(values))
        for key, values in shared.items()
        if key
        not in {
            "selection_partition",
            "scene_id",
            "comparison_id",
            "episode_id",
            "physics_dt_s",
        }
    }
    row_hashes = [
        {"path": str(path.resolve()), "sha256": _sha256_file(path)} for _, _, path in loaded
    ]
    audit = {
        "schema_version": "hm3d-h15-sensor-assembly-audit-v1",
        "assembler_version": ASSEMBLER_VERSION,
        "status": "H15_ASSEMBLY_PASS",
        "rows_dir": str(rows_dir.resolve()),
        "matrix": pilot,
        "selection_partition": next(iter(shared["selection_partition"])),
        "scene_id": next(iter(shared["scene_id"])),
        "comparison_id": next(iter(shared["comparison_id"])),
        "episode_id": next(iter(shared["episode_id"])),
        "physics_dt_s": next(iter(shared["physics_dt_s"])),
        "source_hashes": source_hashes,
        "selected_profile": selected.to_dict(),
        "selected_profile_hash": selected.entitlement_hash,
        "row_files": row_hashes,
    }
    if matrix_ledger is not None:
        _validate_matrix_ledger(matrix_ledger)
        if matrix_ledger_path is None or not matrix_ledger_path.is_file():
            raise ValueError("H15 complete matrix ledger needs an immutable file path")
        audit["matrix_ledger"] = {
            "path": str(matrix_ledger_path.resolve()),
            "sha256": _sha256_file(matrix_ledger_path),
            "status": matrix_ledger["status"],
        }
    command_hash = canonical_sha256(
        {
            "assembler_version": ASSEMBLER_VERSION,
            "selected_mode": selected_mode,
            "row_files": row_hashes,
        }
    )
    p06_payload = {
        "evidence_class": "real_runtime_measurement",
        "runtime_run_id": f"hm3d-h15-assembled-{uuid.uuid4().hex}",
        "runtime_command_sha256": command_hash,
        "source_observation_binding": True,
        "selection_partition": next(iter(shared["selection_partition"])),
        "records": [
            record.to_dict()
            for record in sorted(records, key=lambda row: (row.fleet_size, row.profile.mode))
        ],
        "selected_profile": selected.to_dict(),
        "entitlements": entitlements,
    }
    return p06_payload, audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New P06 payload path")
    parser.add_argument(
        "--audit-output", type=Path, required=True, help="New row-provenance audit path"
    )
    parser.add_argument("--matrix-ledger", type=Path, required=True)
    parser.add_argument(
        "--selected-mode", default="sparse_range_3d", choices=FORMAL_H15_SENSOR_PILOT_MODES
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows_dir = args.rows_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    audit_output = args.audit_output.expanduser().resolve()
    ledger_path = args.matrix_ledger.expanduser().resolve()
    if not rows_dir.is_dir():
        raise FileNotFoundError(f"H15 rows directory is missing: {rows_dir}")
    if not ledger_path.is_file():
        raise FileNotFoundError(f"H15 matrix ledger is missing: {ledger_path}")
    if output.exists() or audit_output.exists():
        raise FileExistsError("refusing to overwrite H15 evidence or its audit")
    payload, audit = assemble(
        rows_dir,
        args.selected_mode,
        matrix_ledger=read_json_object(ledger_path),
        matrix_ledger_path=ledger_path,
    )
    write_json_atomic(audit_output, audit)
    try:
        write_json_atomic(output, payload)
    except BaseException:
        audit_output.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {"status": audit["status"], "output": str(output), "audit": str(audit_output)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
