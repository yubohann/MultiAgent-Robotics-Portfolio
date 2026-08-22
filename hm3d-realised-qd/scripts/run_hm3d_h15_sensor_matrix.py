"""Run every H15 sensor cell in a fresh Isaac process, then assemble P06.

The matrix runner deliberately executes serially.  Multiple concurrent Kit
instances contend for one GPU, producing an invalid throughput comparison.
Completed rows are immutable evidence; use ``--resume`` only to continue a
previously interrupted matrix without rerunning a measured cell.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from assemble_hm3d_h15_sensor_pilot import assemble  # noqa: E402

from aerocity_method.contracts import FORMAL_FLEET_SIZE  # noqa: E402
from aerocity_method.contracts.io import write_json_atomic  # noqa: E402
from aerocity_method.runtime.sensors import FORMAL_H15_SENSOR_PILOT_MODES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac-python", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--collision-usd", type=Path, required=True)
    parser.add_argument("--receiver-positions-json", type=Path, required=True)
    parser.add_argument("--rows-dir", type=Path, required=True)
    parser.add_argument("--p06-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--ledger-output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--physics-dt-s", type=float, default=1.0 / 120.0)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _resolve_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def _row_path(rows_dir: Path, mode: str) -> Path:
    return rows_dir / f"row_N{FORMAL_FLEET_SIZE}_{mode}_v3.json"


def main() -> int:
    args = parse_args()
    if args.steps < 30 or args.physics_dt_s <= 0.0 or args.seed < 0:
        raise ValueError("invalid H15 steps, physics dt or seed")
    python = _resolve_file(args.isaac_python, "Isaac Python")
    collision_usd = _resolve_file(args.collision_usd, "collision USD")
    positions = _resolve_file(args.receiver_positions_json, "receiver position evidence")
    runner = ROOT / "scripts" / "run_hm3d_h15_sensor_pilot.py"
    rows_dir = args.rows_dir.expanduser().resolve()
    rows_dir.mkdir(parents=True, exist_ok=True)
    p06_output = args.p06_output.expanduser().resolve()
    audit_output = args.audit_output.expanduser().resolve()
    ledger_output = args.ledger_output.expanduser().resolve()
    if p06_output.exists() or audit_output.exists() or ledger_output.exists():
        raise FileExistsError("refusing to overwrite H15 P06, audit, or ledger evidence")

    plan = tuple(FORMAL_H15_SENSOR_PILOT_MODES)
    existing = [
        path
        for mode in plan
        if (path := _row_path(rows_dir, mode)).exists()
    ]
    if existing and not args.resume:
        raise FileExistsError(
            "formal H15 rows already exist; inspect them or pass --resume to continue"
        )
    ledger_rows: list[dict[str, object]] = []
    for mode in plan:
        output = _row_path(rows_dir, mode)
        if output.exists():
            ledger_rows.append(
                {
                    "fleet_size": FORMAL_FLEET_SIZE,
                    "mode": mode,
                    "status": "reused_immutable_row",
                    "output": str(output),
                }
            )
            continue
        command = [
            str(python),
            str(runner),
            "--scene-id",
            args.scene_id,
            "--collision-usd",
            str(collision_usd),
            "--receiver-positions-json",
            str(positions),
            "--output",
            str(output),
            "--mode",
            mode,
            "--steps",
            str(args.steps),
            "--physics-dt-s",
            str(args.physics_dt_s),
            "--seed",
            str(args.seed),
            "--headless",
            "--device",
            args.device,
        ]
        completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        row = {
            "fleet_size": FORMAL_FLEET_SIZE,
            "mode": mode,
            "status": "completed" if completed.returncode == 0 and output.is_file() else "failed",
            "returncode": completed.returncode,
            "output": str(output),
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }
        ledger_rows.append(row)
        if row["status"] != "completed":
            write_json_atomic(
                ledger_output,
                {
                    "schema_version": "hm3d-h15-matrix-ledger-v3",
                    "status": "H15_MATRIX_INTERRUPTED",
                    "completed_rows": ledger_rows,
                    "next_required_cell": {"fleet_size": FORMAL_FLEET_SIZE, "mode": mode},
                },
            )
            print(
                json.dumps(
                    {
                        "status": "H15_MATRIX_INTERRUPTED",
                        "fleet_size": FORMAL_FLEET_SIZE,
                        "mode": mode,
                    },
                    sort_keys=True,
                )
            )
            return 2

    ledger = {
        "schema_version": "hm3d-h15-matrix-ledger-v3",
        "status": "H15_MATRIX_COMPLETE",
        "completed_rows": ledger_rows,
        "p06_output": str(p06_output),
        "audit_output": str(audit_output),
    }
    write_json_atomic(ledger_output, ledger)
    try:
        payload, audit = assemble(
            rows_dir,
            "sparse_range_3d",
            matrix_ledger=ledger,
            matrix_ledger_path=ledger_output,
        )
        write_json_atomic(audit_output, audit)
        write_json_atomic(p06_output, payload)
    except BaseException:
        audit_output.unlink(missing_ok=True)
        p06_output.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {"status": "H15_MATRIX_COMPLETE", "p06_output": str(p06_output)},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
