"""Run every H15 sensor cell in a fresh Isaac process and index the rows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

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
    parser.add_argument("--summary-output", type=Path, required=True)
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
    summary_output = args.summary_output.expanduser().resolve()
    ledger_output = args.ledger_output.expanduser().resolve()
    if summary_output.exists() or ledger_output.exists():
        raise FileExistsError("refusing to overwrite H15 summary or ledger")

    plan = tuple(FORMAL_H15_SENSOR_PILOT_MODES)
    existing = [path for mode in plan if (path := _row_path(rows_dir, mode)).exists()]
    if existing and not args.resume:
        raise FileExistsError(
            "H15 rows already exist; inspect them or pass --resume to continue"
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
        "summary_output": str(summary_output),
    }
    write_json_atomic(ledger_output, ledger)
    write_json_atomic(
        summary_output,
        {
            "schema_version": "hm3d-h15-sensor-summary-v1",
            "status": "H15_SENSOR_MATRIX_COMPLETE",
            "mode_rows": [
                {"mode": mode, "output": str(_row_path(rows_dir, mode))} for mode in plan
            ],
        },
    )
    print(json.dumps({"status": "H15_MATRIX_COMPLETE"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
