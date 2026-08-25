"""Repeat CF2X native preflights in isolated Isaac processes.

``--timeout-s`` is a wall-clock safety limit for one owned Isaac process.  It
does not shorten, simulate, or stand in for the benchmark's 300-second task
budget.  Every attempt is a small engineering receipt and is never a formal
benchmark score.
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import math
import os
import sys
from pathlib import Path

BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(BENCH_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT / "src"))

from aerocity_bench.canonical import content_hash, file_hash, read_json, write_json_atomic
from aerocity_bench.cf2x_contract import verify_local_cf2x_asset
from aerocity_bench.errors import HostGuardError
from aerocity_bench.host_guard import isaac_host_lock, run_guarded_process

_PROFILES = (
    "shared-hold",
    "shared-long-hold",
    "shared-long-lateral-hold",
    "shared-lateral-step",
    "shared-lateral-hold",
    "shared-altitude-hold",
    "shared-yaw-hold",
    "open-loop-hover",
    "open-loop-pitch-pulse",
    "open-loop-drop",
)
_RUNTIME_HASH_KEYS = (
    "preflight_script_sha256",
    "dynamics_contract_sha256",
    "cf2x_contract_sha256",
    "cf2x_native_sha256",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cf2x-usd", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--sample-every", type=int, default=15)
    parser.add_argument("--profile", choices=_PROFILES, default="shared-hold")
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=75.0,
        help="host wall-clock limit for one Isaac process; not a simulated episode budget",
    )
    parser.add_argument("--isaac-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--max-final-position-spread-m", type=float, default=1.0e-4)
    parser.add_argument("--max-final-velocity-spread-mps", type=float, default=1.0e-4)
    return parser


def _finite_vector(value: object, *, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must be a three-value list")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must be finite")
    return result  # type: ignore[return-value]


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"preflight evidence is missing {field}")
    if any(character not in "0123456789abcdef" for character in value.lower()):
        raise ValueError(f"preflight evidence has malformed {field}")
    return value


def _finite_scalar(value: object, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"preflight evidence has invalid {field}") from exc
    if not math.isfinite(result):
        raise ValueError(f"preflight evidence has invalid {field}")
    return result


def _validate_long_horizon_hover_evidence(report: dict[str, object]) -> None:
    runtime_quality = report.get("runtime_quality")
    if not isinstance(runtime_quality, dict):
        raise ValueError("long-hold preflight lacks runtime quality evidence")
    metrics = runtime_quality.get("long_horizon_hover")
    thresholds = runtime_quality.get("long_horizon_hover_candidate_thresholds")
    if not isinstance(metrics, dict) or not isinstance(thresholds, dict):
        raise ValueError("long-hold preflight lacks trend metrics or thresholds")
    if thresholds.get("schema") != "org.aerocity.bench.long-horizon-hover-thresholds.v1":
        raise ValueError("long-hold preflight has an unexpected threshold schema")
    if thresholds.get("status") != "candidate_preflight_only":
        raise ValueError("long-hold thresholds improperly claim formal status")
    duration = _finite_scalar(metrics.get("simulated_duration_s"), field="long hover duration")
    minimum_duration = _finite_scalar(
        thresholds.get("minimum_duration_s"), field="long hover minimum duration"
    )
    final_error = _finite_scalar(
        metrics.get("final_altitude_error_m"), field="long hover final altitude error"
    )
    maximum_final_error = _finite_scalar(
        thresholds.get("max_abs_final_altitude_error_m"),
        field="long hover maximum final altitude error",
    )
    terminal_velocity = _finite_scalar(
        metrics.get("terminal_vertical_velocity_mps"),
        field="long hover terminal vertical velocity",
    )
    maximum_terminal_velocity = _finite_scalar(
        thresholds.get("max_abs_terminal_vertical_velocity_mps"),
        field="long hover maximum terminal vertical velocity",
    )
    slope = _finite_scalar(
        metrics.get("late_altitude_slope_mps"), field="long hover late altitude slope"
    )
    maximum_slope = _finite_scalar(
        thresholds.get("max_abs_late_altitude_slope_mps"),
        field="long hover maximum late altitude slope",
    )
    if duration < minimum_duration:
        raise ValueError("long-hold preflight duration is below its candidate gate")
    if abs(final_error) > maximum_final_error:
        raise ValueError("long-hold preflight exceeds its final altitude bound")
    max_altitude_error = _finite_scalar(
        metrics.get("max_abs_altitude_error_m"), field="long hover maximum altitude error"
    )
    maximum_altitude_error = _finite_scalar(
        thresholds.get("max_abs_altitude_error_m"),
        field="long hover permitted maximum altitude error",
    )
    if abs(terminal_velocity) > maximum_terminal_velocity:
        raise ValueError("long-hold preflight exceeds its terminal vertical velocity bound")
    if max_altitude_error > maximum_altitude_error:
        raise ValueError("long-hold preflight exceeds its maximum altitude bound")
    if abs(slope) > maximum_slope:
        raise ValueError("long-hold preflight exceeds its late altitude slope bound")
    max_contact_force = _finite_scalar(
        runtime_quality.get("max_contact_force_n"), field="long hover maximum contact force"
    )
    permitted_contact_force = _finite_scalar(
        thresholds.get("max_contact_force_n"),
        field="long hover permitted contact force",
    )
    if max_contact_force > permitted_contact_force:
        raise ValueError("long-hold preflight records ground contact")


def validate_preflight_report(
    path: Path,
    *,
    expected_profile: str,
    expected_steps: int,
    expected_asset: object | None = None,
) -> dict[str, object]:
    """Fail closed unless a CF2X native-preflight receipt is complete."""

    report = read_json(path)
    expected_hash = report.pop("preflight_hash", None)
    if not isinstance(expected_hash, str) or content_hash(report) != expected_hash:
        raise ValueError("preflight report hash mismatch")
    if report.get("schema") != "org.aerocity.bench.quadrotor-physx-preflight.v2":
        raise ValueError("unexpected preflight report schema")
    if report.get("formal") is not False or report.get("formal_score_eligible") is not False:
        raise ValueError("non-formal preflight attempted to claim formal eligibility")
    if report.get("vehicle_execution_model") != (
        "cf2x_multirotor_per_rotor_thrust_geometry_allocated_root_wrench_physx"
    ):
        raise ValueError("preflight did not identify the CF2X multirotor executor")
    if report.get("steps") != expected_steps:
        raise ValueError("preflight step count differs from the batch contract")
    controller = report.get("controller")
    if not isinstance(controller, dict) or controller.get("profile") != expected_profile:
        raise ValueError("preflight profile differs from the batch contract")
    checks = report.get("checks")
    checks_complete = isinstance(checks, dict) and bool(checks)
    if not checks_complete or any(value is not True for value in checks.values()):
        raise ValueError("preflight contains a failed or incomplete native check")
    if expected_profile in {"shared-long-hold", "shared-long-lateral-hold"}:
        _validate_long_horizon_hover_evidence(report)
    multirotor = report.get("multirotor")
    if not isinstance(multirotor, dict):
        raise ValueError("preflight lacks multirotor instrumentation")
    if multirotor.get("contact_evidence") != "IsaacLab ContactSensor.net_forces_w":
        raise ValueError("preflight lacks native ContactSensor evidence")
    if multirotor.get("direct_root_state_writes_during_loop") is not False:
        raise ValueError("preflight did not prohibit root-state writes during flight")
    if multirotor.get("wrench_application_model") != (
        "derived_geometry_allocation_to_root_body_physx"
    ):
        raise ValueError("preflight lacks the reviewed geometry-allocation execution model")
    if multirotor.get("prop_link_forces_applied_directly") is not False:
        raise ValueError("preflight overclaims direct prop-link force application")
    final = report.get("final_state")
    if not isinstance(final, dict):
        raise ValueError("preflight lacks a final measured state")
    position = _finite_vector(final.get("position_w_m"), name="final position")
    velocity = _finite_vector(final.get("linear_velocity_w_mps"), name="final velocity")
    applied = final.get("applied_rotor_thrust_n")
    if not isinstance(applied, list) or len(applied) != 4:
        raise ValueError("preflight lacks four applied rotor thrust values")
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in applied):
        raise ValueError("preflight applied rotor thrust is invalid")
    runtime = report.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("preflight lacks runtime provenance")
    runtime_provenance = {key: _sha256(runtime.get(key), field=key) for key in _RUNTIME_HASH_KEYS}
    asset = report.get("asset")
    if not isinstance(asset, dict):
        raise ValueError("preflight lacks CF2X asset provenance")
    root_digest = _sha256(asset.get("usd_sha256"), field="asset.usd_sha256")
    schema_digest = _sha256(asset.get("schema_sha256"), field="asset.schema_sha256")
    if asset.get("asset_kind") != "cf2x_local_runtime_dependency":
        raise ValueError("preflight asset is not the local-only CF2X dependency")
    if expected_asset is not None:
        expected_root = getattr(expected_asset, "usd_sha256", None)
        expected_schema = getattr(expected_asset, "schema_sha256", None)
        if root_digest != expected_root or schema_digest != expected_schema:
            raise ValueError("preflight CF2X asset digest differs from the batch contract")
    return {
        "preflight_path": str(path.resolve()),
        "preflight_file_sha256": file_hash(path),
        "preflight_hash": expected_hash,
        "final_position_w_m": list(position),
        "final_linear_velocity_w_mps": list(velocity),
        "cf2x_asset": {"usd_sha256": root_digest, "schema_sha256": schema_digest},
        "runtime_provenance": runtime_provenance,
    }


def _component_spread(samples: list[dict[str, object]], key: str) -> float:
    vectors = [item[key] for item in samples]
    return max(
        abs(float(left[axis]) - float(right[axis]))
        for left in vectors
        for right in vectors
        for axis in range(3)
    )


def _run(args: argparse.Namespace) -> dict[str, object]:
    if not 1 <= args.runs <= 20:
        raise ValueError("runs must be between 1 and 20")
    if args.steps <= 0 or args.sample_every <= 0:
        raise ValueError("steps and sample-every must be positive")
    if args.timeout_s <= 0.0:
        raise ValueError("timeout-s must be positive")
    if args.max_final_position_spread_m < 0.0 or args.max_final_velocity_spread_mps < 0.0:
        raise ValueError("repeatability tolerances must be non-negative")
    python = args.isaac_python.resolve()
    if not python.is_file():
        raise FileNotFoundError(f"Isaac Python executable is missing: {python}")
    asset = verify_local_cf2x_asset(args.cf2x_usd)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"batch output already exists: {output}")
    output.mkdir(parents=True)
    preflight = BENCH_ROOT / "tools" / "quadrotor_physics_preflight.py"
    attempts: list[dict[str, object]] = []
    with isaac_host_lock():
        for index in range(1, args.runs + 1):
            attempt = output / f"attempt-{index:02d}"
            attempt.mkdir()
            receipt = attempt / "preflight.json"
            command = [
                str(python), str(preflight), "--output", str(receipt),
                "--cf2x-usd", str(asset.usd_path), "--steps", str(args.steps),
                "--sample-every", str(args.sample_every), "--profile", args.profile, "--headless",
            ]
            try:
                guarded = run_guarded_process(
                    command,
                    cwd=BENCH_ROOT,
                    environment=dict(os.environ),
                    log_path=attempt / "isaac.log",
                    report_path=attempt / "host_guard.json",
                    timeout_s=args.timeout_s,
                )
                if guarded.returncode != 0:
                    raise RuntimeError(f"Isaac preflight exited with {guarded.returncode}")
                verified = validate_preflight_report(
                    receipt,
                    expected_profile=args.profile,
                    expected_steps=args.steps,
                    expected_asset=asset,
                )
                attempts.append({
                    "attempt": attempt.name, "status": "PASS",
                    "guard_elapsed_s": guarded.elapsed_s,
                    "guard_maximum_commit_fraction": guarded.maximum_commit_fraction,
                    "verified": verified,
                })
            except (HostGuardError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
                attempts.append({
                    "attempt": attempt.name, "status": "FAIL",
                    "error_type": type(exc).__name__, "error": str(exc)[-4000:],
                })
                break
    verified = [item["verified"] for item in attempts if item["status"] == "PASS"]
    position_spread = (
        _component_spread(verified, "final_position_w_m")
        if len(verified) >= 2
        else None
    )
    velocity_spread = (
        _component_spread(verified, "final_linear_velocity_w_mps")
        if len(verified) >= 2
        else None
    )
    status = "PASS"
    if len(verified) != args.runs:
        status = "FAIL"
    if position_spread is not None and position_spread > args.max_final_position_spread_m:
        status = "FAIL"
    if velocity_spread is not None and velocity_spread > args.max_final_velocity_spread_mps:
        status = "FAIL"
    report: dict[str, object] = {
        "schema": "org.aerocity.bench.quadrotor-physx-preflight-batch.v2",
        "status": status,
        "formal_score_eligible": False,
        "purpose": "fresh_process_cf2x_native_dynamics_repeatability_diagnostic_only",
        "host_timeout_semantics": "wall_clock_per_owned_isaac_process_not_simulated_episode_budget",
        "cf2x_asset": asset.fingerprint_payload(),
        "profile": args.profile, "steps": args.steps,
        "runs_requested": args.runs, "runs_verified": len(verified),
        "max_final_position_spread_m": position_spread,
        "max_final_velocity_spread_mps": velocity_spread,
        "position_spread_tolerance_m": args.max_final_position_spread_m,
        "velocity_spread_tolerance_mps": args.max_final_velocity_spread_mps,
        "attempts": attempts,
    }
    report["batch_hash"] = content_hash(report)
    write_json_atomic(output / "batch_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = _run(args)
    except (HostGuardError, OSError, RuntimeError, TimeoutError, ValueError) as exc:
        print(f"quadrotor preflight batch failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
