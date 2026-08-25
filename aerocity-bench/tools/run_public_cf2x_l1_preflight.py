"""Run one public-policy CF2X L1 replay under a failure-accounting host guard.

The native Isaac executor owns flight, evaluation, and its normal failure
receipts.  This launcher owns the process boundary so an abnormal child exit
cannot disappear between the first progress receipt and the final replay
report.  It remains development/calibration-only and never grants formal
score eligibility.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from aerocity_bench.canonical import content_hash, file_hash, write_json_atomic  # noqa: E402
from aerocity_bench.cf2x_fleet_preflight_contract import (  # noqa: E402
    COMPLETE_CALIBRATION_PURPOSE,
    SHORT_PREFLIGHT_PURPOSE,
    validate_fleet_preflight_reports,
    validate_native_run_purpose,
)
from aerocity_bench.errors import HostGuardError  # noqa: E402
from aerocity_bench.host_guard import (  # noqa: E402
    isaac_host_lock,
    run_guarded_process,
    validate_host_guard_pass_receipt,
)
from aerocity_bench.ordinary_config import load_ordinary_config  # noqa: E402
from aerocity_bench.public_boundary import audit_public_layout  # noqa: E402

PUBLIC_METHODS = ("sweep-3d", "atlas-surface-inspector", "atlas-region-greedy")


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-root", type=Path, required=True)
    parser.add_argument("--release-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--cf2x-usd", type=Path, required=True)
    parser.add_argument("--method", choices=PUBLIC_METHODS, required=True)
    parser.add_argument("--isaac-python", type=Path, required=True)
    parser.add_argument("--isaaclab-root", type=Path, required=True)
    parser.add_argument(
        "--run-purpose",
        choices=(SHORT_PREFLIGHT_PURPOSE, COMPLETE_CALIBRATION_PURPOSE),
        default=SHORT_PREFLIGHT_PURPOSE,
    )
    parser.add_argument("--max-sim-time-s", type=float, default=12.0)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args(argv)


def _failure_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.failure.json")


def _progress_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}.progress.json")


def _resolved_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is absent: {resolved}")
    return resolved


def _validate_cf2x_asset(path: Path) -> Path:
    asset = _resolved_file(path, "CF2X USD")
    if asset.name.casefold() != "cf2x.usd" or "5_in_drone" in {
        part.casefold() for part in asset.parts
    }:
        raise ValueError("public L1 launcher requires assets/new/cf2x.usd")
    return asset


def _validated_inputs(args: argparse.Namespace) -> dict[str, Path]:
    layout = args.layout_root.resolve()
    if not layout.is_dir():
        raise FileNotFoundError(f"layout root is absent: {layout}")
    audit_public_layout(layout)
    config = _resolved_file(args.release_config, "release config")
    cf2x = _validate_cf2x_asset(args.cf2x_usd)
    isaac_python = _resolved_file(args.isaac_python, "Isaac Python")
    isaaclab_root = args.isaaclab_root.resolve()
    if not (isaaclab_root / "source" / "isaaclab").is_dir():
        raise ValueError("IsaacLab root lacks source/isaaclab")
    if args.timeout_s <= 0.0:
        raise ValueError("timeout-s must be positive")
    ordinary = load_ordinary_config(config)
    validate_native_run_purpose(
        purpose=str(args.run_purpose),
        execution_mode="public-policy",
        requested_sim_time_s=float(args.max_sim_time_s),
        frozen_episode_duration_s=float(ordinary.raw["execution_contract"]["episode"]["duration_s"]),
    )
    output = args.output.resolve()
    private_output = args.private_output.resolve()
    runtime = args.runtime_root.resolve()
    reserved = (output, private_output, _failure_path(output), _progress_path(output), runtime)
    existing = [str(path) for path in reserved if path.exists()]
    if existing:
        raise FileExistsError(f"public L1 launcher refuses to overwrite evidence: {existing}")
    if output == private_output:
        raise ValueError("public and private output paths must differ")
    return {
        "layout": layout,
        "config": config,
        "cf2x": cf2x,
        "isaac_python": isaac_python,
        "isaaclab_root": isaaclab_root,
        "output": output,
        "private_output": private_output,
        "runtime": runtime,
    }


def _evidence_binding(inputs: dict[str, Path], args: argparse.Namespace) -> str:
    layout = inputs["layout"]
    files = {
        "cityspec_sha256": layout / "scene_authority" / "cityspec.json",
        "task_spec_sha256": layout / "method_public" / "task_spec.json",
        "public_episode_sha256": layout / "method_public" / "episodes" / "episode-0000.json",
        "release_config_sha256": inputs["config"],
        "cf2x_usd_sha256": inputs["cf2x"],
        "launcher_source_sha256": Path(__file__).resolve(),
        "fleet_preflight_source_sha256": REPOSITORY_ROOT / "tools" / "cf2x_l1_fleet_preflight.py",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"public L1 input is absent: {missing}")
    return content_hash(
        {
            "scope": "development-only-public-cf2x-l1",
            "method": str(args.method),
            "run_purpose": str(args.run_purpose),
            "max_sim_time_s": float(args.max_sim_time_s),
            "files": {field: file_hash(path) for field, path in sorted(files.items())},
        }
    )


def _command(inputs: dict[str, Path], args: argparse.Namespace) -> list[str]:
    return [
        str(inputs["isaac_python"]),
        str(REPOSITORY_ROOT / "tools" / "cf2x_l1_fleet_preflight.py"),
        "--layout-root",
        str(inputs["layout"]),
        "--release-config",
        str(inputs["config"]),
        "--output",
        str(inputs["output"]),
        "--private-output",
        str(inputs["private_output"]),
        "--cf2x-usd",
        str(inputs["cf2x"]),
        "--execution-mode",
        "public-policy",
        "--method",
        str(args.method),
        "--run-purpose",
        str(args.run_purpose),
        "--max-sim-time-s",
        str(float(args.max_sim_time_s)),
        "--device",
        str(args.device),
        "--headless",
    ]


def _record_unexpected_child_exit(
    inputs: dict[str, Path], *, returncode: int, binding: str
) -> None:
    """Record the infrastructure failure only when the child left no receipt."""

    failure = _failure_path(inputs["output"])
    if failure.exists() or inputs["output"].exists():
        return
    report = inputs["runtime"] / "host_guard.json"
    write_json_atomic(
        failure,
        {
            "schema": "org.aerocity.bench.cf2x-l1-fleet-preflight-failure.v1",
            "status": "FAIL",
            "formal_score_eligible": False,
            "evidence_scope": "cf2x_public_policy_host_guard",
            "failure_stage": "child_process_exit_without_executor_receipt",
            "child_returncode": returncode,
            "evidence_binding": binding,
            "host_guard_report_sha256": file_hash(report) if report.is_file() else None,
        },
    )


def run(args: argparse.Namespace) -> int:
    inputs = _validated_inputs(args)
    binding = _evidence_binding(inputs, args)
    runtime = inputs["runtime"]
    runtime.mkdir(parents=True)
    environment = dict(os.environ)
    environment["AEROCITY_ISAACLAB_ROOT"] = str(inputs["isaaclab_root"])
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(SOURCE_ROOT) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    with isaac_host_lock():
        guarded = run_guarded_process(
            _command(inputs, args),
            cwd=REPOSITORY_ROOT,
            environment=environment,
            log_path=runtime / "isaac.log",
            report_path=runtime / "host_guard.json",
            timeout_s=float(args.timeout_s),
            evidence_binding=binding,
        )
    if guarded.returncode != 0:
        _record_unexpected_child_exit(inputs, returncode=guarded.returncode, binding=binding)
        raise RuntimeError(f"public CF2X preflight exited with {guarded.returncode}")
    validate_host_guard_pass_receipt(runtime / "host_guard.json", expected_evidence_binding=binding)
    validate_fleet_preflight_reports(inputs["output"], inputs["private_output"])
    print(f"PUBLIC_CF2X_L1_PREFLIGHT=PASS binding={binding}")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(_arguments(argv))
    except HostGuardError as exc:
        print(f"PUBLIC_CF2X_L1_PREFLIGHT=BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
