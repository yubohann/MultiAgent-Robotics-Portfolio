"""Launch one source-locked external G2-I CF2X replay under the host guard.

This is deliberately a development/calibration launcher.  It binds one public
layout, one local external-process manifest, and one CF2X asset, then lets the
existing fleet executor own all PhysX, safety, evaluator, and receipt logic.
It neither opens formal test data nor turns a translated candidate into a
native upstream or multi-UAV claim.
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

from aerocity_bench.adapters import load_external_l1_adapter_manifest  # noqa: E402
from aerocity_bench.canonical import content_hash, file_hash  # noqa: E402
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


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout-root", type=Path, required=True)
    parser.add_argument("--release-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--cf2x-usd", type=Path, required=True)
    parser.add_argument("--external-adapter-manifest", type=Path, required=True)
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
    cf2x = _resolved_file(path, "CF2X USD")
    if cf2x.name.casefold() != "cf2x.usd" or "5_in_drone" in {
        part.casefold() for part in cf2x.parts
    }:
        raise ValueError("external L1 launcher requires assets/new/cf2x.usd")
    return cf2x


def _validated_inputs(args: argparse.Namespace) -> dict[str, Path]:
    layout = args.layout_root.resolve()
    if not layout.is_dir():
        raise FileNotFoundError(f"layout root is absent: {layout}")
    audit_public_layout(layout)
    config = _resolved_file(args.release_config, "release config")
    cf2x = _validate_cf2x_asset(args.cf2x_usd)
    manifest = _resolved_file(args.external_adapter_manifest, "external adapter manifest")
    load_external_l1_adapter_manifest(manifest)
    isaac_python = _resolved_file(args.isaac_python, "Isaac Python")
    isaaclab_root = args.isaaclab_root.resolve()
    if not (isaaclab_root / "source" / "isaaclab").is_dir():
        raise ValueError("IsaacLab root lacks source/isaaclab")
    if args.timeout_s <= 0.0:
        raise ValueError("timeout-s must be positive")
    ordinary = load_ordinary_config(config)
    validate_native_run_purpose(
        purpose=str(args.run_purpose),
        execution_mode="external-process-policy",
        requested_sim_time_s=float(args.max_sim_time_s),
        frozen_episode_duration_s=float(ordinary.raw["execution_contract"]["episode"]["duration_s"]),
    )
    output = args.output.resolve()
    private_output = args.private_output.resolve()
    runtime = args.runtime_root.resolve()
    reserved = (output, private_output, _failure_path(output), _progress_path(output), runtime)
    existing = [str(path) for path in reserved if path.exists()]
    if existing:
        raise FileExistsError(f"external L1 launcher refuses to overwrite evidence: {existing}")
    if output == private_output:
        raise ValueError("public and private output paths must differ")
    return {
        "layout": layout,
        "config": config,
        "cf2x": cf2x,
        "manifest": manifest,
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
        "external_adapter_manifest_sha256": inputs["manifest"],
        "launcher_source_sha256": Path(__file__).resolve(),
        "fleet_preflight_source_sha256": REPOSITORY_ROOT / "tools" / "cf2x_l1_fleet_preflight.py",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"external L1 public input is absent: {missing}")
    return content_hash(
        {
            "scope": "development-only-source-locked-external-cf2x-l1",
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
        "external-process-policy",
        "--external-adapter-manifest",
        str(inputs["manifest"]),
        "--run-purpose",
        str(args.run_purpose),
        "--max-sim-time-s",
        str(float(args.max_sim_time_s)),
        "--device",
        str(args.device),
        "--headless",
    ]


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
    command = _command(inputs, args)
    with isaac_host_lock():
        guarded = run_guarded_process(
            command,
            cwd=REPOSITORY_ROOT,
            environment=environment,
            log_path=runtime / "isaac.log",
            report_path=runtime / "host_guard.json",
            timeout_s=float(args.timeout_s),
            evidence_binding=binding,
        )
    if guarded.returncode != 0:
        raise RuntimeError(f"external CF2X preflight exited with {guarded.returncode}")
    validate_host_guard_pass_receipt(runtime / "host_guard.json", expected_evidence_binding=binding)
    validate_fleet_preflight_reports(inputs["output"], inputs["private_output"])
    print(f"EXTERNAL_CF2X_L1_PREFLIGHT=PASS binding={binding}")
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(_arguments(argv))
    except HostGuardError as exc:
        print(f"EXTERNAL_CF2X_L1_PREFLIGHT=BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
