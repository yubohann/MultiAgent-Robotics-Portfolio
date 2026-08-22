"""Run the locked FUEL planner's isolated ROS 1 interface smoke test.

The tool never launches a CF2X, never supplies target/evaluator state, and
never starts FUEL's trajectory server.  It only validates that the separately
licensed upstream planner consumes public-style ROS inputs and emits a B-spline
route.  A passing result is not a G1-U/G2-I method integration or score.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__:
    from .build_fuel_container import load_lock, verify_source
else:  # Supports direct ``python tools/run_fuel_ros_smoke.py`` execution.
    from build_fuel_container import load_lock, verify_source

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = REPOSITORY_ROOT / "external" / "fuel" / "aerocity_fuel_ros_smoke.py"
DEFAULT_IMAGE = "aerocity-external-fuel:662dd23c7b52"


def _docker_bind_path(path: Path) -> str:
    """Map a resolved Windows path through WSL's standard fixed-drive mount.

    ``wslpath`` can print a localized diagnostic before its result on hosts
    with a degraded WSL NAT service.  That diagnostic is outside the tool's
    control and previously made a valid Docker invocation fail during strict
    Unicode decoding.  The benchmark only supports the standard `/mnt/<drive>`
    mapping for this local smoke, so derive that path without parsing WSL
    process output.
    """

    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if len(drive) != 1 or not drive.isalpha():
        raise ValueError("FUEL ROS smoke requires an absolute Windows drive path")
    relative_parts = resolved.parts[1:]
    return "/mnt/" + drive + "/" + "/".join(relative_parts)


def _docker_command(
    *, image: str, script_path: Path, distribution: str, duration_s: float
) -> list[str]:
    wsl_script = _docker_bind_path(script_path)
    return [
        "wsl.exe",
        "-d",
        distribution,
        "--",
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,mode=1777",
        "--mount",
        f"type=bind,src={wsl_script},dst=/opt/aerocity/aerocity_fuel_ros_smoke.py,readonly",
        "--env",
        "ROS_IP=127.0.0.1",
        "--env",
        "ROS_MASTER_URI=http://127.0.0.1:11311",
        "--env",
        "ROS_HOME=/tmp/aerocity-fuel-ros-home",
        "--env",
        "ROS_LOG_DIR=/tmp/aerocity-fuel-ros-log",
        image,
        "bash",
        "-lc",
        (
            "source /opt/ros/noetic/setup.bash && source /opt/fuel/ws/devel/setup.bash "
            "&& python3 /opt/aerocity/aerocity_fuel_ros_smoke.py --duration-s "
            + str(duration_s)
        ),
    ]


def _result_from_output(output: str) -> dict[str, Any] | None:
    for line in reversed(output.splitlines()):
        try:
            node = json.loads(line)
        except json.JSONDecodeError:
            continue
        if node.get("schema") == "org.aerocity.bench.fuel-ros-smoke.v1":
            return node
    return None


def _emit_report(report: dict[str, Any], output: Path | None) -> None:
    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="clean locked FUEL checkout")
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--distribution", default="Ubuntu-22.04")
    parser.add_argument("--duration-s", type=float, default=12.0)
    parser.add_argument("--output", type=Path, help="write the machine-readable smoke report")
    arguments = parser.parse_args(argv)

    try:
        if arguments.duration_s <= 5.0:
            raise ValueError("--duration-s must exceed five seconds for FUEL frontier warmup")
        if not SMOKE_SCRIPT.is_file():
            raise ValueError(f"missing FUEL smoke script: {SMOKE_SCRIPT}")
        if shutil.which("wsl.exe") is None:
            raise ValueError("FUEL ROS smoke requires WSL Docker on this Windows host")
        lock = load_lock()
        source = verify_source(arguments.source, lock)
        command = _docker_command(
            image=arguments.image,
            script_path=SMOKE_SCRIPT,
            distribution=arguments.distribution,
            duration_s=arguments.duration_s,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"FUEL_ROS_SMOKE_REJECTED: {exc}", file=sys.stderr)
        return 2

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=arguments.duration_s + 30.0,
    )
    combined = "\n".join(item for item in (completed.stdout, completed.stderr) if item)
    result = _result_from_output(combined)
    report = {
        "schema": "org.aerocity.bench.fuel-ros-smoke-host-report.v1",
        "source": source,
        "image": arguments.image,
        "distribution": arguments.distribution,
        "container_exit_code": completed.returncode,
        "planner_report": result,
        "target_truth_exposed": False,
        "benchmark_score_claimed": False,
    }
    _emit_report(report, arguments.output)
    passed = (
        completed.returncode == 0
        and result is not None
        and result.get("status") == "ROUTE_EMITTED"
    )
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
