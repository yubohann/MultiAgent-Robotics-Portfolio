"""Static checks for the isolated FUEL build recipe."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.build_fuel_container import load_lock, verify_source
from tools.run_fuel_ros_smoke import _docker_command, _emit_report, _result_from_output

ROOT = Path(__file__).resolve().parents[1]


def test_fuel_lock_pins_gpl_source_and_immutable_ros_base() -> None:
    lock = load_lock()
    assert lock["upstream_license"] == "GPL-3.0-only"
    assert lock["process_boundary"] == "container"
    assert len(lock["upstream_commit"]) == 40
    assert lock["base_image"].startswith("ros@sha256:")


def test_fuel_recipe_builds_only_locked_source_context() -> None:
    recipe = (ROOT / "external" / "fuel" / "Dockerfile").read_text(encoding="utf-8")
    assert "COPY . /opt/fuel/ws/src/fuel" in recipe
    assert "git -C src/fuel rev-parse HEAD" in recipe
    assert "git -C src/fuel config core.autocrlf true" in recipe
    assert "git -C src/fuel status --porcelain" in recipe
    assert "libnlopt-cxx-dev" in recipe
    assert "libnlopt.so" in recipe
    assert "CATKIN_WHITELIST_PACKAGES" in recipe
    assert "rviz" not in recipe.lower()


def test_fuel_ros_smoke_only_exercises_public_planner_io() -> None:
    smoke = (ROOT / "external" / "fuel" / "aerocity_fuel_ros_smoke.py").read_text(
        encoding="utf-8"
    )
    assert '"/planning/bspline"' in smoke
    assert '"/position_cmd"' not in smoke
    assert "target_truth_exposed" in smoke
    assert "benchmark_score_claimed" in smoke
    assert "ROS_MASTER_URI" in smoke
    assert "ROS_HOME" in smoke
    assert "planner_log_tail" in smoke
    assert "TRIGGER_WARMUP_S = 3.0" in smoke


def test_fuel_smoke_container_is_network_isolated_and_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "tools.run_fuel_ros_smoke._docker_bind_path",
        lambda path: "/mnt/c/aerocity_fuel_ros_smoke.py",
    )
    command = _docker_command(
        image="fuel:test",
        script_path=ROOT / "external" / "fuel" / "aerocity_fuel_ros_smoke.py",
        distribution="Ubuntu-22.04",
        duration_s=12.0,
    )
    assert "--network" in command
    assert command[command.index("--network") + 1] == "none"
    assert "--read-only" in command
    assert "/planning/bspline" not in " ".join(command)


def test_fuel_smoke_result_parser_accepts_only_its_schema() -> None:
    output = "ignored\n{\"schema\": \"other\"}\n" + (
        '{"schema":"org.aerocity.bench.fuel-ros-smoke.v1","status":"ROUTE_EMITTED"}\n'
    )
    assert _result_from_output(output) == {
        "schema": "org.aerocity.bench.fuel-ros-smoke.v1",
        "status": "ROUTE_EMITTED",
    }


def test_fuel_smoke_report_writer_preserves_machine_readable_payload(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    _emit_report({"schema": "org.aerocity.bench.fuel-ros-smoke-host-report.v1"}, output)
    assert output.read_text(encoding="utf-8") == (
        '{\n  "schema": "org.aerocity.bench.fuel-ros-smoke-host-report.v1"\n}\n'
    )


def test_fuel_source_rejects_unversioned_or_dirty_snapshots(tmp_path: Path) -> None:
    lock = load_lock()
    snapshot = tmp_path / "fuel-snapshot"
    snapshot.mkdir()
    with pytest.raises(ValueError, match="Git checkout"):
        verify_source(snapshot, lock)
