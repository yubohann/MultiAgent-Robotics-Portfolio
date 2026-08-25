from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from aerocity_bench.host_guard import _is_isaac_process_record


def _tool():
    path = Path("tools/run_external_cf2x_l1_preflight.py")
    spec = importlib.util.spec_from_file_location("external_cf2x_l1_preflight_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_external_l1_command_uses_process_policy_and_the_pinned_manifest(tmp_path: Path) -> None:
    tool = _tool()
    inputs = {
        "layout": tmp_path / "layout",
        "config": tmp_path / "release.json",
        "cf2x": tmp_path / "cf2x.usd",
        "manifest": tmp_path / "adapter-manifest.json",
        "isaac_python": tmp_path / "python.exe",
        "output": tmp_path / "public.json",
        "private_output": tmp_path / "private.json",
    }
    args = tool._arguments(
        [
            "--layout-root", str(inputs["layout"]), "--release-config", str(inputs["config"]),
            "--output", str(inputs["output"]), "--private-output", str(inputs["private_output"]),
            "--runtime-root", str(tmp_path / "runtime"), "--cf2x-usd", str(inputs["cf2x"]),
            "--external-adapter-manifest", str(inputs["manifest"]),
            "--isaac-python", str(inputs["isaac_python"]),
            "--isaaclab-root", str(tmp_path / "IsaacLab"),
        ]
    )

    command = tool._command(inputs, args)

    assert command[0] == str(inputs["isaac_python"])
    assert command[1].endswith("cf2x_l1_fleet_preflight.py")
    assert "external-process-policy" in command
    assert command[command.index("--external-adapter-manifest") + 1] == str(inputs["manifest"])
    assert "--headless" in command


def test_external_l1_launcher_rejects_the_forbidden_legacy_asset(tmp_path: Path) -> None:
    tool = _tool()
    forbidden = tmp_path / "5_in_drone" / "cf2x.usd"
    forbidden.parent.mkdir()
    forbidden.write_text("usd", encoding="utf-8")

    with pytest.raises(ValueError, match="assets/new/cf2x.usd"):
        tool._validate_cf2x_asset(forbidden)


def test_host_guard_recognizes_a_foreign_hm3d_cf2x_replay() -> None:
    assert _is_isaac_process_record(
        "python.exe", "python scripts/replay_hm3d_cf2x_collision.py --headless"
    )


def test_host_guard_recognizes_an_unlisted_headless_isaaclab_launcher() -> None:
    assert _is_isaac_process_record(
        "python.exe",
        "C:/IsaacLab/envs/env_isaaclab/python.exe "
        "scripts/run_hm3d_p07_exploration_episode.py --headless",
    )


def test_external_l1_main_reports_host_blocks_without_a_traceback(monkeypatch, capsys) -> None:
    tool = _tool()
    from aerocity_bench.errors import HostGuardError

    monkeypatch.setattr(tool, "_arguments", lambda _argv: object())
    monkeypatch.setattr(tool, "run", lambda _args: (_ for _ in ()).throw(HostGuardError("busy")))

    assert tool.main([]) == 2
    assert capsys.readouterr().err.strip() == "EXTERNAL_CF2X_L1_PREFLIGHT=BLOCKED: busy"
