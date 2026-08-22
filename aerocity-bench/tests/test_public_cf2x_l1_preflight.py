from __future__ import annotations

import importlib.util
from pathlib import Path


def _tool():
    path = Path("tools/run_public_cf2x_l1_preflight.py")
    spec = importlib.util.spec_from_file_location("public_cf2x_l1_preflight_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_l1_command_uses_only_a_public_policy(tmp_path: Path) -> None:
    tool = _tool()
    inputs = {
        "layout": tmp_path / "layout",
        "config": tmp_path / "release.json",
        "cf2x": tmp_path / "cf2x.usd",
        "isaac_python": tmp_path / "python.exe",
        "output": tmp_path / "public.json",
        "private_output": tmp_path / "private.json",
    }
    args = tool._arguments(
        [
            "--layout-root", str(inputs["layout"]), "--release-config", str(inputs["config"]),
            "--output", str(inputs["output"]), "--private-output", str(inputs["private_output"]),
            "--runtime-root", str(tmp_path / "runtime"), "--cf2x-usd", str(inputs["cf2x"]),
            "--method", "atlas-region-greedy", "--isaac-python", str(inputs["isaac_python"]),
            "--isaaclab-root", str(tmp_path / "IsaacLab"),
        ]
    )

    command = tool._command(inputs, args)

    assert command[0] == str(inputs["isaac_python"])
    assert "public-policy" in command
    assert command[command.index("--method") + 1] == "atlas-region-greedy"
    assert "--external-adapter-manifest" not in command


def test_unexpected_child_exit_creates_a_failure_receipt(tmp_path: Path) -> None:
    tool = _tool()
    output = tmp_path / "public.json"
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    report = runtime / "host_guard.json"
    report.write_text("{}", encoding="utf-8")

    tool._record_unexpected_child_exit(
        {"output": output, "runtime": runtime}, returncode=-1, binding="binding"
    )

    failure = tool._failure_path(output)
    assert failure.is_file()
    assert "child_process_exit_without_executor_receipt" in failure.read_text(encoding="utf-8")
