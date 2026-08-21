from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE_MODULES = (
    "shared.core.collision_2d",
    "shared.core.dynamic_gate_density_2d",
    "shared.runtime.paths",
    "gate_density_single.scripts.run_gate_density_eval",
    "gate_density_multi_8.scripts.train_dynamic_gate_density_8d_curriculum",
    "single_gate.env.single_gate_env",
    "multi_gate.env.multi_gate_env",
    "multi_gate.configs",
    "single_internal_gate.planners.classic_planners",
)
RUNTIME_MODULES = (
    "multi_gate.env.dynamic_gate_runtime",
    "multi_gate.env.guidance_runtime",
    "multi_gate.env.observation_runtime",
    "multi_gate.env.reward_runtime",
    "multi_gate.env.safety_shields",
)
PYTHON_SOURCE_ROOTS = (
    "aerogate",
    "assets",
    "multi_gate",
    "shared",
    "single_gate",
    "single_internal_gate",
    "gate_density_single",
    "gate_density_multi_8",
    "scripts",
    "tests",
)


def test_core_modules_import() -> None:
    for module_name in CORE_MODULES:
        importlib.import_module(module_name)


def test_multi_gate_runtime_modules_import_without_environment_bootstrap() -> None:
    for module_name in RUNTIME_MODULES:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib, sys; importlib.import_module(sys.argv[1])",
                module_name,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"{module_name}: {result.stderr}"


def test_all_local_python_files_parse() -> None:
    python_files = sorted(path for source_root in PYTHON_SOURCE_ROOTS for path in (ROOT / source_root).rglob("*.py"))
    for path in python_files:
        if "__pycache__" in path.parts:
            continue
        source = path.read_text(encoding="utf-8-sig")
        compile(source, str(path), "exec")
