from __future__ import annotations

import importlib
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


def test_core_modules_import() -> None:
    for module_name in CORE_MODULES:
        importlib.import_module(module_name)


def test_all_local_python_files_parse() -> None:
    for path in sorted(ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")

