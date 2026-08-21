"""Run dependency-light Python contract tests across ROS 2 package boundaries."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ROS_RUNTIME_IMPORTS = ("import rclpy", "from rclpy", "from std_msgs", "import std_msgs")


def package_paths() -> list[Path]:
    paths = [path for path in SRC.iterdir() if path.is_dir() and (path / "setup.py").is_file()]
    paths.append(SRC / "robocon_mid360_simulation" / "scripts")
    return [path for path in paths if path.is_dir()]


def module_uses_ros(module: str) -> bool:
    parts = module.split(".")
    if len(parts) < 2 or not parts[0].startswith("robocon_"):
        return False
    source = SRC / parts[0] / parts[0] / ("/".join(parts[1:]) + ".py")
    if not source.is_file():
        return False
    return any(token in source.read_text(encoding="utf-8") for token in ROS_RUNTIME_IMPORTS)


def requires_ros_runtime(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    if any(token in source for token in ROS_RUNTIME_IMPORTS):
        return True
    tree = ast.parse(source, filename=str(path))
    return any(
        isinstance(node, ast.ImportFrom) and node.module and module_uses_ros(node.module)
        for node in ast.walk(tree)
    )


def contract_tests() -> list[Path]:
    tests = sorted(SRC.rglob("test_*.py"))
    return [path for path in tests if not requires_ros_runtime(path)]


def main() -> int:
    tests = contract_tests()
    if not tests:
        print("No dependency-light contract tests found.")
        return 1
    skipped = len(list(SRC.rglob("test_*.py"))) - len(tests)
    env = os.environ.copy()
    inherited = env.get("PYTHONPATH", "")
    roots = [str(path) for path in package_paths()]
    if inherited:
        roots.append(inherited)
    env["PYTHONPATH"] = os.pathsep.join(roots)
    command = [sys.executable, "-m", "pytest", *map(str, tests)]
    print(f"[contract-tests] running {len(tests)} tests; deferring {skipped} ROS-runtime tests to ROS CI")
    print("[contract-tests]", " ".join(command))
    return subprocess.run(command, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
