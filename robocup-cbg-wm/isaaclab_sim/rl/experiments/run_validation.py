from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TESTS = (
    "tests/test_cbg_world_model.py",
    "tests/test_rl_strategy_contract.py",
    "tests/test_constraint_graph_dynamics.py",
    "tests/test_paired_interventions.py",
    "tests/test_paper_statistics.py",
)


def worktree_diff_files() -> list[str]:
    tracked = subprocess.run(
        ["git", "diff", "--name-only", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return sorted(set(tracked.stdout.splitlines()) | set(untracked.stdout.splitlines()))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run focused CBG-WM paper-suite tests and record evidence.")
    parser.add_argument("tests", nargs="*", default=list(DEFAULT_TESTS))
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "isaaclab_sim" / "output" / "paper" / "cbg_wm_2026" / "validation" / "unit_tests.json",
    )
    args = parser.parse_args()

    test_files = [str(Path(value)).replace("\\", "/") for value in args.tests]
    missing = [value for value in test_files if not (ROOT / value).is_file()]
    if missing:
        payload = {
            "status": "failed",
            "completed": False,
            "exit_code": 2,
            "test_files": test_files,
            "missing_test_files": missing,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "worktree_diff_files": worktree_diff_files(),
        }
    else:
        command = [sys.executable, "-m", "pytest", "-q", *test_files]
        result = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
        payload = {
            "status": "passed" if result.returncode == 0 else "failed",
            "completed": result.returncode == 0,
            "exit_code": result.returncode,
            "command": command,
            "test_files": test_files,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "worktree_diff_files": worktree_diff_files(),
        }

    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return int(payload["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
