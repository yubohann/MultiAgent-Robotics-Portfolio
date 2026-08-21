"""Run the dependency-light verification paths for the curated portfolio projects."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(name: str, cwd: Path, command: list[str], extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    env.update(extra_env or {})
    print(f"[check] {name}")
    completed = subprocess.run(command, cwd=cwd, env=env, check=False)
    if completed.returncode:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        choices=("all", "portfolio", "aerogate", "fraudgraph", "mid360", "robocup", "rivermark"),
        default="all",
        help="run one project check or the complete lightweight suite",
    )
    args = parser.parse_args()
    selected = {args.project} if args.project != "all" else {
        "portfolio", "aerogate", "fraudgraph", "mid360", "robocup", "rivermark"
    }
    python = sys.executable
    with tempfile.TemporaryDirectory(prefix="portfolio-check-") as temporary:
        scratch = Path(temporary)
        if "portfolio" in selected:
            run("portfolio integrity", ROOT, [python, "tools/verify_portfolio.py"])
        if "aerogate" in selected:
            project = ROOT / "aerogate-graph"
            run("aerogate tests", project, [python, "-m", "pytest", "-q"])
            run(
                "aerogate deterministic smoke",
                project,
                [
                    python, "-m", "aerogate", "reproduce", "--scenario", "multi-static",
                    "--agents", "4", "--seeds", "3", "7", "11", "--steps", "8",
                    "--output", str(scratch / "aerogate-reproduction.json"),
                ],
            )
        if "fraudgraph" in selected:
            project = ROOT / "fraudgraph-ml-engineering"
            run("fraudgraph repository validation", project, [python, "scripts/validate_repository.py"])
            run("fraudgraph tests", project, [python, "-m", "pytest", "-q"])
        if "mid360" in selected:
            project = ROOT / "robocon-mid360-autonomy-stack"
            run("mid360 contract tests", project, [python, "tools/run_python_contract_tests.py"])
            run("mid360 metadata validation", project, [python, "tools/validate_project.py"])
        if "robocup" in selected:
            project = ROOT / "robocup-cbg-wm"
            rl_root = project / "isaaclab_sim" / "rl"
            run("robocup rule tests", project, [python, "-m", "pytest", "tests", "-q"], {"PYTHONPATH": str(rl_root)})
            run(
                "robocup self-play smoke",
                project,
                [
                    python, "isaaclab_sim/rl/evaluate_selfplay.py", "--episodes", "2", "--max-steps", "8",
                    "--output", str(scratch / "robocup-selfplay.json"),
                ],
                {"PYTHONPATH": str(rl_root)},
            )
        if "rivermark" in selected:
            project = ROOT / "rivermark" / "code"
            run(
                "rivermark researcher smoke",
                project,
                [python, "-m", "rivermark_benchmark.researcher_entry", str(scratch / "rivermark-smoke")],
                {"PYTHONPATH": str(project / "src")},
            )
    print("[check] complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
