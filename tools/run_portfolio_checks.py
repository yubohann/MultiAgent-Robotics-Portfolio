"""Run the dependency-light verification paths for the curated portfolio projects."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from portfolio_registry import RegistryError, load_registry


ROOT = Path(__file__).resolve().parents[1]
CHECK_HANDLERS = frozenset({"aerogate", "fraudgraph", "mid360", "robocup", "rivermark"})


def run(name: str, cwd: Path, command: list[str], extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    env.update(extra_env or {})
    print(f"[check] {name}")
    completed = subprocess.run(command, cwd=cwd, env=env, check=False)
    if completed.returncode:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}")


def main() -> int:
    try:
        registry = load_registry(ROOT)
    except RegistryError as exc:
        print(f"Portfolio registry check failed: {exc}")
        return 1
    available_checks = registry.verification_keys
    missing_handlers = set(available_checks) - CHECK_HANDLERS
    if missing_handlers:
        print(
            "Portfolio registry check failed: no runner is registered for verification key(s): "
            + ", ".join(sorted(missing_handlers))
        )
        return 1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        choices=("all", "portfolio", *available_checks),
        default="all",
        help="run one project check or the complete lightweight suite",
    )
    args = parser.parse_args()
    selected = {args.project} if args.project != "all" else {"portfolio", *available_checks}
    python = sys.executable
    with tempfile.TemporaryDirectory(prefix="portfolio-check-") as temporary:
        scratch = Path(temporary)
        if "portfolio" in selected:
            run("portfolio integrity", ROOT, [python, "tools/verify_portfolio.py"])
        if "aerogate" in selected:
            project = registry.project_for_verification("aerogate").directory
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
            project = registry.project_for_verification("fraudgraph").directory
            run("fraudgraph repository validation", project, [python, "scripts/validate_repository.py"])
            run("fraudgraph tests", project, [python, "-m", "pytest", "-q"])
        if "mid360" in selected:
            project = registry.project_for_verification("mid360").directory
            run("mid360 contract tests", project, [python, "tools/run_python_contract_tests.py"])
            run("mid360 metadata validation", project, [python, "tools/validate_project.py"])
        if "robocup" in selected:
            project = registry.project_for_verification("robocup").directory
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
            project = registry.project_for_verification("rivermark").directory / "code"
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
