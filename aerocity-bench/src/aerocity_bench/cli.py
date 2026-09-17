"""Command-line entry point for legacy v2 and ordinary-v3 workflows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .audit import validate_release
from .baselines import BASELINES, baseline_descriptors, create_baseline
from .builder import build_release
from .builder_v3 import (
    build_ordinary_release,
    export_public_release,
    validate_ordinary_release,
    validate_public_release,
)
from .canonical import read_json, write_json
from .config import EXPECTED_SPLITS, load_release_config
from .errors import AeroCityError
from .metrics import evaluate_run
from .ordinary_config import (
    FORMAL_SPLITS,
    ORDINARY_SPLITS,
    load_ordinary_config,
    load_public_runtime_contract,
)
from .resources import PRESETS, preset, write_preset
from .runtime import L0FleetRuntime
from .supply_chain import load_official_cc0_lock


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aerocity-bench",
        description="Build, run, and validate AeroCityBench releases.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build an immutable authority release")
    build.add_argument("--release", type=Path, required=True, help="release JSON configuration")
    build.add_argument(
        "--asset-root", type=Path, required=True, help="directory containing asset bundles"
    )
    build.add_argument("--output", type=Path, required=True, help="new output directory")
    build.add_argument(
        "--split", dest="splits", action="append", help="build only this split; may be repeated"
    )
    build.add_argument("--source-commit", help="frozen 40-character source Git commit")
    build.add_argument(
        "--allow-uncommitted-development",
        action="store_true",
        help="build a development artifact from a dirty worktree",
    )

    validate = subparsers.add_parser("validate", help="validate an existing release")
    validate.add_argument("release_root", type=Path)

    export = subparsers.add_parser("export-public", help="export a private-free package")
    export.add_argument("authority_root", type=Path)
    export.add_argument("--output", type=Path, required=True)

    assets = subparsers.add_parser(
        "assets-verify", help="verify official CC0 provenance and USD closure"
    )
    assets.add_argument("--release", type=Path, required=True)
    assets.add_argument("--asset-root", type=Path, required=True)

    baselines = subparsers.add_parser("list-baselines", help="list benchmark-owned baselines")
    baselines.add_argument("--json", action="store_true", dest="as_json")

    run = subparsers.add_parser("run-baseline", help="run a benchmark-owned L0 baseline")
    run.add_argument("authority_root", type=Path)
    run.add_argument("--method", choices=tuple(BASELINES), required=True)
    run.add_argument("--split", choices=ORDINARY_SPLITS, default="calibration")
    run.add_argument("--layout-id")
    run.add_argument("--episode-index", type=int, default=0)
    run.add_argument("--max-steps", type=int)
    run.add_argument("--output", type=Path, required=True, help="new run output directory")

    evaluate = subparsers.add_parser("evaluate", help="evaluate an authority run result")
    evaluate.add_argument(
        "--run",
        type=Path,
        required=True,
        help="run directory or run_result_authority.json file",
    )
    evaluate.add_argument("--episode", type=Path, required=True)
    evaluate.add_argument("--duration-s", type=float, required=True)
    evaluate.add_argument("--output", type=Path)

    list_presets = subparsers.add_parser("list-presets", help="list wheel-bundled release presets")
    list_presets.add_argument("--json", action="store_true", dest="as_json")
    show_preset = subparsers.add_parser("show-preset", help="print a wheel-bundled preset")
    show_preset.add_argument("preset", choices=tuple(PRESETS))
    init_config = subparsers.add_parser("init-config", help="write a bundled preset for editing")
    init_config.add_argument("--preset", choices=tuple(PRESETS), required=True)
    init_config.add_argument("--output", type=Path, required=True)
    return parser


def _git_commit(repo: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip().lower()


def _git_worktree_clean(repo: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return not result.stdout.strip()


def _is_ordinary_config(path: Path) -> bool:
    try:
        return read_json(path).get("schema") == "org.aerocity.bench.release.ordinary.v3"
    except (AttributeError, json.JSONDecodeError, OSError):
        return False


def _is_ordinary_release(path: Path) -> bool:
    try:
        return read_json(path / "release_index.json").get("schema") == (
            "org.aerocity.bench.authority-release-index.ordinary.v1"
        )
    except (AttributeError, json.JSONDecodeError, OSError):
        return False


def _is_public_release(path: Path) -> bool:
    try:
        return read_json(path / "release_index.json").get("schema") == (
            "org.aerocity.bench.public-release-index.ordinary.v1"
        )
    except (AttributeError, json.JSONDecodeError, OSError):
        return False


def _build(args: argparse.Namespace) -> dict[str, Any]:
    if _is_ordinary_config(args.release):
        config = load_ordinary_config(args.release)
        splits = tuple(args.splits) if args.splits else ORDINARY_SPLITS
        repository = Path(__file__).resolve().parents[2]
        source_commit = args.source_commit or _git_commit(repository)
        worktree_clean = _git_worktree_clean(repository)
        if config.raw["release_kind"] == "OFFICIAL" and worktree_clean is not True:
            if not args.allow_uncommitted_development:
                raise ValueError(
                    "OFFICIAL builds require a clean Git worktree; use "
                    "--allow-uncommitted-development only for a development artifact"
                )
            source_commit = "UNCOMMITTED-DEVELOPMENT"
        return build_ordinary_release(
            config,
            args.asset_root,
            args.output,
            splits,
            source_commit=source_commit or "unknown",
            allow_uncommitted_development=args.allow_uncommitted_development,
        )
    config = load_release_config(args.release)
    splits = tuple(args.splits) if args.splits else EXPECTED_SPLITS
    return build_release(config, args.asset_root, args.output, splits)


def _assets_verify(args: argparse.Namespace) -> dict[str, Any]:
    config = load_ordinary_config(args.release)
    assets = config.raw["assets"]
    lock, _, closure = load_official_cc0_lock(
        args.asset_root, str(assets["bundle"]), list(assets["allowlist"])
    )
    return {
        "status": "PASS",
        "bundle": lock.bundle,
        "asset_count": len(lock.records),
        "usd_dependency_closure": closure,
    }


def _select_layout(
    authority_root: Path, split: str, layout_id: str | None
) -> tuple[Path, dict[str, Any]]:
    index = read_json(authority_root / "release_index.json")
    candidates = [
        layout
        for layout in index["layouts"]
        if layout["split"] == split and (layout_id is None or layout["layout_id"] == layout_id)
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"layout selection must resolve exactly once; split={split}, layout_id={layout_id}, "
            f"matches={len(candidates)}"
        )
    layout = candidates[0]
    return authority_root / "splits" / split / layout["layout_id"], layout


def _run_baseline(args: argparse.Namespace) -> dict[str, Any]:
    authority_root = args.authority_root.resolve()
    public_release = _is_public_release(authority_root)
    if public_release:
        validate_public_release(authority_root)
        config = load_public_runtime_contract(authority_root / "benchmark_contract.json")
        if args.split in FORMAL_SPLITS:
            raise ValueError("formal public splits require the blind evaluator")
    else:
        validate_ordinary_release(authority_root)
        config = load_ordinary_config(authority_root / "authority_private" / "release_config.json")
    layout_root, layout = _select_layout(authority_root, args.split, args.layout_id)
    episode_name = f"episode-{args.episode_index:04d}.json"
    evaluator_directory = "development_evaluator" if public_release else "evaluator_private"
    private_episode_path = layout_root / evaluator_directory / "episodes" / episode_name
    public_episode_path = layout_root / "method_public" / "episodes" / episode_name
    if not private_episode_path.is_file() or not public_episode_path.is_file():
        raise ValueError(f"episode index is absent: {args.episode_index}")
    private_episode = read_json(private_episode_path)
    public_episode = read_json(public_episode_path)
    city = read_json(layout_root / "scene_authority" / "cityspec.json")
    task_spec = read_json(layout_root / "method_public" / "task_spec.json")
    policy = create_baseline(
        args.method,
        config,
        task_spec,
        public_episode,
        private_episode=private_episode if BASELINES[args.method].requires_private_truth else None,
    )
    runtime = L0FleetRuntime(
        config,
        city,
        private_episode,
        public_task_spec=task_spec,
        public_episode=public_episode,
    )
    run_result = runtime.run_policy(policy, max_steps=args.max_steps)
    run_result["method"] = baseline_descriptors()[list(BASELINES).index(args.method)]
    run_result["layout_id"] = layout["layout_id"]
    duration = float(config.raw["execution_contract"]["episode"]["duration_s"])
    metric_report = evaluate_run(run_result, private_episode, duration)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "run_result_authority.json", run_result)
    write_json(args.output / "metrics.json", metric_report)
    return {
        "status": "PASS",
        "method": args.method,
        "layout_id": layout["layout_id"],
        "episode_id": private_episode["episode_id"],
        "execution_level": "L0",
        "formal_score_eligible": False,
        "confirmed_recall_auc": metric_report["quality"]["confirmed_recall_auc"],
        "output": str(args.output.resolve()),
    }


def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    run_path = args.run / "run_result_authority.json" if args.run.is_dir() else args.run
    result = evaluate_run(read_json(run_path), read_json(args.episode), args.duration_s)
    if args.output:
        write_json(args.output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "build":
            report = _build(args)
        elif args.command == "validate":
            if _is_ordinary_release(args.release_root):
                report = validate_ordinary_release(args.release_root)
            elif _is_public_release(args.release_root):
                report = validate_public_release(args.release_root)
            else:
                report = validate_release(args.release_root)
        elif args.command == "export-public":
            report = export_public_release(args.authority_root, args.output)
        elif args.command == "assets-verify":
            report = _assets_verify(args)
        elif args.command == "list-baselines":
            report = {"baselines": baseline_descriptors()}
            if not args.as_json:
                for item in report["baselines"]:
                    print(
                        f"{item['method_id']:24} {item['role']:12} "
                        f"profile={item['observation_profile']}"
                    )
                return 0
        elif args.command == "run-baseline":
            report = _run_baseline(args)
        elif args.command == "evaluate":
            report = _evaluate(args)
        elif args.command == "list-presets":
            report = {"presets": sorted(PRESETS)}
            if not args.as_json:
                print("\n".join(report["presets"]))
                return 0
        elif args.command == "show-preset":
            report = preset(args.preset)
        elif args.command == "init-config":
            report = write_preset(args.preset, args.output)
        else:
            raise ValueError(f"unsupported command: {args.command}")
    except (AeroCityError, FileExistsError, ImportError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("status") != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
