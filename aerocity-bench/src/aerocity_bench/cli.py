"""Command-line entry point for legacy v2 and ordinary-v3 workflows."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import subprocess
import sys
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

from .audit import validate_release
from .baselines import BASELINES, baseline_descriptors, create_baseline
from .builder import build_release
from .builder_v3 import (
    build_ordinary_release,
    export_public_release,
    promote_ordinary_release,
    validate_ordinary_release,
    validate_public_release,
)
from .canonical import content_hash, file_hash, read_json, write_json, write_json_atomic
from .config import EXPECTED_SPLITS, load_release_config
from .errors import AeroCityError, GenerationRejected, HostGuardError
from .host_guard import (
    HOST_GUARD_SCHEMA,
    WINDOWS_RUNTIME_COMMIT_LIMIT,
    WINDOWS_START_COMMIT_LIMIT,
    isaac_host_lock,
    run_guarded_process,
)
from .isaac_bridge import (
    REQUIRED_NATIVE_CHECKS,
    REVIEW_BASE_FRAMES,
    VISUAL_REVIEW_EVIDENCE_SCOPE,
    aggregate_review_instance_visibility,
    probe_isaac_runtime,
)
from .metrics import evaluate_run
from .native_gate_contract import load_native_gate_inputs
from .ordinary_config import (
    FORMAL_SPLITS,
    ORDINARY_SCHEMA,
    ORDINARY_SPLITS,
    load_ordinary_config,
    load_public_runtime_contract,
)
from .resources import PRESETS, preset, write_preset
from .runtime import L0FleetRuntime
from .supply_chain import load_official_cc0_lock
from .targets_v3 import sample_visual_review_episode_v3

_BATCH_VALIDATION_TOKEN = object()
L2_REVIEW_WIDTH = 960
L2_REVIEW_HEIGHT = 640


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aerocity-bench",
        description="Build, run, audit, and package AeroCityBench releases.",
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
        help="build a blocked development artifact; never makes it publicly releasable",
    )

    validate = subparsers.add_parser("validate", help="validate an existing release")
    validate.add_argument("release_root", type=Path)

    export = subparsers.add_parser(
        "export-public", help="export a private-free package after all release gates pass"
    )
    export.add_argument("authority_root", type=Path)
    export.add_argument("--output", type=Path, required=True)

    promote = subparsers.add_parser(
        "promote", help="seal a complete authority build with native and scientific evidence"
    )
    promote.add_argument("authority_root", type=Path)
    promote.add_argument("--native-report-dir", type=Path, required=True)
    promote.add_argument("--scientific-report", type=Path, required=True)
    promote.add_argument("--output", type=Path, required=True)

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

    subparsers.add_parser("probe-isaac", help="report whether the native runtime can be launched")
    native_gate = subparsers.add_parser(
        "native-gate",
        help="run fail-closed native Isaac checks for one authority layout",
    )
    native_gate.add_argument("authority_root", type=Path)
    native_gate.add_argument("--split", choices=ORDINARY_SPLITS, default="calibration")
    native_gate.add_argument("--layout-id")
    native_gate.add_argument("--output", type=Path, required=True, help="new native evidence dir")
    native_gate.add_argument("--isaac-python", type=Path)
    native_gate.add_argument("--timeout-s", type=float, default=600.0)
    native_gate.add_argument("--step-count", type=int, default=3)
    list_presets = subparsers.add_parser("list-presets", help="list wheel-bundled release presets")
    list_presets.add_argument("--json", action="store_true", dest="as_json")
    show_preset = subparsers.add_parser("show-preset", help="print a wheel-bundled preset")
    show_preset.add_argument("preset", choices=tuple(PRESETS))
    init_config = subparsers.add_parser("init-config", help="write a bundled preset for editing")
    init_config.add_argument("--preset", choices=tuple(PRESETS), required=True)
    init_config.add_argument("--output", type=Path, required=True)
    capture = subparsers.add_parser(
        "capture-review",
        help="create a private 32-target review episode and capture all Isaac views",
    )
    capture.add_argument("authority_root", type=Path)
    development_splits = tuple(split for split in ORDINARY_SPLITS if split not in FORMAL_SPLITS)
    capture.add_argument("--split", choices=development_splits, default="calibration")
    capture.add_argument("--layout-id")
    capture.add_argument("--target-count", type=int, default=32)
    capture.add_argument(
        "--process",
        choices=("uniform_surface", "clustered_surface", "height_stratified"),
        default="height_stratified",
    )
    capture.add_argument("--output", type=Path, required=True, help="new review output directory")
    capture.add_argument(
        "--isaac-python",
        type=Path,
        help="Isaac Sim Python executable; defaults to the current interpreter",
    )
    capture.add_argument("--width", type=int, default=L2_REVIEW_WIDTH)
    capture.add_argument("--height", type=int, default=L2_REVIEW_HEIGHT)
    capture.add_argument("--timeout-s", type=float, default=600.0)
    capture.add_argument(
        "--prepare-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    batch = subparsers.add_parser(
        "capture-review-batch",
        help="capture isolated, resumable Isaac review evidence for many development layouts",
    )
    batch.add_argument("authority_root", type=Path)
    batch.add_argument(
        "--split",
        dest="splits",
        action="append",
        choices=development_splits,
        help="development split to include; repeat as needed (default: all present)",
    )
    batch.add_argument("--target-count", type=int, default=32)
    batch.add_argument(
        "--process",
        choices=("uniform_surface", "clustered_surface", "height_stratified"),
        default="height_stratified",
    )
    batch.add_argument("--output", type=Path, required=True, help="batch evidence root")
    batch.add_argument("--isaac-python", type=Path)
    batch.add_argument("--width", type=int, default=L2_REVIEW_WIDTH)
    batch.add_argument("--height", type=int, default=L2_REVIEW_HEIGHT)
    batch.add_argument("--timeout-s", type=float, default=600.0)
    batch.add_argument("--max-attempts", type=int, default=2)
    batch.add_argument("--limit", type=int, help="smoke only: cap the number of layouts")
    batch.add_argument("--resume", action="store_true")
    batch.add_argument("--prepare-only", action="store_true", help=argparse.SUPPRESS)
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
    """Return whether ``repo`` is clean, or ``None`` if Git cannot be queried."""

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
        return read_json(path).get("schema") == ORDINARY_SCHEMA
    except (AttributeError, json.JSONDecodeError, OSError):
        return False


def _is_ordinary_release(path: Path) -> bool:
    index_path = path / "release_index.json"
    if not index_path.is_file():
        return False
    try:
        return read_json(index_path).get("schema") == (
            "org.aerocity.bench.authority-release-index.ordinary.v1"
        )
    except (AttributeError, json.JSONDecodeError, OSError):
        return False


def _is_public_release(path: Path) -> bool:
    index_path = path / "release_index.json"
    if not index_path.is_file():
        return False
    try:
        return read_json(index_path).get("schema") == (
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
                    "--allow-uncommitted-development only for a blocked development artifact"
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
    lock, evidence, closure = load_official_cc0_lock(
        args.asset_root, str(assets["bundle"]), list(assets["allowlist"])
    )
    return {
        "status": "PASS",
        "bundle": lock.bundle,
        "asset_count": len(lock.records),
        "registry_hash": lock.registry_hash,
        "provenance_manifest_hash": evidence.manifest_hash,
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


def _capture_script() -> object:
    source = Path(__file__).resolve().parents[2] / "tools" / "isaac_capture.py"
    if source.is_file():
        return source
    packaged = files("aerocity_bench").joinpath("tools", "isaac_capture.py")
    if not packaged.is_file():
        raise FileNotFoundError("the packaged Isaac capture script is absent")
    return packaged


def _capture_failure_detail(views: Path, log_path: Path) -> str:
    progress_path = views / "review_progress.log"
    if progress_path.is_file():
        progress = progress_path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line in reversed(progress):
            if line.startswith("exception="):
                return line.removeprefix("exception=").strip()
    if log_path.is_file():
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:].strip()
        if tail:
            return tail
    return "capture process produced no diagnostic detail"


def _review_layout_authority(
    authority_root: Path, layout: dict[str, Any]
) -> dict[str, Any]:
    layout_root = authority_root / "splits" / str(layout["split"]) / str(layout["layout_id"])
    scene_root = layout_root / "scene_authority"
    manifest_path = layout_root / "authority_manifest.json"
    cityspec_path = scene_root / "cityspec.json"
    stage_path = scene_root / "stage.usda"
    scene_path = scene_root / "scene.usda"
    collision_path = scene_root / "collision.usda"
    required = (manifest_path, cityspec_path, stage_path, scene_path, collision_path)
    if not all(path.is_file() for path in required):
        raise ValueError(f"layout authority files are incomplete: {layout_root}")
    manifest = read_json(manifest_path)
    city = read_json(cityspec_path)
    record = {
        "schema": "org.aerocity.bench.review-layout-authority.v1",
        "split": str(layout["split"]),
        "layout_id": str(layout["layout_id"]),
        "layout_hash": str(layout["layout_hash"]),
        "authority_manifest_hash": str(manifest.get("manifest_hash", "")),
        "authority_manifest_sha256": file_hash(manifest_path),
        "cityspec_sha256": file_hash(cityspec_path),
        "stage_sha256": file_hash(stage_path),
        "scene_sha256": file_hash(scene_path),
        "collision_sha256": file_hash(collision_path),
    }
    if (
        manifest.get("layout_id") != record["layout_id"]
        or manifest.get("layout_hash") != record["layout_hash"]
        or city.get("layout_id") != record["layout_id"]
        or city.get("layout_hash") != record["layout_hash"]
    ):
        raise ValueError(f"layout authority identity mismatch: {layout_root}")
    record["authority_record_hash"] = content_hash(record)
    return record


def _authority_record_valid(record: object, expected: dict[str, Any]) -> bool:
    if not isinstance(record, dict) or record != expected:
        return False
    payload = dict(record)
    claimed = str(payload.pop("authority_record_hash", ""))
    return bool(claimed) and content_hash(payload) == claimed


def _review_frame_modes_valid(frames: object) -> bool:
    if not isinstance(frames, dict):
        return False
    return all(
        isinstance(frame, dict)
        and frame.get("review_overlay_mode")
        == ("local_context" if str(name).startswith("target_close_") else "overview_highlight")
        for name, frame in frames.items()
    )


def _review_frame_resolution_valid(frames: object) -> bool:
    if not isinstance(frames, dict):
        return False
    expected_rgb_shape = [L2_REVIEW_HEIGHT, L2_REVIEW_WIDTH, 3]
    expected_plane_shape = [L2_REVIEW_HEIGHT, L2_REVIEW_WIDTH]
    for frame in frames.values():
        if not isinstance(frame, dict):
            return False
        rgb = frame.get("rgb")
        depth = frame.get("depth")
        instance = frame.get("instance_segmentation")
        if (
            not isinstance(rgb, dict)
            or not isinstance(depth, dict)
            or not isinstance(instance, dict)
            or rgb.get("shape") != expected_rgb_shape
            or depth.get("shape") != expected_plane_shape
            or instance.get("shape") != expected_plane_shape
        ):
            return False
    return True


def _review_pipeline_fingerprint(capture_script_sha256: str | None) -> dict[str, Any]:
    functions = {
        "review_sampler": sample_visual_review_episode_v3,
        "layout_authority": _review_layout_authority,
        "prepared_attempt_verifier": _verified_prepared_attempt,
        "review_attempt_verifier": _verified_review_attempt,
        "review_frame_mode_verifier": _review_frame_modes_valid,
        "review_frame_resolution_verifier": _review_frame_resolution_valid,
        "instance_visibility_aggregator": aggregate_review_instance_visibility,
    }
    sources = {
        name: content_hash(inspect.getsource(function))
        for name, function in functions.items()
    }
    fingerprint = {
        "schema": "org.aerocity.bench.review-pipeline-fingerprint.v1",
        "capture_script_sha256": capture_script_sha256,
        "function_source_hashes": sources,
    }
    fingerprint["pipeline_hash"] = content_hash(fingerprint)
    return fingerprint


def _native_gate_script() -> object:
    source = Path(__file__).resolve().parents[2] / "tools" / "isaac_native_gate.py"
    if source.is_file():
        return source
    packaged = files("aerocity_bench").joinpath("tools", "isaac_native_gate.py")
    if not packaged.is_file():
        raise FileNotFoundError("the packaged Isaac native gate script is absent")
    return packaged


def _native_gate(args: argparse.Namespace) -> dict[str, Any]:
    authority_root = args.authority_root.resolve()
    validate_ordinary_release(authority_root)
    if args.timeout_s <= 0.0:
        raise ValueError("native gate timeout must be positive")
    if args.step_count <= 0:
        raise ValueError("native gate step-count must be positive")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"native gate output already exists: {output}")
    python = (args.isaac_python or Path(sys.executable)).resolve()
    if not python.is_file():
        raise FileNotFoundError(f"Isaac Python executable is absent: {python}")
    layout_root, layout = _select_layout(authority_root, args.split, args.layout_id)
    scene_root = layout_root / "scene_authority"
    stage = (scene_root / "stage.usda").resolve()
    cityspec = (scene_root / "cityspec.json").resolve()
    release_config = (authority_root / "authority_private" / "release_config.json").resolve()
    task_spec = (layout_root / "method_public" / "task_spec.json").resolve()
    public_episodes = sorted((layout_root / "method_public" / "episodes").glob("*.json"))
    if not public_episodes:
        raise ValueError(f"layout has no public episode for native gate: {layout['layout_id']}")
    public_episode = public_episodes[0].resolve()
    _, _, _, _, expected_input_bindings = load_native_gate_inputs(
        release_config,
        task_spec,
        public_episode,
        cityspec,
    )
    output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[1]
    if source_root.name == "src":
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(source_root) if not existing else os.pathsep.join((str(source_root), existing))
        )
    log_path = output / "isaac_native_gate.log"
    host_report_path = output / "host_guard.json"
    script_resource = _native_gate_script()
    native_gate_script_sha256 = ""
    with isaac_host_lock():
        with as_file(script_resource) as script:
            native_gate_script_sha256 = file_hash(Path(script))
            process = run_guarded_process(
                [
                    str(python),
                    str(script),
                    "--stage",
                    str(stage),
                    "--cityspec",
                    str(cityspec),
                    "--release-config",
                    str(release_config),
                    "--task-spec",
                    str(task_spec),
                    "--public-episode",
                    str(public_episode),
                    "--output",
                    str(output),
                    "--step-count",
                    str(args.step_count),
                ],
                cwd=Path(__file__).resolve().parents[2],
                environment=environment,
                log_path=log_path,
                report_path=host_report_path,
                timeout_s=args.timeout_s,
            )
    if process.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise ValueError(f"native Isaac gate process failed with code {process.returncode}: {tail}")
    report_path = output / "native_gate.json"
    if not report_path.is_file():
        raise ValueError("native Isaac gate returned success without a report")
    report = read_json(report_path)
    payload = dict(report)
    report_hash = str(payload.pop("native_gate_hash", ""))
    if content_hash(payload) != report_hash:
        raise ValueError("native Isaac gate report hash mismatch")
    checks = report.get("checks", {})
    if set(checks) != set(REQUIRED_NATIVE_CHECKS):
        raise ValueError("native Isaac gate report has incomplete check fields")
    if report.get("stage_sha256") != file_hash(stage):
        raise ValueError("native Isaac gate report belongs to changed stage bytes")
    if report.get("input_bindings") != expected_input_bindings:
        raise ValueError("native Isaac gate report belongs to changed public inputs")
    if (
        report.get("runtime_fingerprint", {}).get("native_gate_script_sha256")
        != native_gate_script_sha256
    ):
        raise ValueError("native Isaac gate report belongs to changed gate code")
    failed = sorted(name for name, check in checks.items() if check.get("status") != "PASS")
    return {
        "status": "PASS" if not failed else "FAIL",
        "layout_id": layout["layout_id"],
        "split": layout["split"],
        "failed_checks": failed,
        "execution_level": report.get("execution_level"),
        "evidence_scope": report.get("evidence_scope"),
        "formal_score_eligible": not failed and report.get("formal_score_eligible") is True,
        "native_report": str(report_path),
        "native_report_hash": report_hash,
        "host_guard": str(host_report_path),
        "log": str(log_path),
    }


def _capture_review(args: argparse.Namespace) -> dict[str, Any]:
    authority_root = args.authority_root.resolve()
    if args.split in FORMAL_SPLITS:
        raise ValueError("visual review of evaluator-private formal test targets is forbidden")
    if args.target_count <= 0:
        raise ValueError("target-count must be positive")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("capture width and height must be positive")
    if not args.prepare_only and (
        args.width != L2_REVIEW_WIDTH or args.height != L2_REVIEW_HEIGHT
    ):
        raise ValueError(
            f"full L2 review requires the frozen {L2_REVIEW_WIDTH}x{L2_REVIEW_HEIGHT} profile"
        )
    if args.timeout_s <= 0.0:
        raise ValueError("capture timeout must be positive")
    if getattr(args, "_batch_validation_token", None) is not _BATCH_VALIDATION_TOKEN:
        validate_ordinary_release(authority_root)
    if args.output.exists():
        raise FileExistsError(f"review output already exists: {args.output}")
    config = load_ordinary_config(authority_root / "authority_private" / "release_config.json")
    layout_root, layout = _select_layout(authority_root, args.split, args.layout_id)
    authority_record = _review_layout_authority(authority_root, layout)
    city = read_json(layout_root / "scene_authority" / "cityspec.json")
    support = read_json(layout_root / "evaluator_private" / "support_sites.json")
    source_episodes = sorted((layout_root / "evaluator_private" / "episodes").glob("*.json"))
    if not source_episodes:
        raise ValueError(f"layout has no evaluator-private source episode: {layout['layout_id']}")
    source_episode = read_json(source_episodes[0])
    review = sample_visual_review_episode_v3(
        config,
        city,
        list(support["support_sites"]),
        list(source_episode["starts"]),
        target_count=args.target_count,
        process_name=args.process,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    episode_path = output / f"review_episode_{args.target_count}.json"
    write_json(episode_path, review)
    authority_path = output / "review_authority.json"
    write_json(authority_path, authority_record)
    prepared = {
        "status": "PREPARED" if args.prepare_only else "PASS",
        "layout_id": layout["layout_id"],
        "target_count": review["target_count"],
        "formal_score_eligible": False,
        "episode": str(episode_path),
        "authority_record": str(authority_path),
        "audit": review["audit"],
    }
    if args.prepare_only:
        return prepared

    python = (args.isaac_python or Path(sys.executable)).resolve()
    if not python.is_file():
        raise FileNotFoundError(f"Isaac Python executable is absent: {python}")
    views = output / "isaac_views"
    command_tail = [
        "--stage",
        str((layout_root / "scene_authority" / "stage.usda").resolve()),
        "--episode",
        str(episode_path),
        "--authority-record",
        str(authority_path),
        "--output",
        str(views),
        "--view",
        "review",
        "--width",
        str(args.width),
        "--height",
        str(args.height),
    ]
    environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[1]
    if source_root.name == "src":
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(source_root) if not existing else os.pathsep.join((str(source_root), existing))
        )
    script_resource = _capture_script()
    log_path = output / "isaac_capture.log"
    host_report_path = output / "host_guard.json"
    with isaac_host_lock():
        with as_file(script_resource) as script:
            process = run_guarded_process(
                [str(python), str(script), *command_tail],
                cwd=Path(__file__).resolve().parents[2],
                environment=environment,
                log_path=log_path,
                report_path=host_report_path,
                timeout_s=args.timeout_s,
            )
    if process.returncode != 0:
        detail = _capture_failure_detail(views, log_path)
        raise ValueError(
            f"Isaac review capture failed with code {process.returncode}: {detail}"
        )
    report_path = views / "isaac_scene_health_review.json"
    if not report_path.is_file():
        detail = _capture_failure_detail(views, log_path)
        raise ValueError(f"Isaac review capture failed before health report: {detail}")
    health = read_json(report_path)
    health_payload = dict(health)
    health_hash = str(health_payload.pop("health_report_hash", ""))
    start_count = len(health.get("private_target_audit", {}).get("start_positions", []))
    expected_frames = {
        *REVIEW_BASE_FRAMES,
        *(f"target_close_{index:03d}" for index in range(args.target_count)),
        *(f"start_close_{index:03d}" for index in range(start_count)),
    }
    if (
        health.get("status") != "passed"
        or health.get("schema") != "org.aerocity.bench.isaac-scene-health.v6"
        or content_hash(health_payload) != health_hash
        or health.get("private_target_audit", {}).get("target_count") != args.target_count
        or health.get("private_target_audit", {}).get("formal_score_eligible") is not False
        or health.get("private_target_audit", {}).get("start_markers_overlap_free") is not True
        or health.get("review_marker_visibility", {}).get("status") != "PASS"
        or health.get("instance_visibility", {}).get("status") != "PASS"
        or health.get("frame_diversity", {}).get("status") != "PASS"
        or health.get("review_overlay_collision_prim_count") != 0
        or health.get("review_overlay_rigid_body_prim_count") != 0
        or health.get("authority_record") != authority_record
        or set(health.get("frames", {})) != expected_frames
        or not _review_frame_modes_valid(health.get("frames"))
        or not _review_frame_resolution_valid(health.get("frames"))
        or health.get("evidence_scope") != VISUAL_REVIEW_EVIDENCE_SCOPE
        or not (views / "review_contact_sheet.png").is_file()
        or not (views / "target_review_contact_sheet.png").is_file()
    ):
        raise ValueError("Isaac review health report does not satisfy the review contract")
    prepared.update(
        {
            "views": str(views),
            "contact_sheet": str(views / "review_contact_sheet.png"),
            "target_contact_sheet": str(views / "target_review_contact_sheet.png"),
            "health_report": str(report_path),
            "capture_log": str(log_path),
            "host_guard": str(host_report_path),
            "rgb_frame_count": len(expected_frames),
            "depth_frame_count": len(expected_frames),
            "evidence_scope": health["evidence_scope"],
        }
    )
    return prepared


def _verified_review_attempt(
    path: Path, target_count: int, expected_authority: dict[str, Any]
) -> dict[str, Any] | None:
    report_path = path / "isaac_views" / "isaac_scene_health_review.json"
    contact_sheet = path / "isaac_views" / "review_contact_sheet.png"
    target_contact_sheet = path / "isaac_views" / "target_review_contact_sheet.png"
    episode = path / f"review_episode_{target_count}.json"
    host_report_path = path / "host_guard.json"
    authority_path = path / "review_authority.json"
    if not all(
        item.is_file()
        for item in (
            report_path,
            contact_sheet,
            target_contact_sheet,
            episode,
            host_report_path,
            authority_path,
        )
    ):
        return None
    try:
        report = read_json(report_path)
        host_report = read_json(host_report_path)
        review_episode = read_json(episode)
        authority_record = read_json(authority_path)
    except (json.JSONDecodeError, OSError):
        return None
    target_audit = report.get("private_target_audit", {})
    report_payload = dict(report)
    report_hash = str(report_payload.pop("health_report_hash", ""))
    frames = report.get("frames", {})
    episode_payload = dict(review_episode)
    episode_hash = str(episode_payload.pop("episode_hash", ""))
    contact = report.get("contact_sheet", {})
    target_contact = report.get("target_contact_sheet", {})
    expected_frames = {
        *REVIEW_BASE_FRAMES,
        *(f"target_close_{index:03d}" for index in range(target_count)),
        *(
            f"start_close_{index:03d}"
            for index in range(len(review_episode.get("starts", [])))
        ),
    }
    frame_files_match = True
    if isinstance(frames, dict):
        for name, frame in frames.items():
            rgb_path = path / "isaac_views" / f"{name}_rgb.png"
            depth_path = path / "isaac_views" / f"{name}_depth.png"
            mask_path = path / "isaac_views" / f"{name}_instance_segmentation.npz"
            labels_path = path / "isaac_views" / f"{name}_instance_labels.json"
            rgb = frame.get("rgb") if isinstance(frame, dict) else None
            depth = frame.get("depth") if isinstance(frame, dict) else None
            instance = frame.get("instance_segmentation") if isinstance(frame, dict) else None
            if (
                not isinstance(frame, dict)
                or not isinstance(rgb, dict)
                or not isinstance(depth, dict)
                or not isinstance(instance, dict)
                or not rgb_path.is_file()
                or not depth_path.is_file()
                or not mask_path.is_file()
                or not labels_path.is_file()
                or file_hash(rgb_path) != rgb.get("sha256")
                or file_hash(depth_path) != depth.get("sha256")
                or file_hash(mask_path) != instance.get("mask_sha256")
                or file_hash(labels_path) != instance.get("labels_sha256")
            ):
                frame_files_match = False
                break
    if (
        report.get("status") != "passed"
        or report.get("schema") != "org.aerocity.bench.isaac-scene-health.v6"
        or content_hash(report_payload) != report_hash
        or report.get("evidence_scope") != VISUAL_REVIEW_EVIDENCE_SCOPE
        or not _authority_record_valid(authority_record, expected_authority)
        or report.get("authority_record") != expected_authority
        or report.get("layout_id") != expected_authority["layout_id"]
        or report.get("layout_hash") != expected_authority["layout_hash"]
        or report.get("stage_sha256") != expected_authority["stage_sha256"]
        or report.get("scene_sha256") != expected_authority["scene_sha256"]
        or report.get("collision_sha256") != expected_authority["collision_sha256"]
        or target_audit.get("target_count") != target_count
        or target_audit.get("formal_score_eligible") is not False
        or target_audit.get("start_markers_overlap_free") is not True
        or report.get("review_marker_visibility", {}).get("status") != "PASS"
        or target_audit.get("episode_hash") != episode_hash
        or content_hash(episode_payload) != episode_hash
        or review_episode.get("formal_score_eligible") is not False
        or review_episode.get("target_count") != target_count
        or report.get("layout_id") != review_episode.get("layout_id")
        or report.get("instance_visibility", {}).get("status") != "PASS"
        or report.get("frame_diversity", {}).get("status") != "PASS"
        or report.get("review_overlay_collision_prim_count") != 0
        or report.get("review_overlay_rigid_body_prim_count") != 0
        or host_report.get("schema") != HOST_GUARD_SCHEMA
        or host_report.get("status") != "PASS"
        or host_report.get("returncode") != 0
        or host_report.get("trigger") is not None
        or not isinstance(frames, dict)
        or set(frames) != expected_frames
        or not _review_frame_modes_valid(frames)
        or not _review_frame_resolution_valid(frames)
        or not frame_files_match
        or contact.get("sha256") != file_hash(contact_sheet)
        or target_contact.get("sha256") != file_hash(target_contact_sheet)
        or contact.get("shape") != [1706, 960, 3]
        or target_contact.get("shape")
        != [
            106
            + 240 * ((target_count + len(review_episode.get("starts", [])) + 3) // 4),
            1280,
            3,
        ]
        or contact_sheet.stat().st_size <= 0
        or target_contact_sheet.stat().st_size <= 0
    ):
        return None
    return {
        "attempt": path.name,
        "health_report": str(report_path),
        "contact_sheet": str(contact_sheet),
        "target_contact_sheet": str(target_contact_sheet),
        "episode": str(episode),
        "host_guard": str(host_report_path),
    }


def _verified_prepared_attempt(
    path: Path, target_count: int, expected_authority: dict[str, Any]
) -> dict[str, Any] | None:
    episode = path / f"review_episode_{target_count}.json"
    authority_path = path / "review_authority.json"
    if not episode.is_file() or not authority_path.is_file():
        return None
    try:
        review = read_json(episode)
        authority_record = read_json(authority_path)
    except (json.JSONDecodeError, OSError):
        return None
    payload = dict(review)
    expected_hash = str(payload.pop("episode_hash", ""))
    if (
        review.get("schema") != "org.aerocity.bench.visual-review-private.v1"
        or review.get("formal_score_eligible") is not False
        or review.get("target_count") != target_count
        or review.get("layout_id") != expected_authority["layout_id"]
        or review.get("layout_hash") != expected_authority["layout_hash"]
        or not _authority_record_valid(authority_record, expected_authority)
        or content_hash(payload) != expected_hash
    ):
        return None
    return {"attempt": path.name, "episode": str(episode), "prepared_only": True}


def _capture_review_batch(args: argparse.Namespace) -> dict[str, Any]:
    authority_root = args.authority_root.resolve()
    validation = validate_ordinary_release(authority_root)
    if args.max_attempts <= 0:
        raise ValueError("max-attempts must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    if args.target_count <= 0:
        raise ValueError("target-count must be positive")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("capture width and height must be positive")
    if args.timeout_s <= 0.0:
        raise ValueError("capture timeout must be positive")
    selected_splits = tuple(
        dict.fromkeys(
            args.splits
            or (split for split in ORDINARY_SPLITS if split not in FORMAL_SPLITS)
        )
    )
    unknown_splits = sorted(set(selected_splits) - set(ORDINARY_SPLITS))
    if unknown_splits:
        raise ValueError(f"unknown review batch splits: {unknown_splits}")
    disclosed = sorted(set(selected_splits) & set(FORMAL_SPLITS))
    if disclosed:
        raise ValueError(
            f"visual review of evaluator-private formal splits is forbidden: {disclosed}"
        )
    index = read_json(authority_root / "release_index.json")
    layouts = [layout for layout in index["layouts"] if layout["split"] in selected_splits]
    layouts.sort(key=lambda layout: (layout["split"], layout["layout_id"]))
    if args.limit is not None:
        layouts = layouts[: args.limit]
    if not layouts:
        raise ValueError(f"no layouts match development splits: {selected_splits}")
    layout_authorities = {
        (str(layout["split"]), str(layout["layout_id"])): _review_layout_authority(
            authority_root, layout
        )
        for layout in layouts
    }
    review_start_count = 4
    for layout in layouts:
        public_episode_paths = sorted(
            (
                authority_root
                / "splits"
                / str(layout["split"])
                / str(layout["layout_id"])
                / "method_public"
                / "episodes"
            ).glob("*.json")
        )
        if public_episode_paths:
            review_start_count = max(
                review_start_count,
                len(read_json(public_episode_paths[0]).get("starts", [])),
            )

    output = args.output.resolve()
    contract_path = output / "batch_contract.json"
    python = (args.isaac_python or Path(sys.executable)).resolve()
    if not args.prepare_only and not python.is_file():
        raise FileNotFoundError(f"Isaac Python executable is absent: {python}")
    script_sha256 = None
    if not args.prepare_only:
        script_resource = _capture_script()
        with as_file(script_resource) as script:
            script_sha256 = file_hash(script)
    contract = {
        "schema": "org.aerocity.bench.visual-review-batch-contract.v6",
        "authority_root": str(authority_root),
        "release_index_hash": index["release_index_hash"],
        "splits": list(selected_splits),
        "layouts": [
            layout_authorities[(str(layout["split"]), str(layout["layout_id"]))]
            for layout in layouts
        ],
        "target_count": args.target_count,
        "review_start_count": review_start_count,
        "process": args.process,
        "width": args.width,
        "height": args.height,
        "prepare_only": bool(args.prepare_only),
        "timeout_s": float(args.timeout_s),
        "max_attempts": int(args.max_attempts),
        "isaac_python": None if args.prepare_only else str(python),
        "isaac_python_sha256": None if args.prepare_only else file_hash(python),
        "capture_script_sha256": script_sha256,
        "review_pipeline": _review_pipeline_fingerprint(script_sha256),
        "host_guard": {
            "schema": HOST_GUARD_SCHEMA,
            "windows_start_commit_limit": WINDOWS_START_COMMIT_LIMIT,
            "windows_runtime_commit_limit": WINDOWS_RUNTIME_COMMIT_LIMIT,
            "failure_policy": "abort_batch_without_scientific_failure",
        },
        "formal_score_eligible": False,
        "process_isolation": "one_fresh_isaac_process_per_layout",
    }
    contract["contract_hash"] = content_hash(contract)
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"batch output already exists; use --resume: {output}")
        if not contract_path.is_file() or read_json(contract_path) != contract:
            raise ValueError("resume contract differs from the existing batch contract")
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(contract_path, contract)

    pixels = args.width * args.height
    frame_count = len(REVIEW_BASE_FRAMES) + args.target_count + review_start_count
    frame_bytes = frame_count * pixels * 5
    contact_sheet_bytes = (
        960 * (106 + 5 * 320) * 3
        + 1280
        * (106 + 240 * ((args.target_count + review_start_count + 3) // 4))
        * 3
    )
    per_attempt_reserve = int((frame_bytes + contact_sheet_bytes) * 2.5)
    required_free_bytes = max(
        1024**3,
        len(layouts) * args.max_attempts * per_attempt_reserve,
    )
    if shutil.disk_usage(output).free < required_free_bytes:
        raise ValueError(
            "insufficient free space for review batch; "
            f"require at least {required_free_bytes} bytes"
        )

    jobs: list[dict[str, Any]] = []
    host_abort = False
    for layout_index, layout in enumerate(layouts):
        expected_authority = layout_authorities[
            (str(layout["split"]), str(layout["layout_id"]))
        ]
        remaining_layouts = len(layouts) - layout_index
        remaining_required = max(
            512 * 1024**2,
            remaining_layouts * args.max_attempts * per_attempt_reserve,
        )
        if shutil.disk_usage(output).free < remaining_required:
            raise ValueError(
                "insufficient free space while review batch is running; "
                f"require at least {remaining_required} bytes for remaining layouts"
            )
        job_root = output / "scenes" / layout["split"] / layout["layout_id"]
        verified = None
        for existing in sorted(job_root.glob("attempt-*")):
            if args.prepare_only:
                verified = _verified_prepared_attempt(
                    existing, args.target_count, expected_authority
                )
            else:
                verified = _verified_review_attempt(existing, args.target_count, expected_authority)
            if verified is not None:
                break
        errors: list[dict[str, Any]] = []
        if verified is None:
            existing_attempts = sorted(job_root.glob("attempt-*"))
            for attempt_index in range(len(existing_attempts) + 1, args.max_attempts + 1):
                attempt = job_root / f"attempt-{attempt_index:02d}"
                child_values = vars(args).copy()
                child_values.update(
                    {
                        "command": "capture-review",
                        "split": layout["split"],
                        "layout_id": layout["layout_id"],
                        "output": attempt,
                        "_batch_validation_token": _BATCH_VALIDATION_TOKEN,
                    }
                )
                try:
                    _capture_review(argparse.Namespace(**child_values))
                except KeyboardInterrupt:
                    interruption = {
                        "schema": "org.aerocity.bench.visual-review-batch-report.v1",
                        "contract_hash": contract["contract_hash"],
                        "status": "INTERRUPTED",
                        "jobs": jobs,
                        "active_job": {
                            "split": layout["split"],
                            "layout_id": layout["layout_id"],
                            "attempt": attempt.name,
                        },
                    }
                    write_json_atomic(output / "batch_progress.json", interruption)
                    raise
                except HostGuardError as exc:
                    errors.append(
                        {
                            "attempt": attempt.name,
                            "error_type": type(exc).__name__,
                            "category": "execution_host_failure_not_scientific_failure",
                            "message": str(exc)[-4000:],
                        }
                    )
                    host_abort = True
                    break
                except GenerationRejected as exc:
                    errors.append(
                        {
                            "attempt": attempt.name,
                            "error_type": type(exc).__name__,
                            "category": "deterministic_generation_rejection_not_retried",
                            "message": str(exc)[-4000:],
                        }
                    )
                    break
                except (AeroCityError, FileExistsError, ImportError, ValueError, OSError) as exc:
                    errors.append(
                        {
                            "attempt": attempt.name,
                            "error_type": type(exc).__name__,
                            "message": str(exc)[-4000:],
                        }
                    )
                    continue
                if args.prepare_only:
                    verified = _verified_prepared_attempt(
                        attempt, args.target_count, expected_authority
                    )
                else:
                    verified = _verified_review_attempt(
                        attempt, args.target_count, expected_authority
                    )
                if verified is not None:
                    break
                errors.append(
                    {
                        "attempt": attempt.name,
                        "error_type": "PostconditionError",
                        "message": "capture returned without complete verifiable evidence",
                    }
                )
        jobs.append(
            {
                "split": layout["split"],
                "layout_id": layout["layout_id"],
                "status": "PASS" if verified is not None else "FAIL",
                "verified": verified,
                "errors": errors,
            }
        )
        partial = {
            "schema": "org.aerocity.bench.visual-review-batch-report.v1",
            "contract_hash": contract["contract_hash"],
            "status": "IN_PROGRESS",
            "jobs": jobs,
        }
        write_json_atomic(output / "batch_progress.json", partial)
        if host_abort:
            completed = {(job["split"], job["layout_id"]) for job in jobs}
            for remaining in layouts:
                key = (remaining["split"], remaining["layout_id"])
                if key not in completed:
                    jobs.append(
                        {
                            "split": remaining["split"],
                            "layout_id": remaining["layout_id"],
                            "status": "ABORTED_HOST_FAILURE",
                            "verified": None,
                            "errors": [],
                        }
                    )
            break

    passed = sum(job["status"] == "PASS" for job in jobs)
    report = {
        "schema": "org.aerocity.bench.visual-review-batch-report.v1",
        "contract_hash": contract["contract_hash"],
        "status": "PASS" if passed == len(jobs) else "FAIL",
        "authority_validation_status": validation["status"],
        "layout_count": len(jobs),
        "passed_layout_count": passed,
        "failed_layout_count": len(jobs) - passed,
        "formal_score_eligible": False,
        "host_abort": host_abort,
        "jobs": jobs,
    }
    report["report_hash"] = content_hash(report)
    write_json_atomic(output / "batch_report.json", report)
    write_json_atomic(
        output / "batch_progress.json",
        {
            "schema": "org.aerocity.bench.visual-review-batch-report.v1",
            "contract_hash": contract["contract_hash"],
            "status": report["status"],
            "jobs": jobs,
            "batch_report": str((output / "batch_report.json").resolve()),
            "report_hash": report["report_hash"],
        },
    )
    return report


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
        elif args.command == "promote":
            report = promote_ordinary_release(
                args.authority_root,
                args.output,
                native_report_dir=args.native_report_dir,
                scientific_report_path=args.scientific_report,
            )
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
        elif args.command == "probe-isaac":
            report = probe_isaac_runtime()
        elif args.command == "native-gate":
            report = _native_gate(args)
        elif args.command == "list-presets":
            report = {"presets": sorted(PRESETS)}
            if not args.as_json:
                print("\n".join(report["presets"]))
                return 0
        elif args.command == "show-preset":
            report = preset(args.preset)
        elif args.command == "init-config":
            report = write_preset(args.preset, args.output)
        elif args.command == "capture-review":
            report = _capture_review(args)
        elif args.command == "capture-review-batch":
            report = _capture_review_batch(args)
        else:
            raise ValueError(f"unsupported command: {args.command}")
    except (AeroCityError, FileExistsError, ImportError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("status") != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
