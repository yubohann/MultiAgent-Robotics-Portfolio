"""Evaluate the 8-drone checkpoint on team sizes 1..7."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, fields, is_dataclass, replace
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_gate.configs import get_multi_experiment_config, normalize_multi_experiment_config_name
from multi_gate.configs.experiment_config import FORMAL_MULTI_TEAM_SIZES
from multi_gate.env.multi_gate_env import MultiGate2DEnv
from multi_gate.graph_rl.graph_masac import GraphMASACAgent
from multi_gate.training import (
    _load_checkpoint_metadata,
    _multi_episode_success_from_info,
    _multi_resume_compatibility_findings,
    _select_multi_env_class,
    validate_multi_checkpoint_compatibility,
)
from shared.core.dynamic_gate_density_2d import post_clearance, swept_post_clearance


DEFAULT_OUTPUT_ROOT = ROOT / "results" / f"variable_team_size_eval_{datetime.now():%Y%m%d_%H%M%S}"
EXPECTED_EXPERIMENT_ID = "multi_gate_dynamic_gate_density_8d_v1"
EXPECTED_CHECKPOINT_STEM = "graph_flashsac_multi_dynamic_gate_density_8d"
EXPECTED_NODE_SHAPE = [85, 18]
EXPECTED_ACTION_MASK_SHAPE = [34]
DEFAULT_TEAM_SIZES = tuple(range(1, 8))
DEFAULT_PILOT_TEAM_SIZES = (1, 4, 7)
DEFAULT_DYNAMIC_PILOT_GATES = (12, 18, 24)
DEFAULT_STATIC_PILOT_GATES = (18, 30, 42)
DEFAULT_DYNAMIC_GATE_COUNT = 18
DEFAULT_STATIC_GATE_COUNT = 30
DEFAULT_SPEED_MPS = 0.8
DEFAULT_AMPLITUDE_M = 0.75
DEFAULT_GATE_POST_RADIUS_M = 0.65
DEFAULT_DRONE_RADIUS_M = 0.48
DEFAULT_CLEAN_SWEPT_CLEARANCE_M = 0.12


class JsonWriter:
    """Small JSON/CSV writer with numpy-safe conversion."""

    @staticmethod
    def clean(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return JsonWriter.clean(value.tolist())
        if isinstance(value, np.generic):
            return JsonWriter.clean(value.item())
        if isinstance(value, float):
            return value if math.isfinite(value) else None
        if isinstance(value, (list, tuple)):
            return [JsonWriter.clean(item) for item in value]
        if isinstance(value, dict):
            return {str(key): JsonWriter.clean(item) for key, item in value.items()}
        if is_dataclass(value):
            return JsonWriter.clean(asdict(value))
        return value

    @staticmethod
    def write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(JsonWriter.clean(payload), indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: (
                            json.dumps(JsonWriter.clean(value), ensure_ascii=False)
                            if isinstance(value, (dict, list, tuple))
                            else JsonWriter.clean(value)
                        )
                        for key, value in row.items()
                    }
                )


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(JsonWriter.clean(row), ensure_ascii=False) + "\n")


def append_jsonl_many(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(JsonWriter.clean(row), ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def episode_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (str(row.get("condition")), int(row.get("team_size")), int(row.get("seed")))


def parse_int_range(text: str) -> list[int]:
    value = str(text).strip()
    if not value:
        return []
    if ":" in value:
        parts = value.split(":")
        if len(parts) not in {2, 3}:
            raise argparse.ArgumentTypeError(f"Invalid range: {text}")
        start = int(parts[0])
        stop = int(parts[1])
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        return list(range(start, stop, step))
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def finite_or_none(value: Any) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if math.isfinite(resolved) else None


def bool01(value: Any) -> int:
    return 1 if bool(value) else 0


def percentile(values: Iterable[float], q: float) -> float | None:
    arr = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.percentile(arr, q))


def mean(values: Iterable[float]) -> float | None:
    arr = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def std(values: Iterable[float]) -> float | None:
    arr = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    if arr.size <= 1:
        return 0.0 if arr.size == 1 else None
    return float(np.std(arr, ddof=1))


def min_finite(values: Iterable[float]) -> float | None:
    arr = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    if arr.size == 0:
        return None
    return float(np.min(arr))


def bootstrap_ci(values: Iterable[float], *, seed: int = 20260520, samples: int = 1000) -> tuple[float | None, float | None]:
    arr = np.asarray([float(value) for value in values if math.isfinite(float(value))], dtype=np.float64)
    if arr.size == 0:
        return (None, None)
    if arr.size == 1:
        value = float(arr[0])
        return (value, value)
    rng = np.random.default_rng(seed)
    means = np.empty(int(samples), dtype=np.float64)
    for idx in range(int(samples)):
        means[idx] = float(np.mean(rng.choice(arr, size=arr.size, replace=True)))
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataclass_field_names(value: Any) -> set[str]:
    if not is_dataclass(value):
        return set()
    return {field.name for field in fields(value)}


def replace_known(value: Any, **updates: Any) -> Any:
    allowed = dataclass_field_names(value)
    return replace(value, **{key: item for key, item in updates.items() if key in allowed})


def make_eval_config(
    *,
    base_config: Any,
    team_size: int,
    gate_count: int,
    speed_mps: float,
    amplitude_m: float,
    gate_post_radius_m: float,
    drone_radius_m: float,
    disable_guidance_runtime: bool,
    max_episode_steps: int | None,
    disable_terminal_formation_collapse: bool = False,
    formation_line_collapse_min_lateral_bands: int | None = None,
) -> Any:
    gate_cfg = replace_known(
        base_config.dynamic_gate_density,
        gate_count=int(gate_count),
        moving_gate_speed_mps=float(speed_mps),
        moving_gate_amplitude_m=float(amplitude_m),
        gate_post_radius_m=float(gate_post_radius_m),
        drone_radius_m=float(drone_radius_m),
    )
    env_updates: dict[str, Any] = {
        "drone_radius_m": float(drone_radius_m),
        "timeout_counts_as_success": False,
    }
    if bool(disable_terminal_formation_collapse):
        env_updates["formation_line_collapse_terminal"] = False
    if formation_line_collapse_min_lateral_bands is not None:
        env_updates["formation_line_collapse_min_lateral_bands"] = max(
            int(formation_line_collapse_min_lateral_bands),
            0,
        )
    if int(team_size) <= 2:
        env_updates.update(
            {
                "formation_line_collapse_min_lateral_bands": 0,
                "formation_line_collapse_terminal": False,
                "formation_line_collapse_penalty_scale": 0.0,
            }
        )
    if max_episode_steps is not None:
        env_updates["max_episode_steps"] = int(max_episode_steps)
    env_cfg = replace_known(base_config.environment, **env_updates)
    reasoning_cfg = base_config.reasoning
    if disable_guidance_runtime:
        reasoning_cfg = replace_known(
            reasoning_cfg,
            route_guidance_enabled=False,
            guidance_shadow_mode=False,
            guidance_async_enabled=False,
            guidance_cache_enabled=False,
            guidance_provider="none",
        )
    return replace(
        base_config,
        min_agents=1 if int(team_size) == 1 else int(base_config.min_agents),
        default_agents=int(team_size),
        dynamic_gate_density=gate_cfg,
        environment=env_cfg,
        reasoning=reasoning_cfg,
    )


def create_env(config: Any) -> MultiGate2DEnv:
    env_cls = _select_multi_env_class(config)
    return env_cls(
        multi_config=config,
        env_config=config.environment,
        observation_config=config.observation,
        formation_config=config.formation,
        planner_config=config.planner,
    )


def metadata_experiment_id(metadata: dict[str, Any]) -> str:
    signature = dict(metadata.get("training_signature") or {})
    summary = metadata.get("summary") if isinstance(metadata.get("summary"), dict) else {}
    experiment_config = (
        metadata.get("experiment_config") if isinstance(metadata.get("experiment_config"), dict) else {}
    )
    return str(
        signature.get("experiment_id")
        or metadata.get("experiment_id")
        or summary.get("experiment_id")
        or experiment_config.get("experiment_id")
        or ""
    ).strip()


def metadata_algorithm_name(metadata: dict[str, Any]) -> str:
    signature = dict(metadata.get("training_signature") or {})
    return str(metadata.get("algorithm_name") or signature.get("algorithm_name") or "").strip()


def metadata_max_agents_soft(metadata: dict[str, Any]) -> int | None:
    signature = dict(metadata.get("training_signature") or {})
    experiment_config = (
        metadata.get("experiment_config") if isinstance(metadata.get("experiment_config"), dict) else {}
    )
    value = signature.get("max_agents_soft") or experiment_config.get("max_agents_soft")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def metadata_observation_shapes(metadata: dict[str, Any]) -> dict[str, list[int]]:
    signature = dict(metadata.get("training_signature") or {})
    shapes = signature.get("observation_shapes") or metadata.get("observation_shapes") or {}
    if not isinstance(shapes, dict):
        return {}
    return {str(key): [int(dim) for dim in value] for key, value in shapes.items()}


def candidate_paths(root: Path, limit: int) -> list[Path]:
    include_markers = (
        "dynamic_gate_density_8d",
        "graph_flashsac",
        "supervised_e5",
    )
    reject_markers = (
        "_bc_actor",
        "\\dagger\\",
        "/dagger/",
        "\\datasets\\",
        "/datasets/",
        "\\unit_",
        "/unit_",
        "\\_smoke\\",
        "/_smoke/",
        "single",
        "oldmethod",
        "isaaclab_video",
    )
    paths: list[Path] = []
    seen: set[Path] = set()

    def add_path(path: Path) -> None:
        try:
            resolved = path.resolve()
        except OSError:
            return
        if not resolved.exists() or resolved in seen:
            return
        lowered = str(resolved).lower()
        if any(marker in lowered for marker in reject_markers):
            return
        if not any(marker in lowered for marker in include_markers):
            return
        seen.add(resolved)
        paths.append(resolved)

    summary_paths = list((root / "runtime").rglob("training_summary.json"))
    summary_paths.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    for summary_path in summary_paths:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        raw_candidates = [
            summary.get("checkpoint_path"),
            summary.get("best_alias_path"),
            summary.get("best_checkpoint_path"),
            summary.get("selected_checkpoint_path"),
            summary.get("final_checkpoint_path"),
            summary.get("latest_alias_path"),
        ]
        for raw in raw_candidates:
            if not raw:
                continue
            path = Path(str(raw))
            if not path.is_absolute():
                path = (root.parents[1] / path) if str(path).replace("\\", "/").startswith("experiments/") else (root / path)
            add_path(path)

    for path in (root / "runtime").rglob("*.pt"):
        lowered = str(path).lower()
        if not any(marker in lowered for marker in include_markers):
            continue
        if any(marker in lowered for marker in reject_markers):
            continue
        add_path(path)
    return paths[: max(int(limit), 1)]


def audit_checkpoint(
    *,
    checkpoint_path: Path | None,
    base_config: Any,
    output_root: Path,
    candidate_limit: int,
) -> tuple[Path, dict[str, Any]]:
    env = create_env(base_config)
    try:
        candidates = [checkpoint_path] if checkpoint_path is not None else candidate_paths(ROOT, candidate_limit)
        rejected: list[dict[str, Any]] = []
        selected: dict[str, Any] | None = None
        for candidate in candidates:
            if candidate is None:
                continue
            path = Path(candidate)
            if not path.exists():
                rejected.append({"path": str(path), "reason": "missing"})
                continue
            record: dict[str, Any] = {
                "path": str(path),
                "modified_time": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                "size_bytes": int(path.stat().st_size),
            }
            try:
                metadata = _load_checkpoint_metadata(path)
                findings = _multi_resume_compatibility_findings(
                    metadata=metadata,
                    env=env,
                    experiment_config=base_config,
                )
            except Exception as exc:  # noqa: BLE001 - audit must continue over bad files.
                record["reason"] = f"metadata_or_compatibility_error: {exc}"
                rejected.append(record)
                continue
            shapes = metadata_observation_shapes(metadata)
            hard_failures: list[str] = []
            if metadata_experiment_id(metadata) != EXPECTED_EXPERIMENT_ID:
                hard_failures.append("experiment_id")
            if metadata_algorithm_name(metadata) != "graph_flashsac":
                hard_failures.append("algorithm_name")
            if metadata_max_agents_soft(metadata) != 34:
                hard_failures.append("max_agents_soft")
            if shapes.get("node_features") != EXPECTED_NODE_SHAPE:
                hard_failures.append("node_features_shape")
            if shapes.get("action_mask") != EXPECTED_ACTION_MASK_SHAPE:
                hard_failures.append("action_mask_shape")
            if EXPECTED_CHECKPOINT_STEM not in path.name and path.name not in {"best_agent.pt", "latest_agent.pt"}:
                hard_failures.append("checkpoint_name")
            incompatible = [name for name, result in findings.items() if not bool(result["compatible"])]
            if incompatible:
                hard_failures.append(f"compatibility:{','.join(incompatible)}")
            record.update(
                {
                    "experiment_id": metadata_experiment_id(metadata),
                    "algorithm_name": metadata_algorithm_name(metadata),
                    "max_agents_soft": metadata_max_agents_soft(metadata),
                    "observation_shapes": shapes,
                    "compatibility_findings": findings,
                    "metadata_brief": {
                        "checkpoint_step": metadata.get("checkpoint_step"),
                        "checkpoint_kind": metadata.get("checkpoint_kind"),
                        "seed": metadata.get("seed"),
                        "training_signature": metadata.get("training_signature"),
                        "experiment_config": metadata.get("experiment_config"),
                    },
                }
            )
            if hard_failures:
                record["reason"] = "hard_check_failed"
                record["failed_checks"] = hard_failures
                rejected.append(record)
                continue
            record["reason"] = "selected_latest_passing_candidate"
            record["sha256"] = sha256_file(path)
            selected = record
            break
        if selected is None:
            JsonWriter.write_json(output_root / "checkpoint_audit.json", {"selected": None, "rejected": rejected})
            raise RuntimeError("No compatible 8d Graph-FlashSAC checkpoint found.")
        selected_path = Path(str(selected["path"]))
        audit = {
            "selected": selected,
            "why_selected": (
                "First newest candidate passing strict Graph-FlashSAC 8d metadata checks, "
                "shape checks, max_agents_soft=34, and validate_multi_checkpoint_compatibility."
            ),
            "rejected_candidate_count": len(rejected),
            "rejected_candidates": rejected[:200],
            "rejected_candidates_truncated": len(rejected) > 200,
            "required": {
                "experiment_id": EXPECTED_EXPERIMENT_ID,
                "algorithm_name": "graph_flashsac",
                "node_features": EXPECTED_NODE_SHAPE,
                "action_mask": EXPECTED_ACTION_MASK_SHAPE,
                "max_agents_soft": 34,
            },
        }
        JsonWriter.write_json(output_root / "checkpoint_audit.json", audit)
        return selected_path, audit
    finally:
        env.close()


def build_agent(*, env: MultiGate2DEnv, config: Any, checkpoint_path: Path, seed: int, device: str | None) -> GraphMASACAgent:
    validate_multi_checkpoint_compatibility(
        checkpoint_path=checkpoint_path,
        env=env,
        experiment_config=config,
    )
    agent = GraphMASACAgent.from_defaults(
        obs_shapes=env.observation_shapes,
        device=device,
        seed=int(seed),
        obs_config=config.observation,
        masac_config=config.algorithm,
        max_agents_soft=config.max_agents_soft,
        build_replay_buffer=False,
    )
    agent.load_checkpoint(checkpoint_path)
    agent.actor.eval()
    return agent


def span_major_minor(points_xy: np.ndarray) -> tuple[float, float]:
    points = np.asarray(points_xy, dtype=np.float64)
    if points.shape[0] <= 1:
        return (0.0, 0.0)
    centered = points - np.mean(points, axis=0, keepdims=True)
    if points.shape[0] == 2:
        distance = float(np.linalg.norm(points[0] - points[1]))
        return (distance, 0.0)
    cov = centered.T @ centered / max(points.shape[0] - 1, 1)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return (0.0, 0.0)
    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order]
    projected = centered @ axes
    major = float(np.max(projected[:, 0]) - np.min(projected[:, 0]))
    minor = float(np.max(projected[:, 1]) - np.min(projected[:, 1]))
    return (major, minor)


def formation_audit(
    *,
    positions: np.ndarray,
    desired_slots: np.ndarray,
    lateral_band_counts: list[float],
    env_line_collapse_scores: list[float],
    team_size: int,
) -> dict[str, Any]:
    if int(team_size) <= 1:
        return {
            "multi_formation_actual_major_span_p95_m": None,
            "multi_formation_actual_minor_span_p05_m": None,
            "multi_formation_desired_major_span_p50_m": None,
            "multi_formation_desired_minor_span_p50_m": None,
            "multi_formation_major_ratio_p95": None,
            "multi_formation_minor_ratio_p05": None,
            "multi_formation_long_queue_violation": None,
            "formation_line_collapse_rate": None,
            "formation_lateral_band_count_min": None,
            "formation_line_collapse_score_mean": None,
        }
    actual_major: list[float] = []
    actual_minor: list[float] = []
    desired_major: list[float] = []
    desired_minor: list[float] = []
    for pos, slots in zip(positions, desired_slots):
        major, minor = span_major_minor(pos)
        d_major, d_minor = span_major_minor(slots)
        actual_major.append(major)
        actual_minor.append(minor)
        desired_major.append(d_major)
        desired_minor.append(d_minor)
    actual_major_p95 = percentile(actual_major, 95.0)
    actual_minor_p05 = percentile(actual_minor, 5.0)
    desired_major_p50 = percentile(desired_major, 50.0)
    desired_minor_p50 = percentile(desired_minor, 50.0)
    major_ratio = (
        None
        if desired_major_p50 is None or desired_major_p50 <= 1.0e-6 or actual_major_p95 is None
        else float(actual_major_p95 / desired_major_p50)
    )
    minor_ratio = (
        None
        if desired_minor_p50 is None or desired_minor_p50 <= 1.0e-6 or actual_minor_p05 is None
        else float(actual_minor_p05 / desired_minor_p50)
    )
    if int(team_size) == 2:
        long_queue_violation = None
    else:
        long_queue_violation = bool(
            (actual_major_p95 is not None and actual_major_p95 > max(9.0, (desired_major_p50 or 0.0) * 1.8))
            or (major_ratio is not None and major_ratio > 1.8)
            or (minor_ratio is not None and minor_ratio < 0.25)
        )
    collapse_scores = [float(value) for value in env_line_collapse_scores if math.isfinite(float(value))]
    return {
        "multi_formation_actual_major_span_p95_m": actual_major_p95,
        "multi_formation_actual_minor_span_p05_m": actual_minor_p05,
        "multi_formation_desired_major_span_p50_m": desired_major_p50,
        "multi_formation_desired_minor_span_p50_m": desired_minor_p50,
        "multi_formation_major_ratio_p95": major_ratio,
        "multi_formation_minor_ratio_p05": minor_ratio,
        "multi_formation_long_queue_violation": None if long_queue_violation is None else int(long_queue_violation),
        "formation_line_collapse_rate": (
            None
            if not collapse_scores
            else float(sum(1 for value in collapse_scores if value > 0.0) / max(len(collapse_scores), 1))
        ),
        "formation_lateral_band_count_min": min_finite(lateral_band_counts),
        "formation_line_collapse_score_mean": mean(collapse_scores),
    }


def episode_success_from_info(info: dict[str, Any], config: Any) -> bool:
    return bool(
        _multi_episode_success_from_info(
            info,
            timeout_counts_as_success=bool(getattr(config.environment, "timeout_counts_as_success", False)),
        )
    )


def corridor_success_from_info(info: dict[str, Any]) -> bool:
    gate_count = int(info.get("dynamic_gate_count") or 0)
    if gate_count <= 0:
        return True
    return bool(info.get("corridor_completed", False)) and not bool(info.get("side_bypass_failure", False)) and not bool(
        info.get("corridor_miss_failure", False)
    )


def record_frame(info: dict[str, Any], step: int) -> dict[str, Any]:
    return {
        "step": int(step),
        "positions": np.asarray(info.get("agent_positions_xy"), dtype=np.float32),
        "velocities": np.asarray(info.get("agent_velocities_xy"), dtype=np.float32),
        "desired_slots": np.asarray(info.get("desired_slots"), dtype=np.float32),
        "live_gate_centers": np.asarray(info.get("live_gate_centers_xy"), dtype=np.float32),
        "live_gate_posts": np.asarray(info.get("live_gate_post_positions_xy"), dtype=np.float32),
        "virtual_center": np.asarray(info.get("virtual_center_xy"), dtype=np.float32),
        "path_index": int(info.get("path_index") or 0),
        "lateral_band_count": finite_or_none(info.get("formation_lateral_band_count")),
        "line_collapse_score": finite_or_none(info.get("formation_line_collapse_score")),
        "min_clearance": finite_or_none(info.get("min_clearance_m")),
        "min_pair_distance": finite_or_none(info.get("min_pair_distance_m")),
        "route_guidance_source": info.get("route_guidance_source"),
    }


def stack_frames(frames: list[dict[str, Any]], key: str) -> np.ndarray:
    if not frames:
        return np.zeros((0,), dtype=np.float32)
    return np.stack([np.asarray(frame[key], dtype=np.float32) for frame in frames], axis=0)


def run_episode(
    *,
    env: MultiGate2DEnv,
    agent: GraphMASACAgent,
    config: Any,
    condition_name: str,
    condition_kind: str,
    gate_count: int,
    speed_mps: float,
    amplitude_m: float,
    team_size: int,
    seed: int,
    checkpoint_path: Path,
    output_root: Path,
    clean_swept_clearance_m: float,
    save_rollout: bool,
    step_sample_stride: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path | None]:
    obs, info = env.reset(seed=int(seed), num_agents=int(team_size))
    frames = [record_frame(info, 0)]
    sample_rows: list[dict[str, Any]] = []
    total_reward = 0.0
    step_count = 0
    path_length_m = 0.0
    previous_center = np.asarray(info.get("virtual_center_xy"), dtype=np.float32)
    live_clearances: list[float] = []
    swept_clearances: list[float] = []
    env_clearances: list[float] = []
    pair_distances: list[float] = []
    slot_errors: list[float] = []
    max_slot_errors: list[float] = []
    speed_samples: list[float] = []
    shield_active_steps = 0
    shield_norms: list[float] = []
    guidance_query_count = 0
    any_gate_collision = False
    any_height_failure = False
    any_side_bypass = False
    any_corridor_miss = False
    any_formation_line_collapse = False
    last_info = info

    while True:
        previous_positions = np.asarray(last_info.get("agent_positions_xy"), dtype=np.float32)
        previous_posts = np.asarray(last_info.get("live_gate_post_positions_xy"), dtype=np.float32)
        action = agent.act(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        step_count += 1
        current_positions = np.asarray(info.get("agent_positions_xy"), dtype=np.float32)
        current_posts = np.asarray(info.get("live_gate_post_positions_xy"), dtype=np.float32)
        live_clearance = post_clearance(current_positions, current_posts, config=config.dynamic_gate_density)
        swept_clearance = swept_post_clearance(
            previous_positions,
            current_positions,
            previous_posts,
            current_posts,
            config=config.dynamic_gate_density,
        )
        live_clearances.append(float(live_clearance))
        swept_clearances.append(float(swept_clearance))
        env_clearance = finite_or_none(info.get("min_clearance_m"))
        if env_clearance is not None:
            env_clearances.append(env_clearance)
        pair_distance = finite_or_none(info.get("min_pair_distance_m"))
        if pair_distance is not None:
            pair_distances.append(pair_distance)
        slot_error = finite_or_none(info.get("mean_slot_error_m"))
        if slot_error is not None:
            slot_errors.append(slot_error)
        max_slot_error = finite_or_none(info.get("max_slot_error_m"))
        if max_slot_error is not None:
            max_slot_errors.append(max_slot_error)
        velocities = np.asarray(info.get("agent_velocities_xy"), dtype=np.float32)
        if velocities.size:
            speed_samples.extend(float(value) for value in np.linalg.norm(velocities, axis=1))
        current_center = np.asarray(info.get("virtual_center_xy"), dtype=np.float32)
        if current_center.shape == (2,) and previous_center.shape == (2,):
            path_length_m += float(np.linalg.norm(current_center - previous_center))
            previous_center = current_center
        shield_info = info.get("action_safety_shield")
        if isinstance(shield_info, dict):
            if bool(shield_info.get("active", False)):
                shield_active_steps += 1
            shield_norm = finite_or_none(shield_info.get("mean_intervention_norm"))
            if shield_norm is not None:
                shield_norms.append(shield_norm)
        guidance_meta = info.get("route_guidance_meta")
        if isinstance(guidance_meta, dict) and bool(guidance_meta.get("request_submitted", False)):
            guidance_query_count += 1
        any_gate_collision = any_gate_collision or bool(info.get("dynamic_gate_collision", False))
        any_height_failure = any_height_failure or bool(info.get("height_escape_failure", False))
        any_side_bypass = any_side_bypass or bool(info.get("side_bypass_failure", False))
        any_corridor_miss = any_corridor_miss or bool(info.get("corridor_miss_failure", False))
        any_formation_line_collapse = any_formation_line_collapse or bool(
            info.get("formation_line_collapse_failure", False)
        )
        frame = record_frame(info, step_count)
        frames.append(frame)
        if step_count % max(int(step_sample_stride), 1) == 0 or terminated or truncated:
            sample_rows.append(
                {
                    "condition": condition_name,
                    "condition_kind": condition_kind,
                    "team_size": int(team_size),
                    "seed": int(seed),
                    "step": int(step_count),
                    "virtual_center_x_m": float(current_center[0]) if current_center.shape == (2,) else None,
                    "virtual_center_y_m": float(current_center[1]) if current_center.shape == (2,) else None,
                    "min_live_gate_clearance_m": float(live_clearance),
                    "min_swept_gate_clearance_m": float(swept_clearance),
                    "min_environment_clearance_m": env_clearance,
                    "min_pair_distance_m": pair_distance,
                    "mean_slot_error_m": slot_error,
                    "formation_lateral_band_count": frame["lateral_band_count"],
                    "formation_line_collapse_score": frame["line_collapse_score"],
                    "dynamic_gate_collision": int(bool(info.get("dynamic_gate_collision", False))),
                    "done_reason": info.get("done_reason"),
                }
            )
        last_info = info
        if terminated or truncated:
            break

    positions = stack_frames(frames, "positions")
    desired_slots = stack_frames(frames, "desired_slots")
    lateral_bands = [float(frame["lateral_band_count"]) for frame in frames if frame["lateral_band_count"] is not None]
    collapse_scores = [float(frame["line_collapse_score"]) for frame in frames if frame["line_collapse_score"] is not None]
    formation = formation_audit(
        positions=positions,
        desired_slots=desired_slots,
        lateral_band_counts=lateral_bands,
        env_line_collapse_scores=collapse_scores,
        team_size=int(team_size),
    )
    done_reason = str(last_info.get("done_reason") or "unknown")
    success = episode_success_from_info(last_info, config)
    gate_collision = any_gate_collision or done_reason == "gate_post_collision"
    dynamic_gate_collision = bool(gate_collision and condition_kind == "dynamic")
    static_gate_collision = bool(gate_collision and condition_kind == "static")
    agent_collision = bool(done_reason == "agent_collision")
    timeout = bool(done_reason == "timeout")
    out_of_bounds = bool(done_reason == "out_of_bounds")
    corridor_success = corridor_success_from_info(last_info)
    min_live_clearance = min_finite(live_clearances)
    min_swept_clearance = min_finite(swept_clearances)
    min_env_clearance = min_finite(env_clearances)
    visual_geometry_collision = bool(
        (min_live_clearance is not None and min_live_clearance <= 0.0)
        or (min_swept_clearance is not None and min_swept_clearance <= 0.0)
    )
    clearance_threshold_violation = bool(
        min_swept_clearance is not None and min_swept_clearance < float(clean_swept_clearance_m)
    )
    long_queue_violation = formation.get("multi_formation_long_queue_violation")
    formation_invalid = bool(long_queue_violation == 1)
    clean_success = bool(
        success
        and not dynamic_gate_collision
        and not static_gate_collision
        and not agent_collision
        and not timeout
        and not out_of_bounds
        and corridor_success
        and not visual_geometry_collision
        and not clearance_threshold_violation
        and not formation_invalid
        and not any_height_failure
        and not any_side_bypass
        and not any_corridor_miss
    )
    trajectory_path: Path | None = None
    if save_rollout:
        trajectory_dir = output_root / "episodes" / condition_name / f"team_{int(team_size):02d}" / f"seed_{int(seed):02d}"
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        trajectory_path = trajectory_dir / "rollout.npz"
        np.savez_compressed(
            trajectory_path,
            positions=positions,
            velocities=stack_frames(frames, "velocities"),
            desired_slots=desired_slots,
            live_gate_centers=stack_frames(frames, "live_gate_centers"),
            live_gate_posts=stack_frames(frames, "live_gate_posts"),
            virtual_center=stack_frames(frames, "virtual_center"),
            path_waypoints=np.asarray(last_info.get("path_waypoints"), dtype=np.float32),
            live_clearance=np.asarray([float("inf"), *live_clearances], dtype=np.float32),
            swept_clearance=np.asarray([float("inf"), *swept_clearances], dtype=np.float32),
            min_environment_clearance=np.asarray(
                [float("nan") if frame["min_clearance"] is None else float(frame["min_clearance"]) for frame in frames],
                dtype=np.float32,
            ),
            dt_s=np.asarray([float(config.environment.dt_s)], dtype=np.float32),
            fixed_height_m=np.asarray([float(config.environment.fixed_height_m)], dtype=np.float32),
            world_x_bounds_m=np.asarray(config.environment.world_x_bounds_m, dtype=np.float32),
            world_y_bounds_m=np.asarray(config.environment.world_y_bounds_m, dtype=np.float32),
            gate_post_radius_m=np.asarray([float(config.dynamic_gate_density.gate_post_radius_m)], dtype=np.float32),
            drone_radius_m=np.asarray([float(config.environment.drone_radius_m)], dtype=np.float32),
        )
        JsonWriter.write_json(
            trajectory_dir / "rollout_meta.json",
            {
                "condition": condition_name,
                "condition_kind": condition_kind,
                "team_size": int(team_size),
                "seed": int(seed),
                "gate_count": int(gate_count),
                "moving_gate_speed_mps": float(speed_mps),
                "moving_gate_amplitude_m": float(amplitude_m),
                "checkpoint_path": str(checkpoint_path),
                "done_reason": done_reason,
                "success": success,
                "clean_success": clean_success,
                "formation": formation,
                "trajectory_path": str(trajectory_path),
            },
        )

    row = {
        "condition": condition_name,
        "condition_kind": condition_kind,
        "team_size": int(team_size),
        "seed": int(seed),
        "gate_count": int(gate_count),
        "moving_gate_speed_mps": float(speed_mps),
        "moving_gate_amplitude_m": float(amplitude_m),
        "checkpoint_path": str(checkpoint_path),
        "trajectory_path": None if trajectory_path is None else str(trajectory_path),
        "success": int(success),
        "clean_success": int(clean_success),
        "success_but_geometry_invalid": int(success and visual_geometry_collision),
        "success_but_clearance_below_threshold": int(success and clearance_threshold_violation),
        "success_but_formation_invalid": int(success and formation_invalid),
        "done_reason": done_reason,
        "steps": int(step_count),
        "flight_time_s": float(step_count * float(config.environment.dt_s)),
        "episode_reward": float(total_reward),
        "path_length_m": float(path_length_m),
        "mean_speed_mps": mean(speed_samples),
        "max_speed_mps": max(speed_samples) if speed_samples else None,
        "goal_distance_improvement_m": finite_or_none(last_info.get("goal_distance_improvement_m")),
        "goal_progress_ratio": finite_or_none(last_info.get("goal_progress_ratio")),
        "corridor_through_success": int(corridor_success),
        "dynamic_gate_collision": int(dynamic_gate_collision),
        "static_gate_collision": int(static_gate_collision),
        "gate_collision": int(gate_collision),
        "agent_collision": int(agent_collision),
        "timeout": int(timeout),
        "out_of_bounds": int(out_of_bounds),
        "height_escape_failure": int(any_height_failure),
        "side_bypass_failure": int(any_side_bypass),
        "corridor_miss_failure": int(any_corridor_miss),
        "formation_line_collapse_failure": int(any_formation_line_collapse),
        "visual_geometry_collision_violation": int(visual_geometry_collision),
        "clearance_threshold_violation": int(clearance_threshold_violation),
        "min_live_gate_clearance_m": min_live_clearance,
        "min_swept_gate_clearance_m": min_swept_clearance,
        "min_environment_clearance_m": min_env_clearance,
        "min_pair_distance_m": min_finite(pair_distances),
        "mean_slot_error_m": mean(slot_errors),
        "max_slot_error_m": max(max_slot_errors) if max_slot_errors else None,
        "shield_activation_count": int(shield_active_steps),
        "shield_activation_ratio": float(shield_active_steps / max(step_count, 1)),
        "shield_intervention_norm_mean": mean(shield_norms),
        "guidance_query_count": int(guidance_query_count),
        "route_guidance_source_final": last_info.get("route_guidance_source"),
        "planner_call_count": int(last_info.get("planner_call_count") or 0),
        "planner_latency_ms_mean": finite_or_none(last_info.get("planner_latency_ms_mean")),
        "dynamic_gate_motion_range_m_final": finite_or_none(last_info.get("actual_gate_motion_range_m")),
        **formation,
    }
    return row, sample_rows, trajectory_path


def run_condition(
    *,
    condition: dict[str, Any],
    team_sizes: list[int],
    seeds: list[int],
    checkpoint_path: Path,
    base_config: Any,
    output_root: Path,
    args: argparse.Namespace,
    render_selection: set[tuple[str, int, int]],
    completed_keys: set[tuple[str, int, int]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    trajectories: list[Path] = []
    for team_size in team_sizes:
        config = make_eval_config(
            base_config=base_config,
            team_size=team_size,
            gate_count=int(condition["gate_count"]),
            speed_mps=float(condition["speed_mps"]),
            amplitude_m=float(condition["amplitude_m"]),
            gate_post_radius_m=float(args.gate_post_radius_m),
            drone_radius_m=float(args.drone_radius_m),
            disable_guidance_runtime=bool(args.disable_guidance_runtime),
            max_episode_steps=args.max_episode_steps,
            disable_terminal_formation_collapse=bool(args.disable_terminal_formation_collapse),
            formation_line_collapse_min_lateral_bands=args.formation_line_collapse_min_lateral_bands,
        )
        env = create_env(config)
        try:
            agent = build_agent(
                env=env,
                config=config,
                checkpoint_path=checkpoint_path,
                seed=int(seeds[0]) if seeds else 0,
                device=args.device,
            )
            for seed in seeds:
                key = (str(condition["name"]), int(team_size), int(seed))
                if completed_keys is not None and key in completed_keys:
                    print(
                        json.dumps(
                            {
                                "condition": condition["name"],
                                "team_size": int(team_size),
                                "seed": int(seed),
                                "status": "skipped_existing",
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    continue
                save_rollout = (
                    bool(args.render_mp4)
                    and key in render_selection
                )
                row, samples, trajectory_path = run_episode(
                    env=env,
                    agent=agent,
                    config=config,
                    condition_name=str(condition["name"]),
                    condition_kind=str(condition["kind"]),
                    gate_count=int(condition["gate_count"]),
                    speed_mps=float(condition["speed_mps"]),
                    amplitude_m=float(condition["amplitude_m"]),
                    team_size=int(team_size),
                    seed=int(seed),
                    checkpoint_path=checkpoint_path,
                    output_root=output_root,
                    clean_swept_clearance_m=float(args.clean_swept_clearance_m),
                    save_rollout=save_rollout,
                    step_sample_stride=int(args.step_sample_stride),
                )
                rows.append(row)
                step_rows.extend(samples)
                append_jsonl(output_root / "per_episode.jsonl", row)
                append_jsonl_many(output_root / "per_step_sampled.jsonl", samples)
                if completed_keys is not None:
                    completed_keys.add(key)
                if trajectory_path is not None:
                    trajectories.append(trajectory_path)
                print(
                    json.dumps(
                        {
                            "condition": condition["name"],
                            "team_size": int(team_size),
                            "seed": int(seed),
                            "success": row["success"],
                            "clean_success": row["clean_success"],
                            "done_reason": row["done_reason"],
                            "min_swept_gate_clearance_m": row["min_swept_gate_clearance_m"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        finally:
            env.close()
    return rows, step_rows, trajectories


def group_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row.get(item) for item in keys)
        grouped.setdefault(key, []).append(row)
    return grouped


def aggregate_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_team: list[dict[str, Any]] = []
    condition_summary: list[dict[str, Any]] = []
    formation_rows: list[dict[str, Any]] = []
    for key, group in sorted(group_rows(rows, ("condition", "team_size")).items(), key=lambda item: (str(item[0][0]), int(item[0][1]))):
        condition, team_size = key
        done_counts: dict[str, int] = {}
        for row in group:
            reason = str(row.get("done_reason") or "unknown")
            done_counts[reason] = done_counts.get(reason, 0) + 1
        success_values = [float(row.get("success") or 0.0) for row in group]
        clean_success_values = [float(row.get("clean_success") or 0.0) for row in group]
        success_ci = bootstrap_ci(success_values)
        clean_ci = bootstrap_ci(clean_success_values)
        flight_ci = bootstrap_ci([float(row["flight_time_s"]) for row in group if row.get("flight_time_s") is not None])
        clearance_ci = bootstrap_ci(
            [float(row["min_swept_gate_clearance_m"]) for row in group if row.get("min_swept_gate_clearance_m") is not None]
        )
        summary = {
            "condition": condition,
            "condition_kind": group[0].get("condition_kind"),
            "team_size": int(team_size),
            "episodes": len(group),
            "seeds": [int(row["seed"]) for row in group],
            "gate_count": group[0].get("gate_count"),
            "moving_gate_speed_mps": group[0].get("moving_gate_speed_mps"),
            "moving_gate_amplitude_m": group[0].get("moving_gate_amplitude_m"),
            "success_rate_mean": mean(success_values),
            "success_rate_ci95_low": success_ci[0],
            "success_rate_ci95_high": success_ci[1],
            "clean_success_rate_mean": mean(clean_success_values),
            "clean_success_rate_ci95_low": clean_ci[0],
            "clean_success_rate_ci95_high": clean_ci[1],
            "collision_rate_mean": mean([float(row.get("gate_collision") or 0.0) for row in group]),
            "dynamic_gate_collision_rate_mean": mean([float(row.get("dynamic_gate_collision") or 0.0) for row in group]),
            "static_gate_collision_rate_mean": mean([float(row.get("static_gate_collision") or 0.0) for row in group]),
            "agent_collision_rate_mean": mean([float(row.get("agent_collision") or 0.0) for row in group]),
            "timeout_rate_mean": mean([float(row.get("timeout") or 0.0) for row in group]),
            "visual_geometry_collision_violation_rate": mean(
                [float(row.get("visual_geometry_collision_violation") or 0.0) for row in group]
            ),
            "clearance_threshold_violation_rate": mean(
                [float(row.get("clearance_threshold_violation") or 0.0) for row in group]
            ),
            "formation_violation_rate": mean(
                [
                    float(row.get("multi_formation_long_queue_violation") or 0.0)
                    for row in group
                    if row.get("multi_formation_long_queue_violation") is not None
                ]
            ),
            "flight_time_mean_s": mean([float(row["flight_time_s"]) for row in group if row.get("flight_time_s") is not None]),
            "flight_time_std_s": std([float(row["flight_time_s"]) for row in group if row.get("flight_time_s") is not None]),
            "flight_time_ci95_low_s": flight_ci[0],
            "flight_time_ci95_high_s": flight_ci[1],
            "path_length_mean_m": mean([float(row["path_length_m"]) for row in group if row.get("path_length_m") is not None]),
            "path_length_std_m": std([float(row["path_length_m"]) for row in group if row.get("path_length_m") is not None]),
            "min_swept_clearance_mean_m": mean(
                [float(row["min_swept_gate_clearance_m"]) for row in group if row.get("min_swept_gate_clearance_m") is not None]
            ),
            "min_swept_clearance_min_m": min_finite(
                [float(row["min_swept_gate_clearance_m"]) for row in group if row.get("min_swept_gate_clearance_m") is not None]
            ),
            "min_swept_clearance_p05_m": percentile(
                [float(row["min_swept_gate_clearance_m"]) for row in group if row.get("min_swept_gate_clearance_m") is not None],
                5.0,
            ),
            "min_swept_clearance_ci95_low_m": clearance_ci[0],
            "min_swept_clearance_ci95_high_m": clearance_ci[1],
            "min_environment_clearance_mean_m": mean(
                [float(row["min_environment_clearance_m"]) for row in group if row.get("min_environment_clearance_m") is not None]
            ),
            "min_pair_distance_mean_m": mean(
                [float(row["min_pair_distance_m"]) for row in group if row.get("min_pair_distance_m") is not None]
            ),
            "min_pair_distance_min_m": min_finite(
                [float(row["min_pair_distance_m"]) for row in group if row.get("min_pair_distance_m") is not None]
            ),
            "slot_error_mean_m": mean([float(row["mean_slot_error_m"]) for row in group if row.get("mean_slot_error_m") is not None]),
            "slot_error_std_m": std([float(row["mean_slot_error_m"]) for row in group if row.get("mean_slot_error_m") is not None]),
            "shield_activation_ratio_mean": mean(
                [float(row["shield_activation_ratio"]) for row in group if row.get("shield_activation_ratio") is not None]
            ),
            "done_reason_counts": done_counts,
        }
        by_team.append(summary)
        formation_rows.append(
            {
                "condition": condition,
                "team_size": int(team_size),
                "actual_major_span_p95_mean_m": mean(
                    [
                        float(row["multi_formation_actual_major_span_p95_m"])
                        for row in group
                        if row.get("multi_formation_actual_major_span_p95_m") is not None
                    ]
                ),
                "major_ratio_p95_mean": mean(
                    [
                        float(row["multi_formation_major_ratio_p95"])
                        for row in group
                        if row.get("multi_formation_major_ratio_p95") is not None
                    ]
                ),
                "minor_ratio_p05_mean": mean(
                    [
                        float(row["multi_formation_minor_ratio_p05"])
                        for row in group
                        if row.get("multi_formation_minor_ratio_p05") is not None
                    ]
                ),
                "long_queue_violation_rate": summary["formation_violation_rate"],
                "success_but_formation_invalid_count": sum(
                    int(row.get("success_but_formation_invalid") or 0) for row in group
                ),
            }
        )
    for key, group in sorted(group_rows(rows, ("condition",)).items(), key=lambda item: str(item[0][0])):
        condition = key[0]
        condition_summary.append(
            {
                "condition": condition,
                "episodes": len(group),
                "team_sizes": sorted({int(row["team_size"]) for row in group}),
                "success_rate_mean": mean([float(row.get("success") or 0.0) for row in group]),
                "clean_success_rate_mean": mean([float(row.get("clean_success") or 0.0) for row in group]),
                "collision_rate_mean": mean([float(row.get("gate_collision") or 0.0) for row in group]),
                "formation_violation_rate": mean(
                    [
                        float(row.get("multi_formation_long_queue_violation") or 0.0)
                        for row in group
                        if row.get("multi_formation_long_queue_violation") is not None
                    ]
                ),
                "min_swept_clearance_min_m": min_finite(
                    [float(row["min_swept_gate_clearance_m"]) for row in group if row.get("min_swept_gate_clearance_m") is not None]
                ),
            }
        )
    return by_team, condition_summary, formation_rows


def select_static_gate_count(pilot_rows: list[dict[str, Any]]) -> tuple[int, str]:
    if not pilot_rows:
        return DEFAULT_STATIC_GATE_COUNT, "No pilot rows; using default static gate30."
    by_condition = group_rows([row for row in pilot_rows if row.get("condition_kind") == "static"], ("gate_count", "team_size"))
    for candidate in (30, 18):
        rows = by_condition.get((candidate, 7), [])
        if not rows:
            continue
        clean_rate = mean([float(row.get("clean_success") or 0.0) for row in rows]) or 0.0
        geometry_rate = mean([float(row.get("visual_geometry_collision_violation") or 0.0) for row in rows]) or 0.0
        if clean_rate >= 0.6 and geometry_rate <= 0.2:
            return candidate, (
                f"Selected static gate{candidate}: team7 pilot clean_success_rate={clean_rate:.3f}, "
                f"geometry_violation_rate={geometry_rate:.3f}."
            )
    return 18, "Static gate30 did not pass pilot stability; falling back to gate18."


def condition_plan(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pilot: list[dict[str, Any]] = []
    formal: list[dict[str, Any]] = []
    if args.mode in {"pilot", "all"}:
        for gate_count in args.dynamic_pilot_gates:
            pilot.append(
                {
                    "name": f"pilot_dynamic_gate{int(gate_count):02d}",
                    "kind": "dynamic",
                    "gate_count": int(gate_count),
                    "speed_mps": float(DEFAULT_SPEED_MPS),
                    "amplitude_m": float(DEFAULT_AMPLITUDE_M),
                }
            )
        for gate_count in args.static_pilot_gates:
            pilot.append(
                {
                    "name": f"pilot_static_gate{int(gate_count):02d}",
                    "kind": "static",
                    "gate_count": int(gate_count),
                    "speed_mps": 0.0,
                    "amplitude_m": 0.0,
                }
            )
    if args.mode == "smoke":
        formal.append(
            {
                "name": "smoke_dynamic_gate12",
                "kind": "dynamic",
                "gate_count": 12,
                "speed_mps": float(DEFAULT_SPEED_MPS),
                "amplitude_m": float(DEFAULT_AMPLITUDE_M),
            }
        )
    if args.mode in {"formal", "all", "stress"}:
        formal.append(
            {
                "name": f"dynamic_gate{int(args.dynamic_gate_count):02d}",
                "kind": "dynamic",
                "gate_count": int(args.dynamic_gate_count),
                "speed_mps": float(DEFAULT_SPEED_MPS),
                "amplitude_m": float(DEFAULT_AMPLITUDE_M),
            }
        )
        formal.append(
            {
                "name": f"static_gate{int(args.static_gate_count):02d}",
                "kind": "static",
                "gate_count": int(args.static_gate_count),
                "speed_mps": 0.0,
                "amplitude_m": 0.0,
            }
        )
    if args.mode == "stress" or (args.mode == "all" and args.include_stress):
        formal.append(
            {
                "name": "stress_dynamic_gate24",
                "kind": "dynamic",
                "gate_count": 24,
                "speed_mps": float(DEFAULT_SPEED_MPS),
                "amplitude_m": float(DEFAULT_AMPLITUDE_M),
            }
        )
        formal.append(
            {
                "name": "stress_static_gate42",
                "kind": "static",
                "gate_count": 42,
                "speed_mps": 0.0,
                "amplitude_m": 0.0,
            }
        )
    return pilot, formal


def selected_render_keys(
    *,
    formal_conditions: list[dict[str, Any]],
    team_sizes: list[int],
    render_seeds: list[int],
) -> set[tuple[str, int, int]]:
    keys: set[tuple[str, int, int]] = set()
    for condition in formal_conditions:
        for team_size in team_sizes:
            for seed in render_seeds:
                keys.add((str(condition["name"]), int(team_size), int(seed)))
    return keys


def render_rollout_mp4(
    *,
    trajectory_path: Path,
    output_path: Path,
    fps: float,
    frame_stride: int,
    title: str,
) -> dict[str, Any]:
    import imageio.v2 as imageio

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.patches import Circle

    data = np.load(trajectory_path)
    positions = np.asarray(data["positions"], dtype=np.float32)
    desired_slots = np.asarray(data["desired_slots"], dtype=np.float32)
    posts = np.asarray(data["live_gate_posts"], dtype=np.float32)
    path = np.asarray(data["path_waypoints"], dtype=np.float32)
    world_x = np.asarray(data["world_x_bounds_m"], dtype=np.float32)
    world_y = np.asarray(data["world_y_bounds_m"], dtype=np.float32)
    gate_post_radius = float(np.asarray(data["gate_post_radius_m"])[0])
    drone_radius = float(np.asarray(data["drone_radius_m"])[0])
    live_clearance = np.asarray(data["live_clearance"], dtype=np.float32)
    swept_clearance = np.asarray(data["swept_clearance"], dtype=np.float32)
    total_frames = int(positions.shape[0])
    stride = max(int(frame_stride), 1)
    selected_indices = list(range(0, total_frames, stride))
    if selected_indices[-1] != total_frames - 1:
        selected_indices.append(total_frames - 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100)
    canvas = FigureCanvasAgg(fig)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(int(positions.shape[1]), 1)))
    x_pad = 1.0
    y_pad = 1.0
    writer = imageio.get_writer(str(output_path), fps=float(fps), codec="libx264", quality=8)
    try:
        for frame_idx in selected_indices:
            ax.clear()
            ax.set_facecolor("#f7f8fa")
            ax.set_xlim(float(world_x[0]) - x_pad, float(world_x[1]) + x_pad)
            ax.set_ylim(float(world_y[0]) - y_pad, float(world_y[1]) + y_pad)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, color="#d8dde6", linewidth=0.6, alpha=0.8)
            if path.size:
                ax.plot(path[:, 0], path[:, 1], color="#2f6fbb", linewidth=1.6, alpha=0.75)
                ax.scatter(path[0, 0], path[0, 1], s=90, marker="s", color="#1b8a5a", zorder=5)
                ax.scatter(path[-1, 0], path[-1, 1], s=110, marker="*", color="#c43d3d", zorder=5)
            if posts.shape[1] > 0:
                for post in posts[frame_idx]:
                    ax.add_patch(
                        Circle(
                            (float(post[0]), float(post[1])),
                            radius=gate_post_radius,
                            facecolor="#e88c25",
                            edgecolor="#8a4b08",
                            linewidth=0.8,
                            alpha=0.78,
                            zorder=2,
                        )
                    )
                    ax.add_patch(
                        Circle(
                            (float(post[0]), float(post[1])),
                            radius=gate_post_radius + drone_radius,
                            facecolor="none",
                            edgecolor="#d33f49",
                            linewidth=0.6,
                            alpha=0.22,
                            zorder=1,
                        )
                    )
            for agent_idx in range(positions.shape[1]):
                trail_start = max(0, frame_idx - 60)
                trail = positions[trail_start : frame_idx + 1, agent_idx, :]
                color = colors[agent_idx % len(colors)]
                ax.plot(trail[:, 0], trail[:, 1], color=color, linewidth=1.2, alpha=0.5, zorder=3)
                pos = positions[frame_idx, agent_idx]
                slot = desired_slots[frame_idx, agent_idx]
                ax.scatter([slot[0]], [slot[1]], s=36, marker="x", color=color, alpha=0.45, zorder=4)
                ax.add_patch(
                    Circle(
                        (float(pos[0]), float(pos[1])),
                        radius=max(0.18, drone_radius * 0.45),
                        facecolor=color,
                        edgecolor="#111827",
                        linewidth=0.7,
                        alpha=0.95,
                        zorder=6,
                    )
                )
                ax.text(float(pos[0]), float(pos[1]) + 0.34, str(agent_idx + 1), fontsize=7, ha="center", va="bottom")
            live = float(live_clearance[min(frame_idx, live_clearance.shape[0] - 1)])
            swept = float(swept_clearance[min(frame_idx, swept_clearance.shape[0] - 1)])
            ax.set_title(
                f"{title} | frame {frame_idx}/{total_frames - 1} | live={live:.2f}m swept={swept:.2f}m",
                fontsize=10,
            )
            ax.set_xlabel("x [m]")
            ax.set_ylabel("y [m]")
            fig.tight_layout(pad=0.7)
            canvas.draw()
            width, height = canvas.get_width_height()
            rgba = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
            frame = rgba[:, :, :3].copy()
            writer.append_data(frame)
    finally:
        writer.close()
        plt.close(fig)
    return {
        "trajectory_path": str(trajectory_path),
        "mp4_path": str(output_path),
        "frames_source": total_frames,
        "frames_written": len(selected_indices),
        "fps": float(fps),
        "size_bytes": output_path.stat().st_size if output_path.exists() else 0,
    }


def render_mp4s(
    *,
    trajectories: list[Path],
    output_root: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trajectory_path in trajectories:
        meta_path = trajectory_path.with_name("rollout_meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        condition = str(meta.get("condition") or trajectory_path.parents[2].name)
        team_size = int(meta.get("team_size") or 0)
        seed = int(meta.get("seed") or 0)
        mp4_path = output_root / "videos" / condition / f"{condition}_team{team_size:02d}_seed{seed:02d}_top_global.mp4"
        title = f"{condition} team={team_size} seed={seed}"
        row = render_rollout_mp4(
            trajectory_path=trajectory_path,
            output_path=mp4_path,
            fps=float(args.video_fps),
            frame_stride=int(args.video_frame_stride),
            title=title,
        )
        row.update({"condition": condition, "team_size": team_size, "seed": seed})
        rows.append(row)
        print(json.dumps({"rendered_mp4": str(mp4_path), "size_bytes": row["size_bytes"]}, ensure_ascii=False), flush=True)
    JsonWriter.write_json(output_root / "mp4_manifest.json", rows)
    JsonWriter.write_csv(output_root / "mp4_manifest.csv", rows)
    return rows


def plot_metric(
    *,
    rows: list[dict[str, Any]],
    output_path: Path,
    metric: str,
    ylabel: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grouped = group_rows(rows, ("condition",))
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=150)
    for condition, group in sorted(grouped.items(), key=lambda item: str(item[0][0])):
        points = sorted(group, key=lambda row: int(row["team_size"]))
        xs = [int(row["team_size"]) for row in points]
        ys = [row.get(metric) for row in points]
        ax.plot(xs, ys, marker="o", linewidth=1.8, label=str(condition[0]))
    ax.set_xlabel("team size")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#d8dde6", linewidth=0.6)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def write_plots(by_team: list[dict[str, Any]], output_root: Path) -> list[str]:
    plot_specs = [
        ("success_rate_mean", "success rate", "success_rate_vs_team_size.png"),
        ("clean_success_rate_mean", "clean success rate", "clean_success_rate_vs_team_size.png"),
        ("collision_rate_mean", "gate collision rate", "collision_rate_vs_team_size.png"),
        ("timeout_rate_mean", "timeout rate", "timeout_rate_vs_team_size.png"),
        ("flight_time_mean_s", "flight time [s]", "flight_time_vs_team_size.png"),
        ("path_length_mean_m", "path length [m]", "path_length_vs_team_size.png"),
        ("min_swept_clearance_mean_m", "min swept clearance [m]", "min_swept_clearance_vs_team_size.png"),
        ("slot_error_mean_m", "slot error [m]", "slot_error_vs_team_size.png"),
        ("formation_violation_rate", "formation violation rate", "formation_violation_vs_team_size.png"),
    ]
    outputs: list[str] = []
    for metric, ylabel, name in plot_specs:
        path = output_root / "plots" / name
        plot_metric(rows=by_team, output_path=path, metric=metric, ylabel=ylabel)
        outputs.append(str(path))
    return outputs


def write_validation_report(
    *,
    output_root: Path,
    checkpoint_path: Path,
    rows: list[dict[str, Any]],
    by_team: list[dict[str, Any]],
    mp4_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    expected_team_sizes = sorted(set(int(row["team_size"]) for row in rows))
    conditions = sorted(set(str(row["condition"]) for row in rows))
    lines = [
        "# Variable Team Size Evaluation Validation",
        "",
        f"- checkpoint: `{checkpoint_path}`",
        f"- mode: `{args.mode}`",
        f"- conditions: {', '.join(conditions)}",
        f"- team_sizes: {expected_team_sizes}",
        f"- episodes: {len(rows)}",
        f"- inflated gate_post_collision_radius_m: {float(args.gate_post_radius_m):.3f}",
        f"- inflated environment_drone_radius_m: {float(args.drone_radius_m):.3f}",
        f"- clean swept clearance threshold_m: {float(args.clean_swept_clearance_m):.3f}",
        "",
        "## Gates",
        "",
    ]
    problem_rows = [
        row
        for row in rows
        if row.get("success_but_geometry_invalid")
        or row.get("success_but_clearance_below_threshold")
        or row.get("success_but_formation_invalid")
    ]
    if problem_rows:
        lines.append("- Some nominal successes are not clean successes under geometry/formation audit.")
        for row in problem_rows[:30]:
            lines.append(
                f"  - {row['condition']} team={row['team_size']} seed={row['seed']} "
                f"reason={row['done_reason']} swept={row.get('min_swept_gate_clearance_m')} "
                f"long_queue={row.get('multi_formation_long_queue_violation')}"
            )
        if len(problem_rows) > 30:
            lines.append(f"  - truncated: {len(problem_rows) - 30} more")
    else:
        lines.append("- No success was promoted to clean success after a geometry or formation violation.")
    lines.extend(["", "## Summary", ""])
    lines.append("| condition | team | episodes | success | clean_success | collision | formation_violation | min_swept |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in by_team:
        lines.append(
            f"| {row['condition']} | {row['team_size']} | {row['episodes']} | "
            f"{row.get('success_rate_mean')} | {row.get('clean_success_rate_mean')} | "
            f"{row.get('collision_rate_mean')} | {row.get('formation_violation_rate')} | "
            f"{row.get('min_swept_clearance_min_m')} |"
        )
    lines.extend(["", "## MP4", ""])
    if mp4_rows:
        for row in mp4_rows:
            exists = Path(str(row.get("mp4_path"))).exists()
            size = int(row.get("size_bytes") or 0)
            lines.append(
                f"- `{row.get('mp4_path')}` exists={exists} bytes={size} frames={row.get('frames_written')}"
            )
    else:
        lines.append("- MP4 rendering was not requested or no selected trajectories were recorded.")
    (output_root / "validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_config(
    *,
    output_root: Path,
    base_config: Any,
    args: argparse.Namespace,
    pilot_conditions: list[dict[str, Any]],
    formal_conditions: list[dict[str, Any]],
    selected_checkpoint: Path,
) -> None:
    payload = {
        "script": str(Path(__file__).resolve()),
        "selected_checkpoint": str(selected_checkpoint),
        "mode": args.mode,
        "config_name": args.config_name,
        "base_experiment_id": base_config.experiment_id,
        "team_sizes": args.team_sizes,
        "pilot_team_sizes": args.pilot_team_sizes,
        "pilot_seeds": args.pilot_seeds,
        "formal_seeds": args.formal_seeds,
        "render_seeds": args.render_seeds,
        "gate_post_radius_m": args.gate_post_radius_m,
        "drone_radius_m": args.drone_radius_m,
        "clean_swept_clearance_m": args.clean_swept_clearance_m,
        "disable_terminal_formation_collapse": args.disable_terminal_formation_collapse,
        "formation_line_collapse_min_lateral_bands": args.formation_line_collapse_min_lateral_bands,
        "disable_guidance_runtime": args.disable_guidance_runtime,
        "pilot_conditions": pilot_conditions,
        "formal_conditions": formal_conditions,
        "formal_multi_team_sizes_note": {
            "formal_buckets": list(FORMAL_MULTI_TEAM_SIZES),
            "team_size_4_and_6": "interpolation_generalization_test",
            "team_size_1": "OOD_single_agent_no_formation_metrics",
        },
    }
    JsonWriter.write_json(output_root / "config.json", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--config-name", type=str, default="dynamic_gate_density_8d_v1")
    parser.add_argument("--mode", choices=["smoke", "pilot", "formal", "all", "stress"], default="all")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--team-sizes", type=parse_int_range, default=list(DEFAULT_TEAM_SIZES))
    parser.add_argument("--pilot-team-sizes", type=parse_int_range, default=list(DEFAULT_PILOT_TEAM_SIZES))
    parser.add_argument("--pilot-seeds", type=parse_int_range, default=list(range(5)))
    parser.add_argument("--formal-seeds", type=parse_int_range, default=list(range(20)))
    parser.add_argument("--render-seeds", type=parse_int_range, default=[0])
    parser.add_argument("--dynamic-pilot-gates", type=int, nargs="+", default=list(DEFAULT_DYNAMIC_PILOT_GATES))
    parser.add_argument("--static-pilot-gates", type=int, nargs="+", default=list(DEFAULT_STATIC_PILOT_GATES))
    parser.add_argument("--dynamic-gate-count", type=int, default=DEFAULT_DYNAMIC_GATE_COUNT)
    parser.add_argument("--static-gate-count", type=int, default=DEFAULT_STATIC_GATE_COUNT)
    parser.add_argument("--gate-post-radius-m", type=float, default=DEFAULT_GATE_POST_RADIUS_M)
    parser.add_argument("--drone-radius-m", type=float, default=DEFAULT_DRONE_RADIUS_M)
    parser.add_argument("--clean-swept-clearance-m", type=float, default=DEFAULT_CLEAN_SWEPT_CLEARANCE_M)
    parser.add_argument(
        "--disable-terminal-formation-collapse",
        action="store_true",
        help=(
            "Keep rollouts running when the env lateral-band formation-collapse check fires; "
            "formation violations are still audited after rollout."
        ),
    )
    parser.add_argument("--formation-line-collapse-min-lateral-bands", type=int, default=None)
    parser.add_argument("--disable-guidance-runtime", action="store_true", default=True)
    parser.add_argument("--enable-guidance-runtime", dest="disable_guidance_runtime", action="store_false")
    parser.add_argument("--include-stress", action="store_true")
    parser.add_argument("--render-mp4", action="store_true", default=True)
    parser.add_argument("--no-render-mp4", dest="render_mp4", action="store_false")
    parser.add_argument("--video-fps", type=float, default=6.0)
    parser.add_argument("--video-frame-stride", type=int, default=2)
    parser.add_argument("--step-sample-stride", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=500)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    config_name = normalize_multi_experiment_config_name(args.config_name)
    base_config = get_multi_experiment_config(config_name)
    base_eval_config = make_eval_config(
        base_config=base_config,
        team_size=max(2, int(base_config.default_agents)),
        gate_count=int(args.dynamic_gate_count),
        speed_mps=float(DEFAULT_SPEED_MPS),
        amplitude_m=float(DEFAULT_AMPLITUDE_M),
        gate_post_radius_m=float(args.gate_post_radius_m),
        drone_radius_m=float(args.drone_radius_m),
        disable_guidance_runtime=bool(args.disable_guidance_runtime),
        max_episode_steps=args.max_episode_steps,
        disable_terminal_formation_collapse=bool(args.disable_terminal_formation_collapse),
        formation_line_collapse_min_lateral_bands=args.formation_line_collapse_min_lateral_bands,
    )
    selected_checkpoint, checkpoint_audit = audit_checkpoint(
        checkpoint_path=args.checkpoint,
        base_config=base_eval_config,
        output_root=output_root,
        candidate_limit=int(args.candidate_limit),
    )
    pilot_conditions, formal_conditions = condition_plan(args)
    write_config(
        output_root=output_root,
        base_config=base_config,
        args=args,
        pilot_conditions=pilot_conditions,
        formal_conditions=formal_conditions,
        selected_checkpoint=selected_checkpoint,
    )
    all_rows: list[dict[str, Any]] = []
    all_step_rows: list[dict[str, Any]] = []
    trajectories: list[Path] = []
    if bool(args.resume):
        all_rows = load_jsonl(output_root / "per_episode.jsonl")
        all_step_rows = load_jsonl(output_root / "per_step_sampled.jsonl")
    completed_keys = {episode_key(row) for row in all_rows}

    if pilot_conditions and not (bool(args.resume) and (output_root / "obstacle_count_selection_report.json").exists()):
        pilot_render_selection: set[tuple[str, int, int]] = set()
        for condition in pilot_conditions:
            rows, step_rows, condition_trajectories = run_condition(
                condition=condition,
                team_sizes=list(args.pilot_team_sizes),
                seeds=list(args.pilot_seeds),
                checkpoint_path=selected_checkpoint,
                base_config=base_config,
                output_root=output_root,
                args=args,
                render_selection=pilot_render_selection,
                completed_keys=completed_keys,
            )
            all_rows.extend(rows)
            all_step_rows.extend(step_rows)
            trajectories.extend(condition_trajectories)
        selected_static, reason = select_static_gate_count(all_rows)
        JsonWriter.write_json(
            output_root / "obstacle_count_selection_report.json",
            {"selected_static_gate_count": selected_static, "reason": reason},
        )
        (output_root / "obstacle_count_selection_report.md").write_text(
            "# Obstacle Count Selection\n\n"
            f"- selected_static_gate_count: {selected_static}\n"
            f"- reason: {reason}\n",
            encoding="utf-8",
        )
        if args.mode == "all":
            formal_conditions = [
                dict(condition, gate_count=selected_static, name=f"static_gate{selected_static:02d}")
                if condition["kind"] == "static"
                else condition
                for condition in formal_conditions
            ]
    elif pilot_conditions and bool(args.resume) and (output_root / "obstacle_count_selection_report.json").exists():
        selection = json.loads((output_root / "obstacle_count_selection_report.json").read_text(encoding="utf-8"))
        selected_static = int(selection.get("selected_static_gate_count") or args.static_gate_count)
        if args.mode == "all":
            formal_conditions = [
                dict(condition, gate_count=selected_static, name=f"static_gate{selected_static:02d}")
                if condition["kind"] == "static"
                else condition
                for condition in formal_conditions
            ]
    elif args.mode in {"formal", "stress"}:
        JsonWriter.write_json(
            output_root / "obstacle_count_selection_report.json",
            {
                "selected_static_gate_count": int(args.static_gate_count),
                "reason": "Pilot not run in this mode; using requested/default static gate count.",
            },
        )

    render_selection = selected_render_keys(
        formal_conditions=formal_conditions,
        team_sizes=list(args.team_sizes),
        render_seeds=list(args.render_seeds),
    )
    for condition in formal_conditions:
        rows, step_rows, condition_trajectories = run_condition(
            condition=condition,
            team_sizes=list(args.team_sizes),
            seeds=list(args.formal_seeds if args.mode != "smoke" else args.render_seeds),
            checkpoint_path=selected_checkpoint,
            base_config=base_config,
            output_root=output_root,
                args=args,
                render_selection=render_selection,
                completed_keys=completed_keys,
            )
        all_rows.extend(rows)
        all_step_rows.extend(step_rows)
        trajectories.extend(condition_trajectories)

    if bool(args.resume):
        all_rows = load_jsonl(output_root / "per_episode.jsonl")
        all_step_rows = load_jsonl(output_root / "per_step_sampled.jsonl")
        trajectories = [
            Path(str(row["trajectory_path"]))
            for row in all_rows
            if row.get("trajectory_path") and Path(str(row["trajectory_path"])).exists()
        ]

    by_team, condition_summary, formation_rows = aggregate_rows(all_rows)
    JsonWriter.write_csv(output_root / "per_episode.csv", all_rows)
    JsonWriter.write_json(output_root / "per_episode.json", all_rows)
    JsonWriter.write_csv(output_root / "per_step_sampled.csv", all_step_rows)
    JsonWriter.write_csv(output_root / "eval_by_team_size.csv", by_team)
    JsonWriter.write_json(output_root / "eval_by_team_size.json", by_team)
    JsonWriter.write_csv(output_root / "condition_summary.csv", condition_summary)
    JsonWriter.write_csv(output_root / "formation_audit.csv", formation_rows)
    plot_paths = write_plots(by_team, output_root)
    mp4_rows = render_mp4s(trajectories=trajectories, output_root=output_root, args=args) if args.render_mp4 else []
    write_validation_report(
        output_root=output_root,
        checkpoint_path=selected_checkpoint,
        rows=all_rows,
        by_team=by_team,
        mp4_rows=mp4_rows,
        args=args,
    )
    JsonWriter.write_json(
        output_root / "run_summary.json",
        {
            "output_root": str(output_root),
            "checkpoint": str(selected_checkpoint),
            "checkpoint_sha256": checkpoint_audit.get("selected", {}).get("sha256"),
            "episodes": len(all_rows),
            "conditions": sorted(set(str(row["condition"]) for row in all_rows)),
            "plots": plot_paths,
            "mp4_count": len(mp4_rows),
            "validation_report": str(output_root / "validation_report.md"),
        },
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "episodes": len(all_rows),
                "mp4_count": len(mp4_rows),
                "validation_report": str(output_root / "validation_report.md"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()



