"""Collect and load hash-bound public-only pilot data for learned baselines.

The collector uses the same closed-loop runtime and passive ``EpisodeRecorder``
as demos.  Its output is intentionally pilot data, not a formal release.  The
loader verifies every episode manifest and every referenced payload hash before
it exposes tensors to a trainer.  It never reads evaluator-private truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .methods import NATIVE_DESCRIPTORS, NativePolicy, create_native_policy
from .provenance import source_revision
from .recording import EpisodeRecorder, _write_json
from .runtime import PilotRuntimeConfig, PilotSwarmRuntime
from .schema import INFORMATION_PROFILE_MODALITIES
from .validate import validate_episode_manifest


# These are stable researcher-facing names.  Manifest names remain accepted
# as aliases so callers do not need to know the on-disk stream vocabulary.
_MODALITY_ALIASES = {
    "rgb": "rgb",
    "depth": "depth",
    "distance_to_image_plane": "depth",
    "semantic": "semantic",
    "semantic_segmentation": "semantic",
    "lidar": "lidar",
    "radar": "radar",
    "imu": "imu",
    "state": "state",
    "proprioception": "state",
    "action": "action",
    "high_level_action_history": "action",
    "language": "language",
}
_ALL_MODALITIES = frozenset({"rgb", "depth", "semantic", "lidar", "radar", "imu", "state", "action", "language"})
_STREAM_MODALITIES = {
    "rgb": "rgb",
    "depth": "distance_to_image_plane",
    "semantic": "semantic_segmentation",
    "lidar": "lidar",
    "radar": "radar",
    "imu": "imu",
    "state": "proprioception",
    "action": "high_level_action_history",
    "language": "language",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _stream_path(manifest: Mapping[str, Any], modality: str, root: Path) -> Path:
    for stream in manifest.get("streams", []):
        if stream.get("modality") == modality and isinstance(stream.get("path"), str):
            return root / stream["path"]
    raise ValueError(f"episode manifest has no concrete stream for {modality!r}")


def _require_profile(manifest: Mapping[str, Any], profile: str) -> None:
    task = manifest.get("task")
    policy = manifest.get("policy_visible")
    if not isinstance(task, Mapping) or task.get("information_profile") != profile:
        raise ValueError(f"episode task profile must be {profile!r}")
    if not isinstance(policy, Mapping) or policy.get("information_profile") != profile:
        raise ValueError(f"episode policy profile must be {profile!r}")
    expected = INFORMATION_PROFILE_MODALITIES[profile]
    if set(policy.get("modalities", [])) != expected:
        raise ValueError("episode has a non-closed policy-visible modality set")


def _require_valid_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    issues = validate_episode_manifest(manifest, base_dir=path.parent, check_files=True)
    if issues:
        formatted = "; ".join(f"{issue.code}:{issue.path}" for issue in issues)
        raise ValueError(f"invalid source episode {path}: {formatted}")
    return manifest


@dataclass(frozen=True)
class PilotEpisode:
    """Validated pilot episode material available to public/learning training."""

    root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    rgb: np.ndarray | None
    depth: np.ndarray | None
    semantic: np.ndarray | None
    lidar: np.ndarray | None
    radar: np.ndarray | None
    radar_counts: np.ndarray | None
    imu: np.ndarray | None
    states: np.ndarray | None
    actions: np.ndarray | None
    action_sources: tuple[str, ...]
    language: str
    timestamps_ns: np.ndarray
    agent_ids: tuple[int, ...]

    @property
    def frame_count(self) -> int:
        return int(self.timestamps_ns.shape[0])

    @property
    def agent_count(self) -> int:
        return len(self.agent_ids)

    @property
    def sample_count(self) -> int:
        return self.frame_count * self.agent_count

    def source_binding(self) -> dict[str, str]:
        return {
            "episode_manifest": str(self.manifest_path.resolve()),
            "episode_manifest_sha256": sha256_file(self.manifest_path),
        }


def _normalise_modalities(modalities: Iterable[str] | None) -> frozenset[str]:
    if modalities is None:
        return _ALL_MODALITIES
    values = tuple(modalities)
    if not values:
        raise ValueError("modalities must not be empty")
    if any(not isinstance(value, str) for value in values):
        raise TypeError("modalities must contain strings")
    selected: set[str] = set()
    for value in values:
        canonical = _MODALITY_ALIASES.get(value.strip().lower())
        if canonical is None:
            raise ValueError(f"unknown modality {value!r}; choose from {sorted(_ALL_MODALITIES)}")
        if canonical in selected:
            raise ValueError(f"duplicate modality {value!r}")
        selected.add(canonical)
    return frozenset(selected)


def _normalise_agent_ids(agent_ids: Iterable[int] | None, agent_count: int) -> tuple[int, ...]:
    if agent_ids is None:
        return tuple(range(agent_count))
    values = tuple(agent_ids)
    if not values:
        raise ValueError("agent_ids must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("agent_ids must contain integers")
    if len(set(values)) != len(values):
        raise ValueError("agent_ids must be unique")
    if any(value < 0 or value >= agent_count for value in values):
        raise ValueError(f"agent_ids must be within [0, {agent_count})")
    return values


def load_pilot_episode(
    manifest_path: Path,
    *,
    required_profile: str | None = None,
    modalities: Iterable[str] | None = None,
    agent_ids: Iterable[int] | None = None,
) -> PilotEpisode:
    """Open a validated episode, optionally selecting modalities and agents.

    ``modalities=None`` preserves the historical full-load behavior.  A
    selection still validates the complete manifest and file bindings, but
    only decodes requested NPZ members and returns ``None`` for omitted arrays.
    """

    resolved = manifest_path.resolve()
    manifest = _require_valid_manifest(resolved)
    if required_profile is not None:
        _require_profile(manifest, required_profile)
    root = resolved.parent
    manifest_modalities = {
        stream.get("modality") for stream in manifest.get("streams", []) if isinstance(stream, Mapping)
    }
    selected = _normalise_modalities(modalities)
    if modalities is None:
        # Preserve historical "load everything available" semantics across
        # profiles: language and geometry are optional manifest streams.
        selected = frozenset(
            canonical for canonical, stream_modality in _STREAM_MODALITIES.items()
            if stream_modality in manifest_modalities
        )
    for canonical in selected:
        stream_modality = _STREAM_MODALITIES[canonical]
        if stream_modality not in manifest_modalities:
            raise ValueError(f"episode manifest has no stream for requested modality {canonical!r}")

    def npz(modality: str) -> Mapping[str, np.ndarray]:
        with np.load(_stream_path(manifest, modality, root), allow_pickle=False) as payload:
            return {key: payload[key].copy() for key in payload.files}

    sensor_payloads: dict[str, Mapping[str, np.ndarray]] = {}
    for canonical in selected & frozenset(_STREAM_MODALITIES) - {"state", "action", "language"}:
        sensor_payloads[canonical] = npz(_STREAM_MODALITIES[canonical])
    timestamps: np.ndarray | None = None
    for payload in sensor_payloads.values():
        candidate = payload.get("sensor_time_ns")
        if candidate is not None:
            if candidate.ndim != 1:
                raise ValueError("sensor payload lacks one-dimensional sensor_time_ns")
            if timestamps is None:
                timestamps = candidate
            elif not np.array_equal(timestamps, candidate):
                raise ValueError("selected sensor timestamps do not agree")
    agent_count = int(manifest["task"]["agent_count"])
    selected_agent_ids = _normalise_agent_ids(agent_ids, agent_count)

    state_rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    need_rows = bool(selected & {"state", "action"}) or timestamps is None
    if need_rows:
        if timestamps is None and "state" not in selected and "action" not in selected:
            # JSONL is the small canonical timebase when language-only reads
            # are requested; no large sensor member is decoded as a fallback.
            state_rows = _read_jsonl(_stream_path(manifest, "proprioception", root))
        else:
            if "state" in selected or timestamps is None:
                state_rows = _read_jsonl(_stream_path(manifest, "proprioception", root))
            if "action" in selected:
                action_rows = _read_jsonl(_stream_path(manifest, "high_level_action_history", root))
    if timestamps is None:
        if not state_rows or agent_count < 1 or len(state_rows) % agent_count:
            raise ValueError("cannot derive frame timebase from state payload")
        timestamps = np.asarray(
            [int(state_rows[index]["sim_time_ns"]) for index in range(0, len(state_rows), agent_count)],
            dtype=np.int64,
        )
    frame_count = int(timestamps.shape[0])
    if frame_count < 1:
        raise ValueError("episode must contain at least one frame")
    arrays = {canonical: payload for canonical, payload in sensor_payloads.items()}
    def array(canonical: str, key: str) -> np.ndarray | None:
        payload = arrays.get(canonical)
        value = payload.get(key) if payload is not None else None
        if value is not None and (value.ndim < 2 or value.shape[0] != frame_count):
            raise ValueError(f"{canonical} payload frame count does not agree")
        return value

    rgb = array("rgb", "rgb")
    depth = array("depth", "distance_to_image_plane_m")
    semantic = array("semantic", "semantic_segmentation")
    lidar = array("lidar", "lidar_ranges_m")
    radar = array("radar", "range_bearing_doppler_rcs")
    radar_counts = array("radar", "valid_detection_count")
    imu = array("imu", "imu")
    if rgb is not None and (rgb.ndim != 5 or rgb.shape[-1] != 3):
        raise ValueError("unexpected RGB payload dimensions")
    if depth is not None and depth.ndim != 4:
        raise ValueError("unexpected depth payload dimensions")
    if semantic is not None and semantic.ndim != 4:
        raise ValueError("unexpected semantic payload dimensions")
    for value in (rgb, depth, semantic, lidar, radar, radar_counts, imu):
        if value is not None and value.shape[1] != agent_count:
            raise ValueError("payload and manifest agent counts disagree")

    states: np.ndarray | None = None
    actions: np.ndarray | None = None
    sources: list[str] = []
    expected_rows = frame_count * agent_count
    if "state" in selected or "action" in selected:
        if "state" in selected and not state_rows:
            state_rows = _read_jsonl(_stream_path(manifest, "proprioception", root))
        if "action" in selected and not action_rows:
            action_rows = _read_jsonl(_stream_path(manifest, "high_level_action_history", root))
        if "state" in selected and len(state_rows) != expected_rows:
            raise ValueError("state row count does not match sensor payload")
        if "action" in selected and len(action_rows) != expected_rows:
            raise ValueError("action row count does not match sensor payload")
        if "state" in selected:
            states = np.empty((frame_count, len(selected_agent_ids), 8), dtype=np.float32)
        if "action" in selected:
            actions = np.empty((frame_count, len(selected_agent_ids), 4), dtype=np.float32)
        for frame_index in range(frame_count):
            for output_agent_index, agent_id in enumerate(selected_agent_ids):
                row_index = frame_index * agent_count + agent_id
                state_row = state_rows[row_index] if "state" in selected else None
                action_row = action_rows[row_index] if "action" in selected else None
                if state_row is not None:
                    if state_row.get("agent_id") != agent_id or state_row.get("sim_time_ns") != int(timestamps[frame_index]):
                        raise ValueError("state rows are not ordered by frame then agent")
                    position = np.asarray(state_row.get("position_m"), dtype=np.float32)
                    velocity = np.asarray(state_row.get("velocity_mps"), dtype=np.float32)
                    if position.shape != (3,) or velocity.shape != (3,):
                        raise ValueError("malformed state row")
                    assert states is not None
                    states[frame_index, output_agent_index] = np.concatenate((position, velocity, np.asarray((state_row.get("yaw_rad"), state_row.get("yaw_rate_rad_s")), dtype=np.float32)))
                if action_row is not None:
                    if action_row.get("agent_id") != agent_id or action_row.get("sim_time_ns") != int(timestamps[frame_index]):
                        raise ValueError("action rows are not ordered by frame then agent")
                    action = np.asarray(action_row.get("velocity_xyz_mps") + [action_row.get("yaw_rate_rad_s")], dtype=np.float32)
                    if action.shape != (4,) or not np.all(np.isfinite(action)):
                        raise ValueError("malformed action row")
                    assert actions is not None
                    actions[frame_index, output_agent_index] = action
                    sources.append(str(action_row.get("source", "")))

    def select_agents(value: np.ndarray | None) -> np.ndarray | None:
        return value[:, selected_agent_ids, ...] if value is not None else None

    language = ""
    if "language" in selected:
        language_payload = _read_json(_stream_path(manifest, "language", root))
        language = language_payload.get("instruction")
        if not isinstance(language, str) or not language:
            raise ValueError("language payload lacks a public instruction")
    return PilotEpisode(
        root=root,
        manifest_path=resolved,
        manifest=manifest,
        rgb=select_agents(rgb),
        depth=select_agents(depth),
        semantic=select_agents(semantic),
        lidar=select_agents(lidar),
        radar=select_agents(radar),
        radar_counts=select_agents(radar_counts),
        imu=select_agents(imu),
        states=states,
        actions=actions,
        action_sources=tuple(sources),
        language=language,
        timestamps_ns=timestamps.astype(np.int64, copy=True),
        agent_ids=selected_agent_ids,
    )


def load_pilot_episodes(paths: Iterable[Path], *, required_profile: str | None = None) -> tuple[PilotEpisode, ...]:
    episodes = tuple(load_pilot_episode(path, required_profile=required_profile) for path in paths)
    if not episodes:
        raise ValueError("at least one source episode manifest is required")
    return episodes


def _teacher_policy(method_id: str) -> tuple[NativePolicy, Any]:
    descriptor = NATIVE_DESCRIPTORS.get(method_id)
    if descriptor is None:
        raise KeyError(f"unknown native teacher method: {method_id}")
    if descriptor.information_profile not in {
        "language_multisensor_rgbd_lidar_radar_state",
        "multisensor_rgbd_lidar_radar_state",
    }:
        raise ValueError("dataset teacher must use a closed multisensor profile")
    return create_native_policy(method_id), descriptor


def collect_episode(
    output_root: Path,
    *,
    episode_id: str,
    teacher_method: str,
    agent_count: int,
    max_steps: int,
    seed: int,
    overwrite: bool = False,
) -> Path:
    """Collect one passive, hash-bound episode using an allowed native teacher."""

    if agent_count < 1 or max_steps < 2:
        raise ValueError("agent_count must be positive and max_steps must be at least two")
    episode_root = output_root / episode_id
    if episode_root.exists():
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite existing episode: {episode_root}")
        shutil.rmtree(episode_root)
    policy, descriptor = _teacher_policy(teacher_method)
    runtime = PilotSwarmRuntime(
        PilotRuntimeConfig(agent_count=agent_count, max_steps=max_steps, seed=seed),
        information_profile=descriptor.information_profile,
    )
    observations = runtime.reset()
    policy.reset(runtime.mission, agent_count)
    recorder = EpisodeRecorder(
        output_root,
        runtime=runtime,
        descriptor=descriptor,
        policy=policy,
        code_revision=source_revision(),
        episode_id=episode_id,
    )
    recorder.record(runtime.current_frame(), observations)
    while not runtime.done:
        observations, frame = runtime.step(policy.act(observations))
        recorder.record(frame, observations)
    result = recorder.finalize(runtime.evaluate())
    if result.issues:
        raise RuntimeError("collector generated an invalid episode: " + "; ".join(issue.code for issue in result.issues))
    return result.manifest_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts") / "datasets" / "torch-pilot")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--agents", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=36)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--teacher",
        choices=(
            "action_chunk_vla_pilot",
            "grounded_vln_pilot",
            "vlm_grounded_search_pilot",
            "action_conditioned_world_model_mpc_pilot",
        ),
        default="action_chunk_vla_pilot",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.episodes < 1:
        raise SystemExit("--episodes must be positive")
    manifests: list[dict[str, str]] = []
    for index in range(args.episodes):
        episode_id = f"{args.teacher}-seed{args.seed + index:08d}"
        manifest_path = collect_episode(
            args.output_root,
            episode_id=episode_id,
            teacher_method=args.teacher,
            agent_count=args.agents,
            max_steps=args.max_steps,
            seed=args.seed + index,
            overwrite=args.overwrite,
        )
        manifests.append({"path": str(manifest_path), "sha256": sha256_file(manifest_path)})
    index_path = args.output_root / "dataset_index.json"
    _write_json(
        index_path,
        {
            "schema": "org.rivermark.torch-pilot-dataset.v1",
            "backend": "rivermark-kinematic-pilot-v1",
            "formal_benchmark_admission": False,
            "teacher_method": args.teacher,
            "episodes": manifests,
        },
    )
    print(json.dumps({"status": "completed", "index": str(index_path), "episode_count": len(manifests)}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
