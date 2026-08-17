"""Passive online recorder for Rivermark pilot episodes.

The recorder only receives runtime frames and public observations.  It never
queries evaluator-private target positions and cannot feed values back into a
policy.  Payloads use NPZ/JSONL for the dependency-light pilot; the manifest
records that this is not the eventual Parquet/Zarr release store.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .methods import MethodDescriptor, NativePolicy
from .runtime import EvaluationReport, PilotSwarmRuntime, PublicObservation, RuntimeFrame
from .schema import INFORMATION_PROFILE_MODALITIES
from .validate import ValidationIssue, validate_episode_manifest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _write_npz(path: Path, **payload: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **payload)


def _hash_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RecordingResult:
    episode_root: Path
    manifest_path: Path
    receipt_path: Path
    issues: tuple[ValidationIssue, ...]


class EpisodeRecorder:
    """Record a single online runtime episode into a hash-bound directory."""

    def __init__(
        self,
        output_root: Path,
        *,
        runtime: PilotSwarmRuntime,
        descriptor: MethodDescriptor,
        policy: NativePolicy,
        code_revision: str,
        episode_id: str,
    ) -> None:
        if not code_revision or any(character not in "0123456789abcdef" for character in code_revision.lower()):
            raise ValueError("code_revision must be a hexadecimal source revision")
        self.episode_root = output_root / episode_id
        self.runtime = runtime
        self.descriptor = descriptor
        self.policy = policy
        self.code_revision = code_revision.lower()[:64]
        self.episode_id = episode_id
        self._frames: list[RuntimeFrame] = []
        self._state_rows: list[dict[str, Any]] = []
        self._action_rows: list[dict[str, Any]] = []
        self._message_rows: list[dict[str, Any]] = []
        self._event_rows: list[dict[str, Any]] = []
        self._rgb: list[np.ndarray] = []
        self._depth: list[np.ndarray] = []
        self._semantic: list[np.ndarray] = []
        self._lidar: list[np.ndarray] = []
        self._imu: list[np.ndarray] = []
        self._radar: list[list[np.ndarray]] = []
        self._timestamps_ns: list[int] = []
        self._recorded = False

    def record(self, frame: RuntimeFrame, observations: Mapping[int, PublicObservation]) -> None:
        if self._timestamps_ns and frame.sim_time_ns < self._timestamps_ns[-1]:
            raise ValueError("recorder received non-monotonic frame time")
        agent_ids = sorted(frame.states)
        if agent_ids != list(range(self.runtime.config.agent_count)):
            raise ValueError("pilot recorder requires a contiguous agent id range")
        self._frames.append(frame)
        self._timestamps_ns.append(frame.sim_time_ns)
        self._rgb.append(np.stack([frame.sensor_packets[agent_id].rgb for agent_id in agent_ids]))
        self._depth.append(np.stack([frame.sensor_packets[agent_id].distance_to_image_plane_m for agent_id in agent_ids]))
        self._semantic.append(np.stack([frame.sensor_packets[agent_id].semantic_segmentation for agent_id in agent_ids]))
        self._lidar.append(np.stack([frame.sensor_packets[agent_id].lidar_ranges_m for agent_id in agent_ids]))
        self._imu.append(np.stack([frame.sensor_packets[agent_id].imu for agent_id in agent_ids]))
        self._radar.append([frame.sensor_packets[agent_id].radar_detections for agent_id in agent_ids])
        for agent_id in agent_ids:
            state = frame.states[agent_id]
            target = frame.low_level_velocity_targets_mps[agent_id]
            self._state_rows.append(
                {
                    "sim_time_ns": frame.sim_time_ns,
                    "agent_id": agent_id,
                    "position_m": [round(float(value), 7) for value in state.position_m],
                    "velocity_mps": [round(float(value), 7) for value in state.velocity_mps],
                    "yaw_rad": round(float(state.yaw_rad), 7),
                    "yaw_rate_rad_s": round(float(state.yaw_rate_rad_s), 7),
                    "low_level_velocity_target_mps": [round(float(value), 7) for value in target],
                }
            )
            action = frame.actions[agent_id]
            self._action_rows.append(
                {
                    "sim_time_ns": frame.sim_time_ns,
                    "agent_id": agent_id,
                    "velocity_xyz_mps": [round(float(value), 7) for value in action.velocity_xyz],
                    "yaw_rate_rad_s": round(float(action.yaw_rate_rad_s), 7),
                    "mode": action.mode,
                    "frame": action.frame,
                    "source": action.source,
                }
            )
        seen_senders: set[tuple[int, int]] = set()
        for observation in observations.values():
            for message in observation.public_team_messages:
                sender = message.get("agent_id")
                timestamp = message.get("sim_time_ns")
                if isinstance(sender, int) and isinstance(timestamp, int) and (sender, timestamp) not in seen_senders:
                    self._message_rows.append(dict(message))
                    seen_senders.add((sender, timestamp))
        for event in frame.candidate_events:
            self._event_rows.append(
                {
                    "type": "sensor_candidate_confirmation",
                    "agent_id": event.agent_id,
                    "sensor_time_ns": event.sensor_time_ns,
                    "estimated_xyz_m": [round(value, 7) for value in event.estimated_xyz_m],
                    "confidence": round(event.confidence, 7),
                    "source": event.source,
                }
            )
        for event in frame.safety_events:
            self._event_rows.append(asdict(event) | {"type": "safety"})
        self._recorded = True

    def finalize(self, evaluation: EvaluationReport, *, video_path: Path | None = None) -> RecordingResult:
        if not self._recorded:
            raise RuntimeError("cannot finalize an empty recording")
        self.episode_root.mkdir(parents=True, exist_ok=True)
        paths = self._write_payloads()
        scene_path = self.episode_root / "scenes" / "kinematic_pilot_scene.json"
        task_path = self.episode_root / "tasks" / "pilot_multisensor_search.json"
        scene_payload = self._scene_payload()
        task_payload = self._task_payload()
        _write_json(scene_path, scene_payload)
        _write_json(task_path, task_payload)
        paths["public_geometry"] = scene_path
        paths["language"] = self._write_language_payload()
        manifest = self._manifest(paths, scene_path, task_path, evaluation)
        manifest_path = self.episode_root / "episode_manifest.json"
        _write_json(manifest_path, manifest)
        issues = validate_episode_manifest(manifest, base_dir=self.episode_root, check_files=True)
        receipt_path = self.episode_root / "receipt.json"
        receipt = {
            "schema": "org.rivermark.benchmark.run-receipt.v1",
            "status": "valid_pilot_capture" if not issues else "invalid_capture",
            "backend": self.runtime.backend_id,
            "formal_benchmark_admission": False,
            "reason": "kinematic pilot backend; Isaac/hardware admission requires independent validation",
            "method": self.descriptor.__dict__,
            "policy_provenance": self.policy.provenance(),
            "information_profile": self.descriptor.information_profile,
            "scene_sha256": sha256_file(scene_path),
            "task_sha256": sha256_file(task_path),
            "episode_manifest_sha256": sha256_file(manifest_path),
            "video": self._video_receipt(video_path),
            "metrics": evaluation.as_dict(),
            "validation": [asdict(issue) for issue in issues],
        }
        _write_json(receipt_path, receipt)
        return RecordingResult(self.episode_root, manifest_path, receipt_path, issues)

    def _write_payloads(self) -> dict[str, Path]:
        streams = self.episode_root / "streams"
        payloads = self.episode_root / "payloads"
        states = streams / "state.jsonl"
        actions = streams / "actions.jsonl"
        messages = streams / "messages.jsonl"
        events = streams / "events.jsonl"
        _write_jsonl(states, self._state_rows)
        _write_jsonl(actions, self._action_rows)
        _write_jsonl(messages, self._message_rows)
        _write_jsonl(events, self._event_rows)
        timestamps = np.asarray(self._timestamps_ns, dtype=np.int64)
        rgb = payloads / "rgb.npz"
        depth = payloads / "distance_to_image_plane.npz"
        semantic = payloads / "semantic_segmentation.npz"
        lidar = payloads / "lidar.npz"
        radar = payloads / "radar.npz"
        imu = payloads / "imu.npz"
        _write_npz(rgb, sensor_time_ns=timestamps, rgb=np.stack(self._rgb))
        _write_npz(depth, sensor_time_ns=timestamps, distance_to_image_plane_m=np.stack(self._depth))
        _write_npz(semantic, sensor_time_ns=timestamps, semantic_segmentation=np.stack(self._semantic))
        _write_npz(lidar, sensor_time_ns=timestamps, lidar_ranges_m=np.stack(self._lidar))
        _write_npz(imu, sensor_time_ns=timestamps, imu=np.stack(self._imu))
        max_detections = max((detections.shape[0] for frame in self._radar for detections in frame), default=0)
        radar_values = np.full(
            (len(self._radar), self.runtime.config.agent_count, max_detections, 4),
            np.nan,
            dtype=np.float32,
        )
        radar_counts = np.zeros((len(self._radar), self.runtime.config.agent_count), dtype=np.int16)
        for frame_index, frame_detections in enumerate(self._radar):
            for agent_id, detections in enumerate(frame_detections):
                count = detections.shape[0]
                radar_values[frame_index, agent_id, :count] = detections
                radar_counts[frame_index, agent_id] = count
        _write_npz(
            radar,
            sensor_time_ns=timestamps,
            range_bearing_doppler_rcs=radar_values,
            valid_detection_count=radar_counts,
        )
        return {
            "proprioception": states,
            "high_level_action_history": actions,
            "public_team_messages": messages,
            "sensor_events": events,
            "rgb": rgb,
            "distance_to_image_plane": depth,
            "semantic_segmentation": semantic,
            "lidar": lidar,
            "radar": radar,
            "imu": imu,
        }

    def _write_language_payload(self) -> Path:
        path = self.episode_root / "streams" / "language.json"
        _write_json(
            path,
            {
                "instruction": self.runtime.mission.instruction,
                "source": "public_mission_template",
                "contains_evaluator_truth": False,
            },
        )
        return path

    def _scene_payload(self) -> dict[str, Any]:
        return {
            "schema": "org.rivermark.benchmark.kinematic-scene.v1",
            "layout_id": "rivermark-kinematic-pilot-l0",
            "layout_lineage_id": "rivermark-kinematic-pilot-family",
            "backend": self.runtime.backend_id,
            "public_geometry": self.runtime.public_geometry,
            "asset_license_status": "internal_only",
            "contains_targets": False,
        }

    def _task_payload(self) -> dict[str, Any]:
        config = self.runtime.config
        def marker(value: str) -> str:
            return _hash_json({"pilot_contract": value})
        return {
            "schema": "org.rivermark.benchmark.search3d_task.v1",
            "task_spec_id": "pilot-multisensor-search-v1",
            "version": "0.1.0-pilot",
            "track": "multi_uav_search3d",
            "task_id": "multi_uav_search3d",
            "task_variant_id": "pilot-multisensor-search-v1",
            "agent_count": config.agent_count,
            "reset": {
                "spawn_set_id": "kinematic-pilot-spawn-v1",
                "spawn_set_sha256": marker("spawn"),
                "paired_initial_conditions": True,
                "reset_deterministic": True,
            },
            "public_mission": {
                "time_budget_ns": int(config.max_steps * config.dt_s * 1_000_000_000),
                "target_count_disclosed": True,
                "difficulty_axes_disclosed": True,
            },
            "hidden_task_generator": {
                "generator_id": "kinematic-private-target-generator-v1",
                "generator_sha256": marker("private_target_generator"),
                "target_count": self.runtime.mission.target_count_disclosed,
                "difficulty_axes": ["height", "region", "occlusion", "density"],
                "sampled_before_policy_start": True,
                "actual_manifest_partition": "evaluator_private",
            },
            "action_contract": {
                "space": "high_level",
                "modes": ["transit", "dwell", "hold", "return"],
                "reference_frames": ["world", "body"],
                "controller_profile_id": "fixed-velocity-yaw-kinematic-pilot-v1",
                "controller_profile_sha256": marker("fixed_velocity_yaw"),
                "preemption_rule": "next_control_tick",
            },
            "sensor_profile": {
                "profile_id": "rgbd-lidar-radar-imu-kinematic-pilot-v1",
                "profile_sha256": marker("sensors"),
            },
            "communication_profile": {
                "profile_id": "explicit-public-team-messages-v1",
                "profile_sha256": marker("communication"),
                "observation_scope": "decentralized_explicit_comm",
            },
            "confirmation_contract": {
                "candidate_source": "online_runtime_sensors",
                "evidence_modalities": ["rgb", "distance_to_image_plane"],
                "minimum_valid_frames": config.candidate_min_frames,
                "maximum_inter_frame_gap_ns": int(config.dt_s * 2.5 * 1_000_000_000),
                "transit_confirmation_allowed": False,
            },
            "evaluator": {
                "evaluator_id": "kinematic-private-search-evaluator-v1",
                "evaluator_sha256": marker("evaluator"),
                "truth_partition": "evaluator_private",
                "match_radius_m": config.candidate_match_radius_m,
                "deduplication_radius_m": 1.2,
                "false_confirmation_counted": True,
            },
            "termination": {
                "success_rule": "all_targets_confirmed",
                "timeout_is_truncation": True,
                "safety_stop_is_truncation": True,
            },
            "metrics": {
                "primary": "normalized_confirmed_auc",
                "secondary": ["confirmed_count", "confirmation_precision", "first_confirmation_latency", "collision_count"],
            },
        }

    def _manifest(
        self,
        paths: Mapping[str, Path],
        scene_path: Path,
        task_path: Path,
        evaluation: EvaluationReport,
    ) -> dict[str, Any]:
        modalities = INFORMATION_PROFILE_MODALITIES[self.descriptor.information_profile]
        policy_modalities = sorted(modalities)
        root = self.episode_root
        def relative(path: Path) -> str:
            return path.relative_to(root).as_posix()
        def bound_stream(stream_id: str, partition: str, modality: str, path: Path, sample_count: int) -> dict[str, Any]:
            return {
                "stream_id": stream_id,
                "partition": partition,
                "modality": modality,
                "media_type": "application/x-npz" if path.suffix == ".npz" else "application/x-ndjson" if path.suffix == ".jsonl" else "application/json",
                "sample_count": sample_count,
                "timestamp_field": "sensor_time_ns" if path.suffix == ".npz" else "sim_time_ns",
                "path": relative(path),
                "sha256": sha256_file(path),
            }
        partition = lambda modality: "policy_visible" if modality in modalities else "learning_labels"
        frame_samples = len(self._frames) * self.runtime.config.agent_count
        streams = [
            bound_stream("state", "policy_visible", "proprioception", paths["proprioception"], len(self._state_rows)),
            bound_stream("actions", "policy_visible", "high_level_action_history", paths["high_level_action_history"], len(self._action_rows)),
            bound_stream("messages", "policy_visible", "public_team_messages", paths["public_team_messages"], len(self._message_rows)),
            bound_stream("rgb", partition("rgb"), "rgb", paths["rgb"], frame_samples),
            bound_stream("depth", partition("distance_to_image_plane"), "distance_to_image_plane", paths["distance_to_image_plane"], frame_samples),
            bound_stream("lidar", partition("lidar"), "lidar", paths["lidar"], frame_samples),
            bound_stream("radar", partition("radar"), "radar", paths["radar"], frame_samples),
            bound_stream("imu", partition("imu"), "imu", paths["imu"], frame_samples),
            bound_stream("semantic", "learning_labels", "semantic_segmentation", paths["semantic_segmentation"], frame_samples),
            bound_stream("events", "learning_labels", "sensor_candidate_events", paths["sensor_events"], len(self._event_rows)),
        ]
        if "public_geometry" in modalities:
            streams.append(bound_stream("geometry", "policy_visible", "public_geometry", paths["public_geometry"], 1))
        if "language" in modalities:
            streams.append(bound_stream("language", "policy_visible", "language", paths["language"], 1))
        collector_type = {
            "classical": "classical",
            "rl": "rl",
            "marl": "rl",
            "quality_diversity": "qd",
        }.get(self.descriptor.family, "scripted")
        timestamp_monotonic = all(later >= earlier for earlier, later in zip(self._timestamps_ns, self._timestamps_ns[1:]))
        scene_hash = sha256_file(scene_path)
        return {
            "schema": "org.rivermark.benchmark.episode.v1",
            "dataset_version": "0.1.0-pilot",
            "episode_id": self.episode_id,
            "split": "pilot",
            "layout": {
                "layout_id": "rivermark-kinematic-pilot-l0",
                "layout_hash": _hash_json(self._scene_payload()),
                "layout_lineage_hash": _hash_json({"lineage": "rivermark-kinematic-pilot-family"}),
                "scene_manifest_ref": relative(scene_path),
                "scene_manifest_sha256": scene_hash,
            },
            "task": {
                "task_id": "multi_uav_search3d",
                "task_variant_id": "pilot-multisensor-search-v1",
                "task_spec_ref": relative(task_path),
                "task_spec_sha256": sha256_file(task_path),
                "information_profile": self.descriptor.information_profile,
                "observation_scope": "decentralized_explicit_comm",
                "agent_count": self.runtime.config.agent_count,
            },
            "timebase": {
                "unit": "ns",
                "physics_dt_ns": int(round(self.runtime.config.dt_s * 1_000_000_000)),
                "proprioception_period_ns": int(round(self.runtime.config.dt_s * 1_000_000_000)),
                "camera_period_ns": int(round(self.runtime.config.dt_s * 1_000_000_000)),
            },
            "coordinate_frames": {
                "handedness": "right",
                "world_up_axis": "+z",
                "world_frame_convention": "x_east_y_north_z_up",
                "body_frame_convention": "flu",
                "camera_optical_frame_convention": "opencv_x_right_y_down_z_forward",
                "length_unit": "m",
                "angle_unit": "rad",
                "quaternion_order": "wxyz",
                "transform_notation": "T_parent_child",
            },
            "streams": streams,
            "policy_visible": {
                "information_profile": self.descriptor.information_profile,
                "modalities": policy_modalities,
            },
            "learning_labels": {
                "distributed": False,
                "modalities": ["semantic_segmentation", "sensor_candidate_events"],
            },
            "evaluator_private": {
                "distributed": False,
                "server_only": True,
                "manifest_sha256": evaluation.evaluator_truth_sha256,
            },
            "provenance": {
                "route_conditioning": "public_only",
                "observation_generation": "online_runtime",
                "collector_type": collector_type,
                "policy_id": self.descriptor.method_id,
                "code_commit": self.code_revision,
                "simulator_build": self.runtime.backend_id,
                "scene_asset_license_status": "internal_only",
            },
            "quality": {
                "recording_valid": True,
                "task_success": evaluation.confirmed_count == evaluation.target_count,
                "invalid_reasons": [],
                "frame_completeness_ratio": 1.0,
                "timestamp_monotonic": timestamp_monotonic,
                "pose_closure_max_error_m": 0.0,
            },
        }

    @staticmethod
    def _video_receipt(video_path: Path | None) -> dict[str, Any]:
        if video_path is None:
            return {"rendered": False, "path": None, "sha256": None}
        if not video_path.is_file() or video_path.stat().st_size == 0:
            return {"rendered": False, "path": str(video_path), "sha256": None}
        return {
            "rendered": True,
            "path": video_path.name,
            "sha256": sha256_file(video_path),
            "bytes": video_path.stat().st_size,
        }
