"""Small, actually trainable multimodal pilot models and checkpoint policies.

These are deliberately not presented as foundation models.  They are compact
PyTorch baselines that make the VLM/VLA/world-model data paths executable in a
resource-bounded pilot: RGB plus public language and state feed an action
chunk, RGB plus language feeds a grounding head, and a state-action MLP feeds
MPC.  Every checkpoint has metadata and a SHA-256 provenance record.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .runtime import HighLevelAction, PublicMission, PublicObservation

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - tested through fail-closed commands.
    torch = None
    nn = None


def _require_torch() -> Any:
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for learned Rivermark pilot models")
    return torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def language_token(instruction: str | None, *, vocabulary_size: int = 64) -> int:
    """Stable public-language token; it never encodes hidden task truth."""

    text = (instruction or "").encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:4], "little") % vocabulary_size


class _VisionLanguageBackbone(nn.Module if nn is not None else object):
    """Small sensor-fusion backbone for the closed multisensor pilot profile.

    This deliberately consumes the full public sensor set rather than merely
    declaring it in metadata.  Radar is represented by a fixed public summary
    because the pilot produces a variable number of detections per tick.
    """

    def __init__(self, *, language_vocab: int = 64, language_dim: int = 12) -> None:
        _require_torch()
        super().__init__()
        assert nn is not None
        self.visual = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.depth = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 12, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.language = nn.Embedding(language_vocab, language_dim)
        self.state = nn.Sequential(nn.Linear(8, 24), nn.ReLU())
        self.lidar = nn.Sequential(nn.Linear(72, 16), nn.ReLU())
        self.radar = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
        self.imu = nn.Sequential(nn.Linear(6, 8), nn.ReLU())
        self.output_dim = 32 + 12 + language_dim + 24 + 16 + 8 + 8

    def forward_features(
        self,
        rgb: Any,
        depth: Any,
        lidar: Any,
        radar: Any,
        imu: Any,
        token: Any,
        state: Any,
    ) -> Any:
        visual = self.visual(rgb).flatten(1)
        depth_features = self.depth(depth).flatten(1)
        return torch.cat(
            (
                visual,
                depth_features,
                self.language(token),
                self.state(state),
                self.lidar(lidar),
                self.radar(radar),
                self.imu(imu),
            ),
            dim=1,
        )


class TinyVisionLanguageActionNet(nn.Module if nn is not None else object):
    """RGB + public language + proprioception to continuous action chunks."""

    def __init__(self, *, chunk_size: int = 3) -> None:
        _require_torch()
        super().__init__()
        assert nn is not None
        self.chunk_size = chunk_size
        self.backbone = _VisionLanguageBackbone()
        self.head = nn.Sequential(
            nn.Linear(self.backbone.output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, chunk_size * 4),
            nn.Tanh(),
        )

    def forward(
        self,
        rgb: Any,
        depth: Any,
        lidar: Any,
        radar: Any,
        imu: Any,
        token: Any,
        state: Any,
    ) -> Any:
        return self.head(
            self.backbone.forward_features(rgb, depth, lidar, radar, imu, token, state)
        ).reshape((-1, self.chunk_size, 4))


class TinyVisionLanguageGrounderNet(nn.Module if nn is not None else object):
    """RGB + public language to marker visibility and image-plane direction."""

    def __init__(self) -> None:
        _require_torch()
        super().__init__()
        assert nn is not None
        self.backbone = _VisionLanguageBackbone()
        self.head = nn.Sequential(nn.Linear(self.backbone.output_dim, 48), nn.ReLU(), nn.Linear(48, 3))

    def forward(
        self,
        rgb: Any,
        depth: Any,
        lidar: Any,
        radar: Any,
        imu: Any,
        token: Any,
        state: Any,
    ) -> Any:
        return self.head(self.backbone.forward_features(rgb, depth, lidar, radar, imu, token, state))


class TinyActionDynamicsNet(nn.Module if nn is not None else object):
    """Action-conditioned prediction from public state and sensor summaries."""

    def __init__(self) -> None:
        _require_torch()
        super().__init__()
        assert nn is not None
        self.net = nn.Sequential(
            nn.Linear(31, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
        )

    def forward(self, state: Any, action: Any, sensor_context: Any) -> Any:
        return self.net(torch.cat((state, action, sensor_context), dim=1))


def _load_checkpoint(
    path: Path,
    expected_kind: str,
    expected_profile: str,
    metadata_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    library = _require_torch()
    checkpoint = path.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint is missing: {checkpoint}")
    metadata_file = (metadata_path or checkpoint.with_suffix(".rivermark.json")).resolve()
    if not metadata_file.is_file():
        raise FileNotFoundError(f"checkpoint metadata is missing: {metadata_file}")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    if metadata.get("schema") != "org.rivermark.torch-pilot.v1":
        raise ValueError("unsupported learned-pilot metadata schema")
    if metadata.get("model_kind") != expected_kind:
        raise ValueError(f"expected a {expected_kind} checkpoint, got {metadata.get('model_kind')!r}")
    if metadata.get("information_profile") != expected_profile:
        raise ValueError(
            f"{expected_kind} checkpoint declares {metadata.get('information_profile')!r}, "
            f"not required profile {expected_profile!r}"
        )
    expected_hash = metadata.get("checkpoint_sha256")
    if not isinstance(expected_hash, str) or expected_hash != sha256_file(checkpoint):
        raise ValueError("checkpoint SHA-256 does not match its immutable metadata")
    try:
        payload = library.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:  # Older torch releases do not implement weights_only.
        payload = library.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError("learned-pilot checkpoint lacks a state_dict")
    return payload, metadata, metadata_file


class _TorchPolicyBase:
    def __init__(
        self,
        checkpoint: Path,
        *,
        expected_kind: str,
        expected_profile: str,
        metadata_path: Path | None = None,
    ) -> None:
        self.checkpoint = checkpoint.resolve()
        self.payload, self.metadata, self.metadata_path = _load_checkpoint(
            self.checkpoint, expected_kind, expected_profile, metadata_path
        )
        self.expected_kind = expected_kind
        self.expected_profile = expected_profile
        self.mission: PublicMission | None = None
        self.agent_count = 0

    def reset(
        self,
        mission: PublicMission,
        agent_count: int,
        *,
        public_geometry: Mapping[str, Any] | None = None,
    ) -> None:
        del public_geometry
        self.mission = mission
        self.agent_count = agent_count

    def provenance(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "implementation_kind": "trained_torch_pilot_checkpoint",
            "external_dependency": "torch",
            "checkpoint": str(self.checkpoint),
            "checkpoint_sha256": sha256_file(self.checkpoint),
            "adapter_metadata": str(self.metadata_path),
            "adapter_metadata_sha256": sha256_file(self.metadata_path),
            "model_kind": self.expected_kind,
            "training_backend": self.metadata.get("training_backend"),
            "training_sample_count": self.metadata.get("training_sample_count"),
        }


def _rgb_tensor(rgb: np.ndarray) -> Any:
    library = _require_torch()
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("VLM/VLA policy needs HxWx3 RGB")
    return library.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().unsqueeze(0) / 255.0


def _depth_tensor(depth: np.ndarray) -> Any:
    library = _require_torch()
    if depth.ndim != 2:
        raise ValueError("VLM/VLA policy needs HxW distance-to-image-plane")
    finite = np.nan_to_num(depth, nan=18.0, posinf=18.0, neginf=0.0)
    return library.from_numpy(np.ascontiguousarray(finite)).float().unsqueeze(0).unsqueeze(0) / 18.0


def _lidar_tensor(lidar: np.ndarray) -> Any:
    library = _require_torch()
    if lidar.shape != (72,):
        raise ValueError("VLM/VLA policy needs a 72-beam LiDAR scan")
    return library.from_numpy(np.ascontiguousarray(lidar)).float().unsqueeze(0) / 18.0


def radar_summary(detections: np.ndarray) -> np.ndarray:
    """Create a fixed, public summary from variable-length radar detections."""

    if detections.ndim != 2 or detections.shape[1:] != (4,):
        raise ValueError("radar detections must have shape (N, 4)")
    if detections.shape[0] == 0:
        return np.zeros(4, dtype=np.float32)
    finite = detections[np.all(np.isfinite(detections), axis=1)]
    if finite.shape[0] == 0:
        return np.zeros(4, dtype=np.float32)
    return np.asarray(
        (
            min(1.0, finite.shape[0] / 8.0),
            float(np.clip(np.min(finite[:, 0]) / 18.0, 0.0, 1.0)),
            float(np.clip(np.mean(finite[:, 2]) / 3.0, -1.0, 1.0)),
            float(np.clip(np.max(finite[:, 3]), 0.0, 1.0)),
        ),
        dtype=np.float32,
    )


def _radar_tensor(detections: np.ndarray) -> Any:
    library = _require_torch()
    return library.from_numpy(radar_summary(detections)).float().unsqueeze(0)


def _imu_tensor(imu: np.ndarray) -> Any:
    library = _require_torch()
    if imu.shape != (6,):
        raise ValueError("VLM/VLA policy needs a 6D IMU vector")
    return library.from_numpy(np.ascontiguousarray(imu)).float().unsqueeze(0)


def world_sensor_context(
    rgb: np.ndarray,
    depth: np.ndarray,
    lidar: np.ndarray,
    radar: np.ndarray,
    imu: np.ndarray,
) -> np.ndarray:
    """Public multimodal features used by the compact dynamics model.

    This is intentionally a small state predictor, not a video world model.
    RGB is represented by its public channel means; depth, LiDAR, radar, and
    IMU all enter the learned dynamics prediction through documented features.
    """

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("world model needs HxWx3 RGB")
    if depth.ndim != 2 or lidar.shape != (72,) or imu.shape != (6,):
        raise ValueError("world model received malformed public sensors")
    finite_depth = np.nan_to_num(depth, nan=18.0, posinf=18.0, neginf=0.0)
    lidar_values = np.nan_to_num(lidar, nan=18.0, posinf=18.0, neginf=0.0)
    center = lidar_values.shape[0] // 2
    lidar_features = np.asarray(
        (
            np.min(lidar_values) / 18.0,
            np.mean(lidar_values) / 18.0,
            np.min(lidar_values[max(0, center - 4) : center + 5]) / 18.0,
            np.std(lidar_values) / 18.0,
        ),
        dtype=np.float32,
    )
    depth_features = np.asarray(
        (np.min(finite_depth) / 18.0, np.mean(finite_depth) / 18.0), dtype=np.float32
    )
    return np.concatenate(
        (
            rgb.astype(np.float32).mean(axis=(0, 1)) / 255.0,
            depth_features,
            lidar_features,
            radar_summary(radar),
            imu.astype(np.float32),
        )
    ).astype(np.float32, copy=False)


def _state_tensor(observation: PublicObservation) -> Any:
    library = _require_torch()
    return library.from_numpy(np.ascontiguousarray(observation.proprioception)).float().unsqueeze(0)


def _token_tensor(observation: PublicObservation) -> Any:
    library = _require_torch()
    return library.tensor([language_token(observation.language)], dtype=library.long)


class TinyVlaCheckpointPolicy(_TorchPolicyBase):
    """A trained visual-language action-chunk policy for the pilot runtime."""

    method_id = "tiny_vla_checkpoint"

    def __init__(self, checkpoint: Path, metadata_path: Path | None = None) -> None:
        super().__init__(
            checkpoint,
            expected_kind="tiny_vla_action_chunk",
            expected_profile="language_multisensor_rgbd_lidar_radar_state",
            metadata_path=metadata_path,
        )
        chunk_size = int(self.metadata.get("chunk_size", 3))
        if chunk_size < 1 or chunk_size > 16:
            raise ValueError("invalid VLA action chunk length")
        self.model = TinyVisionLanguageActionNet(chunk_size=chunk_size)
        self.model.load_state_dict(self.payload["state_dict"])
        self.model.eval()
        self._scale = np.asarray(self.metadata.get("action_scale", [2.3, 2.3, 1.25, 1.4]), dtype=np.float32)
        if self._scale.shape != (4,) or np.any(self._scale <= 0.0):
            raise ValueError("VLA metadata needs positive 4D action_scale")
        self._chunks: dict[int, list[np.ndarray]] = {}

    def reset(self, mission: PublicMission, agent_count: int, *, public_geometry: Mapping[str, Any] | None = None) -> None:
        super().reset(mission, agent_count, public_geometry=public_geometry)
        self._chunks = {agent_id: [] for agent_id in range(agent_count)}

    def act(self, observations: Mapping[int, PublicObservation]) -> Mapping[int, HighLevelAction]:
        library = _require_torch()
        actions: dict[int, HighLevelAction] = {}
        for agent_id, observation in observations.items():
            if (
                observation.information_profile != "language_multisensor_rgbd_lidar_radar_state"
                or observation.rgb is None
                or observation.distance_to_image_plane_m is None
                or observation.lidar_ranges_m is None
                or observation.radar_detections is None
                or observation.imu is None
                or observation.language is None
            ):
                raise RuntimeError("VLA checkpoint requires the full public language/multisensor profile")
            if not self._chunks[agent_id]:
                with library.no_grad():
                    chunk = self.model(
                        _rgb_tensor(observation.rgb),
                        _depth_tensor(observation.distance_to_image_plane_m),
                        _lidar_tensor(observation.lidar_ranges_m),
                        _radar_tensor(observation.radar_detections),
                        _imu_tensor(observation.imu),
                        _token_tensor(observation),
                        _state_tensor(observation),
                    )[0]
                self._chunks[agent_id] = [row.detach().cpu().numpy().astype(np.float32) for row in chunk]
            command = np.clip(self._chunks[agent_id].pop(0), -1.0, 1.0) * self._scale
            actions[agent_id] = HighLevelAction(
                velocity_xyz=tuple(float(value) for value in command[:3]),
                yaw_rate_rad_s=float(command[3]),
                mode="transit",
                source=self.method_id,
            )
        return actions


class TinyVlmGroundingCheckpointPolicy(_TorchPolicyBase):
    """A trained visual-language grounding policy with public fallback search."""

    method_id = "tiny_vlm_grounding_checkpoint"

    def __init__(self, checkpoint: Path, metadata_path: Path | None = None) -> None:
        super().__init__(
            checkpoint,
            expected_kind="tiny_vlm_grounder",
            expected_profile="language_multisensor_rgbd_lidar_radar_state",
            metadata_path=metadata_path,
        )
        self.model = TinyVisionLanguageGrounderNet()
        self.model.load_state_dict(self.payload["state_dict"])
        self.model.eval()

    def _fallback_velocity(self, observation: PublicObservation) -> np.ndarray:
        assert self.mission is not None
        width, height = self.mission.bounds_xy_m
        position = observation.proprioception[:3].astype(np.float64)
        lane_y = 1.8 + (observation.agent_id + 0.5) * (height - 3.6) / max(1, self.agent_count)
        direction = 1.0 if ((observation.sim_time_ns // 2_000_000_000 + observation.agent_id) % 2 == 0) else -1.0
        return np.array((direction * 1.75, np.clip((lane_y - position[1]) * 0.45, -0.9, 0.9), np.clip((2.8 - position[2]) * 0.7, -0.8, 0.8)))

    def act(self, observations: Mapping[int, PublicObservation]) -> Mapping[int, HighLevelAction]:
        library = _require_torch()
        actions: dict[int, HighLevelAction] = {}
        for agent_id, observation in observations.items():
            if (
                observation.information_profile != "language_multisensor_rgbd_lidar_radar_state"
                or observation.rgb is None
                or observation.distance_to_image_plane_m is None
                or observation.lidar_ranges_m is None
                or observation.radar_detections is None
                or observation.imu is None
                or observation.language is None
            ):
                raise RuntimeError("VLM checkpoint requires the full public language/multisensor profile")
            with library.no_grad():
                output = self.model(
                    _rgb_tensor(observation.rgb),
                    _depth_tensor(observation.distance_to_image_plane_m),
                    _lidar_tensor(observation.lidar_ranges_m),
                    _radar_tensor(observation.radar_detections),
                    _imu_tensor(observation.imu),
                    _token_tensor(observation),
                    _state_tensor(observation),
                )[0]
            visibility = float(library.sigmoid(output[0]).cpu())
            direction = library.tanh(output[1:3]).cpu().numpy()
            yaw = float(observation.proprioception[6])
            forward = np.array((math.cos(yaw), math.sin(yaw)))
            lateral = np.array((-math.sin(yaw), math.cos(yaw)))
            if visibility >= 0.5:
                velocity = np.array((forward[0] * 2.05 + lateral[0] * float(direction[0]) * 1.1, forward[1] * 2.05 + lateral[1] * float(direction[0]) * 1.1, float(direction[1]) * 0.65))
            else:
                velocity = self._fallback_velocity(observation)
            desired_yaw = math.atan2(float(velocity[1]), float(velocity[0]))
            yaw_error = (desired_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi
            actions[agent_id] = HighLevelAction(
                velocity_xyz=tuple(float(value) for value in velocity),
                yaw_rate_rad_s=float(np.clip(yaw_error * 1.4, -1.4, 1.4)),
                mode="transit",
                source=self.method_id,
            )
        return actions


class LearnedWorldModelMpcCheckpointPolicy(_TorchPolicyBase):
    """A trained action-conditioned state model used to rank MPC candidates."""

    method_id = "learned_world_model_mpc_checkpoint"

    def __init__(self, checkpoint: Path, metadata_path: Path | None = None) -> None:
        super().__init__(
            checkpoint,
            expected_kind="tiny_action_world_model",
            expected_profile="multisensor_rgbd_lidar_radar_state",
            metadata_path=metadata_path,
        )
        self.model = TinyActionDynamicsNet()
        self.model.load_state_dict(self.payload["state_dict"])
        self.model.eval()
        self.input_mean = np.asarray(self.metadata.get("input_mean"), dtype=np.float32)
        self.input_std = np.asarray(self.metadata.get("input_std"), dtype=np.float32)
        self.output_mean = np.asarray(self.metadata.get("output_mean"), dtype=np.float32)
        self.output_std = np.asarray(self.metadata.get("output_std"), dtype=np.float32)
        if any(array.shape != (8,) for array in (self.input_mean, self.input_std, self.output_mean, self.output_std)):
            raise ValueError("world-model metadata requires 8D normalization arrays")
        if np.any(self.input_std <= 0.0) or np.any(self.output_std <= 0.0):
            raise ValueError("world-model normalization standard deviations must be positive")
        if self.metadata.get("sensor_context_dim") != 19:
            raise ValueError("world-model metadata requires the 19D public sensor context")

    def _goal(self, observation: PublicObservation) -> np.ndarray:
        assert self.mission is not None
        width, height = self.mission.bounds_xy_m
        lane_y = 1.7 + (observation.agent_id + 0.5) * (height - 3.4) / max(1, self.agent_count)
        phase = (observation.sim_time_ns // 4_000_000_000 + observation.agent_id) % 2
        return np.array((width - 2.0 if phase == 0 else 2.0, lane_y, 2.75))

    def _predict_delta(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        library = _require_torch()
        normalized_state = (state - self.input_mean) / self.input_std
        with library.no_grad():
            output = self.model(
                library.from_numpy(normalized_state.astype(np.float32)).unsqueeze(0),
                library.from_numpy(action.astype(np.float32)).unsqueeze(0),
                library.from_numpy(self._sensor_context.astype(np.float32)).unsqueeze(0),
            )[0].cpu().numpy()
        return output * self.output_std + self.output_mean

    def act(self, observations: Mapping[int, PublicObservation]) -> Mapping[int, HighLevelAction]:
        actions: dict[int, HighLevelAction] = {}
        for agent_id, observation in observations.items():
            if (
                observation.information_profile != "multisensor_rgbd_lidar_radar_state"
                or observation.rgb is None
                or observation.distance_to_image_plane_m is None
                or observation.lidar_ranges_m is None
                or observation.radar_detections is None
                or observation.imu is None
            ):
                raise RuntimeError("world-model checkpoint requires the full public multisensor profile")
            state = observation.proprioception.astype(np.float32)
            self._sensor_context = world_sensor_context(
                observation.rgb,
                observation.distance_to_image_plane_m,
                observation.lidar_ranges_m,
                observation.radar_detections,
                observation.imu,
            )
            goal = self._goal(observation)
            delta = goal - state[:3]
            norm = max(float(np.linalg.norm(delta[:2])), 1e-6)
            nominal = np.array((delta[0] / norm * 2.25, delta[1] / norm * 2.25, np.clip(delta[2], -0.85, 0.85), 0.0), dtype=np.float32)
            candidates = (
                nominal,
                nominal + np.array((0.0, 0.8, 0.0, 0.0), dtype=np.float32),
                nominal + np.array((0.0, -0.8, 0.0, 0.0), dtype=np.float32),
                nominal + np.array((0.0, 0.0, 0.45, 0.0), dtype=np.float32),
            )
            clearance = float(np.min(observation.lidar_ranges_m)) if observation.lidar_ranges_m is not None else 3.0
            def score(action: np.ndarray) -> float:
                predicted = state + self._predict_delta(state, action)
                return -float(np.linalg.norm(goal - predicted[:3])) - max(0.0, 1.2 - clearance) * 7.0
            chosen = max(candidates, key=score)
            yaw = float(state[6])
            desired_yaw = math.atan2(float(chosen[1]), float(chosen[0]))
            yaw_error = (desired_yaw - yaw + math.pi) % (2.0 * math.pi) - math.pi
            actions[agent_id] = HighLevelAction(
                velocity_xyz=tuple(float(value) for value in chosen[:3]),
                yaw_rate_rad_s=float(np.clip(yaw_error * 1.4, -1.4, 1.4)),
                mode="transit",
                source=self.method_id,
            )
        return actions
