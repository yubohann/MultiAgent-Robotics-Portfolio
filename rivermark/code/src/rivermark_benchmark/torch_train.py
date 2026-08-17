"""Train provenance-bound compact multimodal Rivermark pilot checkpoints.

This module trains three intentionally small models from ``dataset.py``
episodes.  They are real PyTorch checkpoints, but not external foundation
models, Isaac captures, or formal benchmark results.  Every run writes a
metadata sidecar that binds all source manifests and their hashes before the
checkpoint's own SHA-256 is written.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .dataset import PilotEpisode, load_pilot_episodes, sha256_file
from .learned import (
    TinyActionDynamicsNet,
    TinyVisionLanguageActionNet,
    TinyVisionLanguageGrounderNet,
    language_token,
    radar_summary,
    world_sensor_context,
)
from .provenance import detect_source_provenance

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - exercised through CLI fail-closed behavior.
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


ACTION_SCALE = np.asarray((2.3, 2.3, 1.25, 1.4), dtype=np.float32)
VLA_PROFILE = "language_multisensor_rgbd_lidar_radar_state"
WORLD_PROFILE = "multisensor_rgbd_lidar_radar_state"
METADATA_SCHEMA = "org.rivermark.torch-pilot.v1"


def _require_torch() -> Any:
    if torch is None or nn is None or DataLoader is None or TensorDataset is None:
        raise RuntimeError("PyTorch is required to train Rivermark torch-pilot checkpoints")
    return torch


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    library = _require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=path.parent, suffix=".tmp") as stream:
        temporary = Path(stream.name)
    try:
        library.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def _normalizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, np.float32(1e-4))


def _radar_at(episode: PilotEpisode, frame_index: int, agent_id: int) -> np.ndarray:
    count = int(episode.radar_counts[frame_index, agent_id])
    return episode.radar[frame_index, agent_id, :count].astype(np.float32, copy=True)


def _image_batch(values: list[np.ndarray], *, channels: int) -> np.ndarray:
    data = np.stack(values).astype(np.float32)
    if channels == 3:
        return np.ascontiguousarray(data.transpose(0, 3, 1, 2) / 255.0)
    return np.ascontiguousarray(data[:, None, :, :] / 18.0)


def _sensor_arrays(episodes: Iterable[PilotEpisode], *, profile: str) -> dict[str, np.ndarray]:
    rgb: list[np.ndarray] = []
    depth: list[np.ndarray] = []
    lidar: list[np.ndarray] = []
    radar: list[np.ndarray] = []
    imu: list[np.ndarray] = []
    states: list[np.ndarray] = []
    tokens: list[int] = []
    for episode in episodes:
        for frame_index in range(episode.frame_count):
            for agent_id in range(episode.agent_count):
                rgb.append(episode.rgb[frame_index, agent_id])
                depth.append(episode.depth[frame_index, agent_id])
                lidar.append(episode.lidar[frame_index, agent_id] / 18.0)
                radar.append(radar_summary(_radar_at(episode, frame_index, agent_id)))
                imu.append(episode.imu[frame_index, agent_id])
                states.append(episode.states[frame_index, agent_id])
                tokens.append(language_token(episode.language))
    if not rgb:
        raise ValueError("source data has no public sensor samples")
    return {
        "rgb": _image_batch(rgb, channels=3),
        "depth": _image_batch(depth, channels=1),
        "lidar": np.asarray(lidar, dtype=np.float32),
        "radar": np.asarray(radar, dtype=np.float32),
        "imu": np.asarray(imu, dtype=np.float32),
        "state": np.asarray(states, dtype=np.float32),
        "token": np.asarray(tokens, dtype=np.int64),
        "profile": np.asarray([profile]),
    }


def _source_metadata(episodes: Sequence[PilotEpisode]) -> list[dict[str, str]]:
    return [episode.source_binding() for episode in episodes]


def _train_model(
    model: Any,
    dataset: Any,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    loss_fn: Any,
) -> list[float]:
    library = _require_torch()
    generator = library.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True, generator=generator)
    optimizer = library.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    losses: list[float] = []
    for _ in range(epochs):
        running_loss = 0.0
        count = 0
        for batch in loader:
            *inputs, target = batch
            optimizer.zero_grad(set_to_none=True)
            prediction = model(*inputs)
            loss = loss_fn(prediction, target)
            loss.backward()
            library.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            running_loss += float(loss.detach().cpu()) * target.shape[0]
            count += int(target.shape[0])
        losses.append(running_loss / max(count, 1))
    model.eval()
    return losses


def _save_checkpoint(
    checkpoint: Path,
    *,
    model: Any,
    metadata: dict[str, Any],
) -> tuple[Path, Path]:
    _atomic_torch_save(checkpoint, {"state_dict": model.state_dict()})
    metadata_path = checkpoint.with_suffix(".rivermark.json")
    final_metadata = dict(metadata)
    final_metadata["checkpoint_sha256"] = sha256_file(checkpoint)
    _atomic_json(metadata_path, final_metadata)
    return checkpoint, metadata_path


def _common_metadata(
    *,
    model_kind: str,
    information_profile: str,
    episodes: Sequence[PilotEpisode],
    sample_count: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    source = detect_source_provenance()
    return {
        "schema": METADATA_SCHEMA,
        "model_kind": model_kind,
        "implementation_kind": "trained_torch_pilot_checkpoint",
        "training_backend": "rivermark-kinematic-pilot-v1",
        "formal_benchmark_admission": False,
        "information_profile": information_profile,
        "training_sample_count": sample_count,
        "source_episodes": _source_metadata(episodes),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "source_revision": source.source_revision,
        "source_tree_sha256": source.source_tree_sha256,
        "source_worktree_dirty": source.source_worktree_dirty,
    }


@dataclass(frozen=True)
class TrainResult:
    checkpoint: Path
    metadata: Path
    model_kind: str
    sample_count: int
    final_loss: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint),
            "metadata": str(self.metadata),
            "model_kind": self.model_kind,
            "sample_count": self.sample_count,
            "final_loss": self.final_loss,
        }


def train_vla(
    episode_paths: Sequence[Path],
    output: Path,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    chunk_size: int = 3,
) -> TrainResult:
    library = _require_torch()
    if chunk_size < 1 or chunk_size > 8:
        raise ValueError("chunk_size must be in [1, 8]")
    episodes = load_pilot_episodes(episode_paths, required_profile=VLA_PROFILE)
    arrays = _sensor_arrays(episodes, profile=VLA_PROFILE)
    targets: list[np.ndarray] = []
    valid_indices: list[int] = []
    offset = 0
    for episode in episodes:
        for frame_index in range(episode.frame_count):
            for agent_id in range(episode.agent_count):
                # The recorder writes the action that produced each frame.
                # Observation t therefore supervises action t+1, not the
                # reset/previous action stored beside observation t.
                if frame_index + chunk_size >= episode.frame_count:
                    offset += 1
                    continue
                actions = episode.actions[frame_index + 1 : frame_index + 1 + chunk_size, agent_id]
                targets.append(np.clip(actions / ACTION_SCALE, -1.0, 1.0))
                valid_indices.append(offset)
                offset += 1
    if not targets:
        raise ValueError("source episodes are shorter than requested VLA action chunk")
    index = np.asarray(valid_indices, dtype=np.intp)
    tensor_dataset = TensorDataset(
        library.from_numpy(arrays["rgb"][index]),
        library.from_numpy(arrays["depth"][index]),
        library.from_numpy(arrays["lidar"][index]),
        library.from_numpy(arrays["radar"][index]),
        library.from_numpy(arrays["imu"][index]),
        library.from_numpy(arrays["token"][index]),
        library.from_numpy(arrays["state"][index]),
        library.from_numpy(np.asarray(targets, dtype=np.float32)),
    )
    model = TinyVisionLanguageActionNet(chunk_size=chunk_size)
    losses = _train_model(
        model,
        tensor_dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        loss_fn=library.nn.MSELoss(),
    )
    metadata = _common_metadata(
        model_kind="tiny_vla_action_chunk",
        information_profile=VLA_PROFILE,
        episodes=episodes,
        sample_count=len(tensor_dataset),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    metadata.update({"chunk_size": chunk_size, "action_scale": ACTION_SCALE.tolist(), "final_training_loss": losses[-1]})
    checkpoint, metadata_path = _save_checkpoint(output, model=model, metadata=metadata)
    return TrainResult(checkpoint, metadata_path, "tiny_vla_action_chunk", len(tensor_dataset), losses[-1])


def train_vlm(
    episode_paths: Sequence[Path],
    output: Path,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> TrainResult:
    library = _require_torch()
    episodes = load_pilot_episodes(episode_paths, required_profile=VLA_PROFILE)
    arrays = _sensor_arrays(episodes, profile=VLA_PROFILE)
    labels: list[np.ndarray] = []
    for episode in episodes:
        for frame_index in range(episode.frame_count):
            for agent_id in range(episode.agent_count):
                mask = episode.semantic[frame_index, agent_id] == 2
                visible = float(np.any(mask))
                if visible:
                    columns = np.nonzero(mask)[1]
                    rows = np.nonzero(mask)[0]
                    horizontal = float(np.clip((float(columns.mean()) / (mask.shape[1] - 1)) * 2.0 - 1.0, -1.0, 1.0))
                    vertical = float(np.clip(1.0 - (float(rows.mean()) / (mask.shape[0] - 1)) * 2.0, -1.0, 1.0))
                else:
                    horizontal, vertical = 0.0, 0.0
                labels.append(np.asarray((visible, horizontal, vertical), dtype=np.float32))
    targets = np.asarray(labels, dtype=np.float32)
    tensor_dataset = TensorDataset(
        library.from_numpy(arrays["rgb"]),
        library.from_numpy(arrays["depth"]),
        library.from_numpy(arrays["lidar"]),
        library.from_numpy(arrays["radar"]),
        library.from_numpy(arrays["imu"]),
        library.from_numpy(arrays["token"]),
        library.from_numpy(arrays["state"]),
        library.from_numpy(targets),
    )
    model = TinyVisionLanguageGrounderNet()
    def grounding_loss(prediction: Any, target: Any) -> Any:
        visible = library.nn.functional.binary_cross_entropy_with_logits(prediction[:, 0], target[:, 0])
        direction = library.nn.functional.mse_loss(library.tanh(prediction[:, 1:]), target[:, 1:])
        return visible + direction * 0.35
    losses = _train_model(
        model,
        tensor_dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        loss_fn=grounding_loss,
    )
    metadata = _common_metadata(
        model_kind="tiny_vlm_grounder",
        information_profile=VLA_PROFILE,
        episodes=episodes,
        sample_count=len(tensor_dataset),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    metadata.update({"label_source": "learning_labels.semantic_segmentation", "final_training_loss": losses[-1]})
    checkpoint, metadata_path = _save_checkpoint(output, model=model, metadata=metadata)
    return TrainResult(checkpoint, metadata_path, "tiny_vlm_grounder", len(tensor_dataset), losses[-1])


def train_world_model(
    episode_paths: Sequence[Path],
    output: Path,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> TrainResult:
    library = _require_torch()
    episodes = load_pilot_episodes(episode_paths, required_profile=WORLD_PROFILE)
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    deltas: list[np.ndarray] = []
    for episode in episodes:
        for frame_index in range(episode.frame_count - 1):
            for agent_id in range(episode.agent_count):
                states.append(episode.states[frame_index, agent_id])
                actions.append(episode.actions[frame_index + 1, agent_id])
                contexts.append(
                    world_sensor_context(
                        episode.rgb[frame_index, agent_id],
                        episode.depth[frame_index, agent_id],
                        episode.lidar[frame_index, agent_id],
                        _radar_at(episode, frame_index, agent_id),
                        episode.imu[frame_index, agent_id],
                    )
                )
                deltas.append(episode.states[frame_index + 1, agent_id] - episode.states[frame_index, agent_id])
    state_values = np.asarray(states, dtype=np.float32)
    action_values = np.asarray(actions, dtype=np.float32)
    context_values = np.asarray(contexts, dtype=np.float32)
    delta_values = np.asarray(deltas, dtype=np.float32)
    if not len(state_values):
        raise ValueError("world-model training needs at least two frames per source episode")
    input_mean, input_std = _normalizer(state_values)
    output_mean, output_std = _normalizer(delta_values)
    state_normalized = (state_values - input_mean) / input_std
    target_normalized = (delta_values - output_mean) / output_std
    tensor_dataset = TensorDataset(
        library.from_numpy(state_normalized),
        library.from_numpy(action_values),
        library.from_numpy(context_values),
        library.from_numpy(target_normalized),
    )
    model = TinyActionDynamicsNet()
    losses = _train_model(
        model,
        tensor_dataset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        loss_fn=library.nn.MSELoss(),
    )
    metadata = _common_metadata(
        model_kind="tiny_action_world_model",
        information_profile=WORLD_PROFILE,
        episodes=episodes,
        sample_count=len(tensor_dataset),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    metadata.update(
        {
            "input_mean": input_mean.tolist(),
            "input_std": input_std.tolist(),
            "output_mean": output_mean.tolist(),
            "output_std": output_std.tolist(),
            "sensor_context_dim": int(context_values.shape[1]),
            "action_units": "world_velocity_mps_and_yaw_rate_rad_s",
            "target": "next_public_proprioception_delta",
            "final_training_loss": losses[-1],
        }
    )
    checkpoint, metadata_path = _save_checkpoint(output, model=model, metadata=metadata)
    return TrainResult(checkpoint, metadata_path, "tiny_action_world_model", len(tensor_dataset), losses[-1])


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("vla", "vlm", "world-model"), required=True)
    parser.add_argument("--episode-manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--chunk-size", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0.0:
        raise SystemExit("--epochs, --batch-size, and --learning-rate must be positive")
    library = _require_torch()
    library.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    trainer = {
        "vla": lambda: train_vla(
            args.episode_manifest,
            args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            chunk_size=args.chunk_size,
        ),
        "vlm": lambda: train_vlm(
            args.episode_manifest,
            args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        ),
        "world-model": lambda: train_world_model(
            args.episode_manifest,
            args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
        ),
    }[args.model]
    result = trainer()
    print(json.dumps({"status": "completed", **result.as_dict()}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
