"""Execute native Rivermark pilots and render evidence-bound MP4 demos."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .methods import (
    NATIVE_DESCRIPTORS,
    MethodDescriptor,
    create_native_policy,
    create_sb3_checkpoint_policy,
    list_methods,
)
from .learned import (
    LearnedWorldModelMpcCheckpointPolicy,
    TinyVlaCheckpointPolicy,
    TinyVlmGroundingCheckpointPolicy,
)
from .marl import SharedMarlCheckpointPolicy
from .qd_train import PyribsMapElitesCheckpointPolicy
from .provenance import source_revision
from .recording import EpisodeRecorder, _write_json
from .runtime import PilotRuntimeConfig, PilotSwarmRuntime, RuntimeFrame


def _opencv() -> Any:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "MP4 rendering requires OpenCV. Use the IsaacLab conda Python or install the demo extra; no synthetic placeholder video will be written."
        ) from exc
    return cv2


def _source_revision() -> str:
    return source_revision()


class DemoRenderer:
    """Render a dashboard only from public runtime state and sensor packets."""

    def __init__(self, runtime: PilotSwarmRuntime, method_id: str) -> None:
        self.runtime = runtime
        self.method_id = method_id
        self._tracks: dict[int, list[np.ndarray]] = {agent: [] for agent in range(runtime.config.agent_count)}

    def render(self, frame: RuntimeFrame) -> np.ndarray:
        cv2 = _opencv()
        canvas = np.full((540, 960, 3), (21, 27, 31), dtype=np.uint8)
        map_width, map_height = 520, 500
        map_origin = (16, 24)
        cv2.rectangle(canvas, map_origin, (map_origin[0] + map_width, map_origin[1] + map_height), (48, 77, 60), -1)
        world_width, world_height = self.runtime.config.world_size_xy_m
        def world_to_pixel(point: np.ndarray) -> tuple[int, int]:
            return (
                map_origin[0] + int(np.clip(point[0] / world_width, 0.0, 1.0) * map_width),
                map_origin[1] + map_height - int(np.clip(point[1] / world_height, 0.0, 1.0) * map_height),
            )
        for obstacle in self.runtime.public_geometry["obstacles"]:
            center = np.asarray((*obstacle["center_xy_m"], 0.0))
            pixel = world_to_pixel(center)
            radius = max(2, int(obstacle["radius_m"] / world_width * map_width))
            cv2.circle(canvas, pixel, radius, (78, 78, 84), -1)
            cv2.circle(canvas, pixel, radius, (150, 150, 158), 1)
        palette = ((71, 158, 255), (254, 195, 70), (80, 211, 149), (236, 102, 151), (168, 132, 255), (95, 206, 220), (230, 141, 84), (235, 235, 235))
        for agent_id, state in frame.states.items():
            color = palette[agent_id % len(palette)]
            self._tracks[agent_id].append(state.position_m.copy())
            points = [world_to_pixel(point) for point in self._tracks[agent_id][-120:]]
            if len(points) > 1:
                cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, color, 1, cv2.LINE_AA)
            center = world_to_pixel(state.position_m)
            cv2.circle(canvas, center, 5, color, -1)
            heading = (center[0] + int(np.cos(state.yaw_rad) * 11), center[1] - int(np.sin(state.yaw_rad) * 11))
            cv2.line(canvas, center, heading, (240, 240, 240), 1, cv2.LINE_AA)
            cv2.putText(canvas, str(agent_id), (center[0] + 6, center[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        for event in frame.candidate_events:
            center = world_to_pixel(np.asarray(event.estimated_xyz_m))
            cv2.drawMarker(canvas, center, (40, 45, 245), cv2.MARKER_TILTED_CROSS, 14, 2, cv2.LINE_AA)
        cv2.putText(canvas, self.method_id, (16, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (241, 244, 246), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"t={frame.sim_time_ns / 1e9:05.1f}s  agents={len(frame.states)}  candidates={len(frame.candidate_events)}", (16, 538), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (199, 208, 214), 1, cv2.LINE_AA)

        packet = frame.sensor_packets[0]
        rgb = cv2.cvtColor(packet.rgb, cv2.COLOR_RGB2BGR)
        rgb = cv2.resize(rgb, (400, 300), interpolation=cv2.INTER_NEAREST)
        canvas[38:338, 544:944] = rgb
        depth = packet.distance_to_image_plane_m
        scaled = np.clip(depth / self.runtime.config.lidar_max_range_m * 255.0, 0.0, 255.0).astype(np.uint8)
        depth_bgr = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
        depth_bgr = cv2.resize(depth_bgr, (400, 160), interpolation=cv2.INTER_NEAREST)
        canvas[362:522, 544:944] = depth_bgr
        cv2.putText(canvas, "agent 0 online RGB", (544, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (241, 244, 246), 1, cv2.LINE_AA)
        cv2.putText(canvas, "agent 0 online distance-to-image-plane", (544, 354), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (241, 244, 246), 1, cv2.LINE_AA)
        return canvas


class Mp4Writer:
    """Write browser-compatible H.264 MP4 through the bundled ffmpeg binary."""

    def __init__(self, path: Path, *, fps: int, frame_size: tuple[int, int]) -> None:
        self.path = path
        self.frame_size = frame_size
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import imageio_ffmpeg  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "H.264 MP4 rendering requires imageio-ffmpeg. Install the demo extra in the IsaacLab environment."
            ) from exc
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        width, height = frame_size
        self._process = subprocess.Popen(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "bgr24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-y",
                str(path),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._closed = False

    def write(self, frame: np.ndarray) -> None:
        width, height = self.frame_size
        if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
            raise ValueError(f"unexpected demo frame shape: {frame.shape}")
        if self._process.stdin is None:
            raise RuntimeError("ffmpeg input pipe is unavailable")
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            stderr = ""
            if self._process.stderr is not None:
                stderr = self._process.stderr.read().decode("utf-8", errors="replace")
                self._process.stderr.close()
            raise RuntimeError(f"ffmpeg stopped while encoding {self.path}: {stderr}") from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.stdin is not None:
            self._process.stdin.close()
        stderr = ""
        if self._process.stderr is not None:
            stderr = self._process.stderr.read().decode("utf-8", errors="replace")
            self._process.stderr.close()
        return_code = self._process.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg failed while encoding {self.path}: {stderr}")
        if not self.path.is_file() or self.path.stat().st_size < 1024:
            raise RuntimeError(f"MP4 writer produced no usable video: {self.path}")


@dataclass(frozen=True)
class DemoResult:
    method_id: str
    episode_root: str
    video_path: str
    receipt_path: str
    metrics: dict[str, Any]
    validation_issue_count: int


def run_native_demo(
    method_id: str,
    output_root: Path,
    *,
    agent_count: int,
    max_steps: int,
    seed: int,
    fps: int,
    overwrite: bool = False,
) -> DemoResult:
    descriptor = NATIVE_DESCRIPTORS.get(method_id)
    if descriptor is None:
        raise KeyError(f"not a runnable native method: {method_id}")
    episode_root = output_root / method_id
    if episode_root.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing demo: {episode_root}; pass --overwrite")
    runtime = PilotSwarmRuntime(
        PilotRuntimeConfig(agent_count=agent_count, max_steps=max_steps, seed=seed),
        information_profile=descriptor.information_profile,
    )
    policy = create_native_policy(method_id)
    observations = runtime.reset()
    policy.reset(
        runtime.mission,
        runtime.config.agent_count,
        public_geometry=runtime.public_geometry if descriptor.information_profile == "geometry_state" else None,
    )
    recorder = EpisodeRecorder(
        output_root,
        runtime=runtime,
        descriptor=descriptor,
        policy=policy,
        code_revision=_source_revision(),
        episode_id=method_id,
    )
    renderer = DemoRenderer(runtime, method_id)
    video_path = episode_root / "demo.mp4"
    writer = Mp4Writer(video_path, fps=fps, frame_size=(960, 540))
    try:
        initial_frame = runtime.current_frame()
        recorder.record(initial_frame, observations)
        writer.write(renderer.render(initial_frame))
        while not runtime.done:
            actions = policy.act(observations)
            observations, frame = runtime.step(actions)
            recorder.record(frame, observations)
            writer.write(renderer.render(frame))
    finally:
        writer.close()
    evaluation = runtime.evaluate()
    recording = recorder.finalize(evaluation, video_path=video_path)
    if recording.issues:
        raise RuntimeError("generated recording did not pass manifest validation: " + "; ".join(issue.code for issue in recording.issues))
    return DemoResult(
        method_id=method_id,
        episode_root=str(recording.episode_root),
        video_path=str(video_path),
        receipt_path=str(recording.receipt_path),
        metrics=evaluation.as_dict(),
        validation_issue_count=len(recording.issues),
    )


def run_sb3_checkpoint_demo(
    checkpoint: Path,
    output_root: Path,
    *,
    metadata_path: Path | None,
    agent_count: int,
    max_steps: int,
    seed: int,
    fps: int,
    overwrite: bool = False,
) -> DemoResult:
    """Run a real SB3 checkpoint and bind its weights into the receipt."""

    policy = create_sb3_checkpoint_policy(checkpoint, metadata_path)
    method_id = f"sb3_{policy.metadata['algorithm']}_checkpoint"
    descriptor = MethodDescriptor(
        method_id=method_id,
        family="rl",
        information_profile="state_only",
        implementation_kind=str(policy.metadata.get("implementation_kind", "external_checkpoint_adapter")),
        description="Stable-Baselines3 checkpoint executed through the provenance-checked Rivermark state-only adapter.",
        requires=("stable_baselines3",),
        checkpoint_required=True,
    )
    episode_root = output_root / method_id
    if episode_root.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing demo: {episode_root}; pass --overwrite")
    runtime = PilotSwarmRuntime(
        PilotRuntimeConfig(agent_count=agent_count, max_steps=max_steps, seed=seed),
        information_profile="state_only",
    )
    observations = runtime.reset()
    policy.reset(runtime.mission, runtime.config.agent_count)
    recorder = EpisodeRecorder(
        output_root,
        runtime=runtime,
        descriptor=descriptor,
        policy=policy,
        code_revision=_source_revision(),
        episode_id=method_id,
    )
    renderer = DemoRenderer(runtime, method_id)
    video_path = episode_root / "demo.mp4"
    writer = Mp4Writer(video_path, fps=fps, frame_size=(960, 540))
    try:
        initial_frame = runtime.current_frame()
        recorder.record(initial_frame, observations)
        writer.write(renderer.render(initial_frame))
        while not runtime.done:
            observations, frame = runtime.step(policy.act(observations))
            recorder.record(frame, observations)
            writer.write(renderer.render(frame))
    finally:
        writer.close()
    evaluation = runtime.evaluate()
    recording = recorder.finalize(evaluation, video_path=video_path)
    if recording.issues:
        raise RuntimeError("generated recording did not pass manifest validation: " + "; ".join(issue.code for issue in recording.issues))
    return DemoResult(
        method_id=method_id,
        episode_root=str(recording.episode_root),
        video_path=str(video_path),
        receipt_path=str(recording.receipt_path),
        metrics=evaluation.as_dict(),
        validation_issue_count=len(recording.issues),
    )


def run_checkpoint_demo(
    policy: Any,
    descriptor: MethodDescriptor,
    output_root: Path,
    *,
    agent_count: int,
    max_steps: int,
    seed: int,
    fps: int,
    overwrite: bool = False,
) -> DemoResult:
    """Run any provenance-checked local checkpoint through the normal recorder."""

    method_id = descriptor.method_id
    episode_root = output_root / method_id
    if episode_root.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing demo: {episode_root}; pass --overwrite")
    runtime = PilotSwarmRuntime(
        PilotRuntimeConfig(agent_count=agent_count, max_steps=max_steps, seed=seed),
        information_profile=descriptor.information_profile,
    )
    observations = runtime.reset()
    policy.reset(runtime.mission, runtime.config.agent_count)
    recorder = EpisodeRecorder(
        output_root,
        runtime=runtime,
        descriptor=descriptor,
        policy=policy,
        code_revision=_source_revision(),
        episode_id=method_id,
    )
    renderer = DemoRenderer(runtime, method_id)
    video_path = episode_root / "demo.mp4"
    writer = Mp4Writer(video_path, fps=fps, frame_size=(960, 540))
    try:
        initial_frame = runtime.current_frame()
        recorder.record(initial_frame, observations)
        writer.write(renderer.render(initial_frame))
        while not runtime.done:
            observations, frame = runtime.step(policy.act(observations))
            recorder.record(frame, observations)
            writer.write(renderer.render(frame))
    finally:
        writer.close()
    evaluation = runtime.evaluate()
    recording = recorder.finalize(evaluation, video_path=video_path)
    if recording.issues:
        raise RuntimeError("generated recording did not pass manifest validation: " + "; ".join(issue.code for issue in recording.issues))
    return DemoResult(
        method_id=method_id,
        episode_root=str(recording.episode_root),
        video_path=str(video_path),
        receipt_path=str(recording.receipt_path),
        metrics=evaluation.as_dict(),
        validation_issue_count=len(recording.issues),
    )


def compose_videos(results: Sequence[DemoResult], output_path: Path, *, fps: int) -> None:
    """Concatenate generated method videos without inventing any frames."""

    cv2 = _opencv()
    writer = Mp4Writer(output_path, fps=fps, frame_size=(960, 540))
    try:
        for result in results:
            capture = cv2.VideoCapture(result.video_path)
            if not capture.isOpened():
                raise RuntimeError(f"cannot read generated method video: {result.video_path}")
            frame_count = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_count += 1
                writer.write(frame)
            capture.release()
            if frame_count == 0:
                raise RuntimeError(f"generated method video has no readable frames: {result.video_path}")
    finally:
        writer.close()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list registered methods")
    parser.add_argument("--all", action="store_true", help="run every native pilot method")
    parser.add_argument("--method", action="append", choices=sorted(NATIVE_DESCRIPTORS), help="run one native method; repeatable")
    parser.add_argument("--sb3-checkpoint", type=Path, help="run a real SB3 checkpoint through the state-only adapter")
    parser.add_argument("--sb3-metadata", type=Path, help="adapter metadata; defaults beside --sb3-checkpoint")
    parser.add_argument("--torch-vla-checkpoint", type=Path, help="run a provenance-checked trained tiny VLA checkpoint")
    parser.add_argument("--torch-vla-metadata", type=Path, help="metadata beside --torch-vla-checkpoint by default")
    parser.add_argument("--torch-vlm-checkpoint", type=Path, help="run a provenance-checked trained tiny VLM checkpoint")
    parser.add_argument("--torch-vlm-metadata", type=Path, help="metadata beside --torch-vlm-checkpoint by default")
    parser.add_argument("--torch-world-model-checkpoint", type=Path, help="run a provenance-checked trained tiny world-model checkpoint")
    parser.add_argument("--torch-world-model-metadata", type=Path, help="metadata beside --torch-world-model-checkpoint by default")
    parser.add_argument("--pyribs-archive", type=Path, help="run a provenance-checked pyribs MAP-Elites archive")
    parser.add_argument("--pyribs-metadata", type=Path, help="metadata beside --pyribs-archive by default")
    parser.add_argument("--marl-checkpoint", type=Path, help="run a provenance-checked shared decentralized MARL checkpoint")
    parser.add_argument("--marl-metadata", type=Path, help="metadata beside --marl-checkpoint by default")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts") / "demos")
    parser.add_argument("--agents", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=72)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backend", choices=("pilot", "isaaclab"), default="pilot")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.list:
        for descriptor in list_methods():
            print(f"{descriptor.method_id:42} {descriptor.family:18} {descriptor.implementation_kind}")
        return 0
    if args.backend != "pilot":
        raise SystemExit(
            "The IsaacLab backend is fail-closed until a local Kit/scene/radar smoke receipt exists. Use --backend pilot for the executable kinematic pilot."
        )
    selected = sorted(NATIVE_DESCRIPTORS) if args.all else args.method or []
    checkpoint_options = (
        args.sb3_checkpoint,
        args.torch_vla_checkpoint,
        args.torch_vlm_checkpoint,
        args.torch_world_model_checkpoint,
        args.pyribs_archive,
        args.marl_checkpoint,
    )
    checkpoint_count = sum(option is not None for option in checkpoint_options)
    if checkpoint_count and (args.all or selected):
        raise SystemExit("checkpoint/archive options cannot be combined with --method or --all")
    if checkpoint_count > 1:
        raise SystemExit("choose exactly one checkpoint/archive option")
    if not selected and checkpoint_count == 0:
        raise SystemExit("choose --method METHOD, --all, or one checkpoint/archive option")
    if args.agents < 1 or args.max_steps < 1 or args.fps < 1:
        raise SystemExit("--agents, --max-steps, and --fps must be positive")
    if args.sb3_checkpoint is not None:
        results = [
            run_sb3_checkpoint_demo(
                args.sb3_checkpoint,
                args.output_root,
                metadata_path=args.sb3_metadata,
                agent_count=args.agents,
                max_steps=args.max_steps,
                seed=args.seed,
                fps=args.fps,
                overwrite=args.overwrite,
            )
        ]
    elif args.torch_vla_checkpoint is not None:
        results = [
            run_checkpoint_demo(
                TinyVlaCheckpointPolicy(args.torch_vla_checkpoint, args.torch_vla_metadata),
                MethodDescriptor(
                    "tiny_vla_checkpoint",
                    "vla",
                    "language_multisensor_rgbd_lidar_radar_state",
                    "trained_torch_pilot_checkpoint",
                    "Trained compact multimodal VLA checkpoint executed through the pilot adapter.",
                    ("torch",),
                    True,
                ),
                args.output_root,
                agent_count=args.agents,
                max_steps=args.max_steps,
                seed=args.seed,
                fps=args.fps,
                overwrite=args.overwrite,
            )
        ]
    elif args.torch_vlm_checkpoint is not None:
        results = [
            run_checkpoint_demo(
                TinyVlmGroundingCheckpointPolicy(args.torch_vlm_checkpoint, args.torch_vlm_metadata),
                MethodDescriptor(
                    "tiny_vlm_grounding_checkpoint",
                    "vlm",
                    "language_multisensor_rgbd_lidar_radar_state",
                    "trained_torch_pilot_checkpoint",
                    "Trained compact multimodal VLM grounding checkpoint executed through the pilot adapter.",
                    ("torch",),
                    True,
                ),
                args.output_root,
                agent_count=args.agents,
                max_steps=args.max_steps,
                seed=args.seed,
                fps=args.fps,
                overwrite=args.overwrite,
            )
        ]
    elif args.torch_world_model_checkpoint is not None:
        results = [
            run_checkpoint_demo(
                LearnedWorldModelMpcCheckpointPolicy(
                    args.torch_world_model_checkpoint,
                    args.torch_world_model_metadata,
                ),
                MethodDescriptor(
                    "learned_world_model_mpc_checkpoint",
                    "world_model",
                    "multisensor_rgbd_lidar_radar_state",
                    "trained_torch_pilot_checkpoint",
                    "Trained action-conditioned public-sensor world-model checkpoint executed through MPC.",
                    ("torch",),
                    True,
                ),
                args.output_root,
                agent_count=args.agents,
                max_steps=args.max_steps,
                seed=args.seed,
                fps=args.fps,
                overwrite=args.overwrite,
            )
        ]
    elif args.pyribs_archive is not None:
        results = [
            run_checkpoint_demo(
                PyribsMapElitesCheckpointPolicy(args.pyribs_archive, args.pyribs_metadata),
                MethodDescriptor(
                    "pyribs_map_elites_checkpoint",
                    "quality_diversity",
                    "state_only",
                    "trained_pyribs_map_elites_archive",
                    "Real pyribs MAP-Elites archive selected from public route descriptors.",
                    ("ribs",),
                    True,
                ),
                args.output_root,
                agent_count=args.agents,
                max_steps=args.max_steps,
                seed=args.seed,
                fps=args.fps,
                overwrite=args.overwrite,
            )
        ]
    elif args.marl_checkpoint is not None:
        results = [
            run_checkpoint_demo(
                SharedMarlCheckpointPolicy(args.marl_checkpoint, args.marl_metadata),
                MethodDescriptor(
                    "shared_marl_actor_critic_checkpoint",
                    "marl",
                    "state_only",
                    "trained_torch_marl_pilot_checkpoint",
                    "Shared-parameter decentralized actor-critic trained through the PettingZoo public-observation pilot environment.",
                    ("torch", "pettingzoo"),
                    True,
                ),
                args.output_root,
                agent_count=args.agents,
                max_steps=args.max_steps,
                seed=args.seed,
                fps=args.fps,
                overwrite=args.overwrite,
            )
        ]
    else:
        results = [
            run_native_demo(
                method_id,
                args.output_root,
                agent_count=args.agents,
                max_steps=args.max_steps,
                seed=args.seed,
                fps=args.fps,
                overwrite=args.overwrite,
            )
            for method_id in selected
        ]
    composed = None
    if len(results) > 1:
        composed_path = args.output_root / "demo.mp4"
        compose_videos(results, composed_path, fps=args.fps)
        composed = str(composed_path)
    _write_json(
        args.output_root / "manifest.json",
        {
            "schema": "org.rivermark.benchmark.demo-suite.v1",
            "backend": "rivermark-kinematic-pilot-v1",
            "formal_benchmark_admission": False,
            "results": [asdict(result) for result in results],
            "composed_video": composed,
        },
    )
    print(json.dumps({"status": "completed", "methods": [result.method_id for result in results], "composed_video": composed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
