from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.methods import NATIVE_DESCRIPTORS, create_native_policy
from rivermark_benchmark.recording import EpisodeRecorder
from rivermark_benchmark.runtime import PilotRuntimeConfig, PilotSwarmRuntime
from rivermark_benchmark.validate import validate_episode_manifest


class PilotRuntimeTests(unittest.TestCase):
    def test_multisensor_observation_has_no_target_truth(self) -> None:
        runtime = PilotSwarmRuntime(
            PilotRuntimeConfig(agent_count=3, max_steps=4),
            information_profile="multisensor_rgbd_lidar_radar_state",
        )
        observations = runtime.reset()
        observation = observations[0]
        self.assertEqual(observation.rgb.shape, (72, 96, 3))
        self.assertEqual(observation.distance_to_image_plane_m.shape, (72, 96))
        self.assertEqual(observation.lidar_ranges_m.shape, (72,))
        self.assertEqual(observation.imu.shape, (6,))
        self.assertNotIn("target", observation.__dict__)
        self.assertNotIn("seed", observation.__dict__)
        self.assertNotIn("evaluator", observation.__dict__)

    def test_native_policies_emit_complete_actions(self) -> None:
        for method_id, descriptor in NATIVE_DESCRIPTORS.items():
            with self.subTest(method_id=method_id):
                runtime = PilotSwarmRuntime(
                    PilotRuntimeConfig(agent_count=3, max_steps=2),
                    information_profile=descriptor.information_profile,
                )
                observations = runtime.reset()
                policy = create_native_policy(method_id)
                policy.reset(
                    runtime.mission,
                    runtime.config.agent_count,
                    public_geometry=runtime.public_geometry if descriptor.information_profile == "geometry_state" else None,
                )
                actions = policy.act(observations)
                self.assertEqual(set(actions), {0, 1, 2})
                _, frame = runtime.step(actions)
                self.assertEqual(frame.sensor_packets[0].rgb.shape, (72, 96, 3))

    def test_recorder_writes_hash_bound_valid_episode(self) -> None:
        method_id = "action_conditioned_world_model_mpc_pilot"
        descriptor = NATIVE_DESCRIPTORS[method_id]
        runtime = PilotSwarmRuntime(
            PilotRuntimeConfig(agent_count=3, max_steps=3),
            information_profile=descriptor.information_profile,
        )
        observations = runtime.reset()
        policy = create_native_policy(method_id)
        policy.reset(runtime.mission, 3)
        with tempfile.TemporaryDirectory() as temporary:
            recorder = EpisodeRecorder(
                Path(temporary),
                runtime=runtime,
                descriptor=descriptor,
                policy=policy,
                code_revision="0123456789abcdef",
                episode_id="pilot-world-model-test",
            )
            recorder.record(runtime.current_frame(), observations)
            while not runtime.done:
                observations, frame = runtime.step(policy.act(observations))
                recorder.record(frame, observations)
            result = recorder.finalize(runtime.evaluate())
            self.assertEqual(result.issues, ())
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(validate_episode_manifest(manifest, base_dir=result.episode_root, check_files=True), ())
            self.assertFalse(manifest["evaluator_private"]["distributed"])


if __name__ == "__main__":
    unittest.main()
