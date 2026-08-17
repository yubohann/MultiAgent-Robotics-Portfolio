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

try:
    import torch  # noqa: F401
except ImportError:
    torch = None

try:
    import gymnasium  # noqa: F401
    import pettingzoo  # noqa: F401
except ImportError:
    gymnasium = None
    pettingzoo = None

from rivermark_benchmark.dataset import collect_episode, load_pilot_episode, sha256_file
from rivermark_benchmark.learned import (
    LearnedWorldModelMpcCheckpointPolicy,
    TinyVlaCheckpointPolicy,
    TinyVlmGroundingCheckpointPolicy,
)
from rivermark_benchmark.runtime import PilotRuntimeConfig, PilotSwarmRuntime
from rivermark_benchmark.torch_train import train_vla, train_vlm, train_world_model
from rivermark_benchmark.marl import SharedMarlCheckpointPolicy, train_shared_marl


MARL_TRAINING_AVAILABLE = torch is not None and gymnasium is not None and pettingzoo is not None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class LearnedPipelineTests(unittest.TestCase):
    def _episode(self, root: Path, *, teacher: str, seed: int) -> Path:
        return collect_episode(
            root,
            episode_id=f"{teacher}-test",
            teacher_method=teacher,
            agent_count=2,
            max_steps=5,
            seed=seed,
        )

    def test_collect_load_and_train_all_public_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            language_manifest = self._episode(root, teacher="action_chunk_vla_pilot", seed=712)
            world_manifest = self._episode(
                root,
                teacher="action_conditioned_world_model_mpc_pilot",
                seed=713,
            )
            language_episode = load_pilot_episode(
                language_manifest,
                required_profile="language_multisensor_rgbd_lidar_radar_state",
            )
            world_episode = load_pilot_episode(
                world_manifest,
                required_profile="multisensor_rgbd_lidar_radar_state",
            )
            self.assertTrue(language_episode.language)
            self.assertEqual(world_episode.language, "")
            self.assertEqual(language_episode.sample_count, 12)
            self.assertEqual(world_episode.states.shape, (6, 2, 8))

            vla = train_vla(
                [language_manifest],
                root / "vla.pt",
                epochs=1,
                batch_size=4,
                learning_rate=0.001,
                seed=10,
                chunk_size=2,
            )
            vlm = train_vlm(
                [language_manifest],
                root / "vlm.pt",
                epochs=1,
                batch_size=4,
                learning_rate=0.001,
                seed=11,
            )
            world = train_world_model(
                [world_manifest],
                root / "world.pt",
                epochs=1,
                batch_size=4,
                learning_rate=0.001,
                seed=12,
            )
            self.assertGreater(vla.sample_count, 0)
            self.assertEqual(vlm.sample_count, language_episode.sample_count)
            self.assertEqual(world.sample_count, 10)
            for result in (vla, vlm, world):
                metadata = json.loads(result.metadata.read_text(encoding="utf-8"))
                self.assertEqual(metadata["checkpoint_sha256"], sha256_file(result.checkpoint))
                self.assertEqual(metadata["source_episodes"][0]["episode_manifest_sha256"], sha256_file(
                    language_manifest if result is not world else world_manifest
                ))

            policies = (
                (
                    TinyVlaCheckpointPolicy(vla.checkpoint),
                    "language_multisensor_rgbd_lidar_radar_state",
                ),
                (
                    TinyVlmGroundingCheckpointPolicy(vlm.checkpoint),
                    "language_multisensor_rgbd_lidar_radar_state",
                ),
                (
                    LearnedWorldModelMpcCheckpointPolicy(world.checkpoint),
                    "multisensor_rgbd_lidar_radar_state",
                ),
            )
            for policy, profile in policies:
                runtime = PilotSwarmRuntime(
                    PilotRuntimeConfig(agent_count=2, max_steps=2, seed=44),
                    information_profile=profile,
                )
                observations = runtime.reset()
                policy.reset(runtime.mission, 2)
                actions = policy.act(observations)
                self.assertEqual(set(actions), {0, 1})
                runtime.step(actions)

    def test_checkpoint_hash_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = self._episode(root, teacher="action_chunk_vla_pilot", seed=900)
            result = train_vla(
                [manifest],
                root / "tampered.pt",
                epochs=1,
                batch_size=4,
                learning_rate=0.001,
                seed=13,
                chunk_size=1,
            )
            with result.checkpoint.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                TinyVlaCheckpointPolicy(result.checkpoint)

    def test_source_manifest_hash_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = self._episode(root, teacher="action_chunk_vla_pilot", seed=901)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["streams"][0]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid source episode"):
                load_pilot_episode(manifest_path)

    @unittest.skipUnless(MARL_TRAINING_AVAILABLE, "MARL extra (Gymnasium, PettingZoo, and PyTorch) is not installed")
    def test_shared_marl_trains_from_public_messages_and_rolls_out(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = train_shared_marl(
                Path(temporary) / "shared-marl.pt",
                updates=2,
                agent_count=3,
                episode_steps=4,
                learning_rate=0.001,
                ppo_epochs=1,
                minibatch_size=8,
                seed=22,
            )
            metadata = json.loads(result.metadata.read_text(encoding="utf-8"))
            self.assertEqual(metadata["policy_parameter_sharing"], "one_shared_policy_for_all_agents")
            self.assertFalse(metadata["reward_uses_evaluator_private_truth"])
            policy = SharedMarlCheckpointPolicy(result.checkpoint)
            runtime = PilotSwarmRuntime(
                PilotRuntimeConfig(agent_count=3, max_steps=2, seed=23),
                information_profile="state_only",
            )
            observations = runtime.reset()
            policy.reset(runtime.mission, 3)
            actions = policy.act(observations)
            self.assertEqual(set(actions), {0, 1, 2})
            runtime.step(actions)
            with result.checkpoint.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                SharedMarlCheckpointPolicy(result.checkpoint)


if __name__ == "__main__":
    unittest.main()
