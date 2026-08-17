from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.dataset import collect_episode, load_pilot_episode


class CpuSmokeDatasetTests(unittest.TestCase):
    def test_collected_pilot_episode_round_trips_through_public_loader(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rivermark-cpu-smoke-") as temporary:
            manifest_path = collect_episode(
                Path(temporary),
                episode_id="cpu-smoke-0001",
                teacher_method="action_chunk_vla_pilot",
                agent_count=2,
                max_steps=4,
                seed=7,
            )
            episode = load_pilot_episode(manifest_path)
            self.assertEqual(episode.frame_count, 5)
            self.assertEqual(episode.agent_count, 2)
            self.assertEqual(episode.rgb.shape, (5, 2, 72, 96, 3))
            self.assertEqual(episode.states.shape, (5, 2, 8))
            self.assertEqual(episode.actions.shape, (5, 2, 4))
            self.assertTrue(all(source for source in episode.action_sources))
            self.assertEqual(episode.manifest["dataset_version"], "0.1.0-pilot")
            self.assertEqual(episode.manifest["split"], "pilot")

    def test_loader_can_select_small_state_action_projection(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rivermark-cpu-select-") as temporary:
            manifest_path = collect_episode(
                Path(temporary), episode_id="select-0001", teacher_method="action_chunk_vla_pilot",
                agent_count=2, max_steps=4, seed=9,
            )
            episode = load_pilot_episode(manifest_path, modalities=("state", "action"))
            self.assertIsNone(episode.rgb)
            self.assertIsNone(episode.depth)
            self.assertEqual(episode.states.shape, (5, 2, 8))
            self.assertEqual(episode.actions.shape, (5, 2, 4))
            self.assertEqual(episode.agent_ids, (0, 1))

    def test_loader_selects_aliases_and_preserves_original_agent_id(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rivermark-cpu-agent-") as temporary:
            manifest_path = collect_episode(
                Path(temporary), episode_id="agent-0001", teacher_method="action_chunk_vla_pilot",
                agent_count=3, max_steps=3, seed=10,
            )
            episode = load_pilot_episode(manifest_path, modalities=("rgb", "depth"), agent_ids=(2,))
            self.assertEqual(episode.rgb.shape, (4, 1, 72, 96, 3))
            self.assertEqual(episode.depth.shape, (4, 1, 72, 96))
            self.assertEqual(episode.agent_ids, (2,))
            self.assertIsNone(episode.states)
            self.assertIsNone(episode.actions)

    def test_loader_rejects_invalid_selection(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rivermark-cpu-invalid-") as temporary:
            manifest_path = collect_episode(
                Path(temporary), episode_id="invalid-0001", teacher_method="action_chunk_vla_pilot",
                agent_count=2, max_steps=3, seed=11,
            )
            with self.assertRaises(ValueError):
                load_pilot_episode(manifest_path, modalities=("rgb", "RGB"))
            with self.assertRaises(ValueError):
                load_pilot_episode(manifest_path, modalities=("unknown",))
            with self.assertRaises(ValueError):
                load_pilot_episode(manifest_path, agent_ids=(0, 0))
            with self.assertRaises(ValueError):
                load_pilot_episode(manifest_path, agent_ids=(2,))


if __name__ == "__main__":
    unittest.main()
