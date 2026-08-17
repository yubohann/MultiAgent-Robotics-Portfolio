from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import ribs  # noqa: F401
except ImportError:
    ribs = None

from rivermark_benchmark.qd_train import PyribsMapElitesCheckpointPolicy, train_map_elites
from rivermark_benchmark.runtime import PilotRuntimeConfig, PilotSwarmRuntime


@unittest.skipIf(ribs is None, "pyribs is not installed")
class PyribsMapElitesTests(unittest.TestCase):
    def test_archive_trains_and_runs_public_state_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = train_map_elites(
                Path(temporary) / "archive.npz",
                iterations=2,
                batch_size=3,
                rollout_steps=3,
                seed=81,
            )
            self.assertTrue(result.archive_path.is_file())
            self.assertGreater(result.elite_count, 0)
            policy = PyribsMapElitesCheckpointPolicy(result.archive_path)
            runtime = PilotSwarmRuntime(
                PilotRuntimeConfig(agent_count=2, max_steps=2, seed=82),
                information_profile="state_only",
            )
            observations = runtime.reset()
            policy.reset(runtime.mission, 2)
            actions = policy.act(observations)
            self.assertEqual(set(actions), {0, 1})
            runtime.step(actions)

    def test_archive_hash_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            result = train_map_elites(
                Path(temporary) / "archive.npz",
                iterations=1,
                batch_size=2,
                rollout_steps=3,
                seed=83,
            )
            with result.archive_path.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                PyribsMapElitesCheckpointPolicy(result.archive_path)


if __name__ == "__main__":
    unittest.main()
