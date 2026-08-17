from __future__ import annotations

import hashlib
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
    import gymnasium  # noqa: F401
    import stable_baselines3  # noqa: F401
except ImportError:
    stable_baselines3 = None

from rivermark_benchmark.methods import create_sb3_checkpoint_policy
from rivermark_benchmark.train import _SingleAgentStateEnv


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@unittest.skipIf(stable_baselines3 is None, "Stable-Baselines3 is not installed")
class StableBaselines3AdapterTests(unittest.TestCase):
    def _checkpoint_and_metadata(self, root: Path) -> tuple[Path, Path]:
        from stable_baselines3 import PPO

        environment = _SingleAgentStateEnv(seed=71, agent_count=2, max_steps=3)
        model = PPO("MlpPolicy", environment, n_steps=8, batch_size=8, seed=71, verbose=0, device="cpu")
        checkpoint = root / "tiny_ppo.zip"
        model.save(str(checkpoint.with_suffix("")))
        metadata = checkpoint.with_suffix(".rivermark.json")
        metadata.write_text(
            json.dumps(
                {
                    "schema": "org.rivermark.sb3-adapter.v1",
                    "implementation_kind": "trained_sb3_pilot_checkpoint",
                    "algorithm": "ppo",
                    "information_profile": "state_only",
                    "observation_mean": [0.0] * 8,
                    "observation_std": [1.0] * 8,
                    "action_scale": [2.3, 2.3, 1.25, 1.4],
                    "checkpoint_sha256": _sha256(checkpoint),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return checkpoint, metadata

    def test_missing_or_mismatched_checkpoint_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint, metadata_path = self._checkpoint_and_metadata(Path(temporary))
            policy = create_sb3_checkpoint_policy(checkpoint, metadata_path)
            self.assertEqual(policy.provenance()["checkpoint_sha256"], _sha256(checkpoint))

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata.pop("checkpoint_sha256")
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                create_sb3_checkpoint_policy(checkpoint, metadata_path)

            metadata["checkpoint_sha256"] = _sha256(checkpoint)
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with checkpoint.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                create_sb3_checkpoint_policy(checkpoint, metadata_path)


if __name__ == "__main__":
    unittest.main()
