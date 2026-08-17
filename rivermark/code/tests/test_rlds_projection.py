from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.rlds_projection import (
    RldsProjectionError,
    iter_rlds_records,
    project_state_action_to_rlds,
    verify_rlds_interchange,
)


def _source() -> dict[str, np.ndarray]:
    steps, agents = 3, 2
    return {
        "command_time_ns": np.asarray([0, 10, 20], dtype="<i8"),
        "effective_time_ns": np.asarray([5, 15, 25], dtype="<i8"),
        "root_pos_w_m": np.arange(steps * agents * 3, dtype="<f4").reshape(steps, agents, 3),
        "root_quat_wxyz": np.tile(np.asarray([1, 0, 0, 0], dtype="<f4"), (steps, agents, 1)),
        "root_lin_vel_w_mps": np.zeros((steps, agents, 3), dtype="<f4"),
        "root_ang_vel_b_radps": np.zeros((steps, agents, 3), dtype="<f4"),
        "applied_thrust_n": np.ones((steps, agents, 4), dtype="<f4"),
        "desired_pos_w_m": np.ones((steps, agents, 3), dtype="<f4"),
        "desired_vel_w_mps": np.zeros((steps, agents, 3), dtype="<f4"),
        "target_thrust_n": np.ones((steps, agents, 4), dtype="<f4"),
    }


class RldsProjectionTests(unittest.TestCase):
    def test_explicit_mapping_preserves_terminal_and_truncation_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "rlds"
            result = project_state_action_to_rlds(
                root,
                episode_id="fixture-0001",
                source_values=_source(),
                rewards=[1.0, 2.0],
                discounts=[0.99, 0.5],
                source_provenance={
                    "source_capture_receipt_sha256": "a" * 64,
                    "source_revision": "b" * 40,
                    "collection_protocol_id": "protocol-v1",
                    "split": "train",
                    "episode_index": 0,
                    "episode_seed": 7,
                },
                truncated=True,
                termination_reason="fixed_horizon",
                allow_initial_command_drop=True,
            )
            self.assertEqual(result.step_count, 3)
            self.assertEqual(result.dropped_initial_command_count, 1)
            report = verify_rlds_interchange(root)
            self.assertEqual(report["status"], "valid")
            records = list(iter_rlds_records(root / "episode.jsonl"))
            steps = records[1:-1]
            self.assertEqual([step["source_action_index"] for step in steps], [1, 2, 2])
            self.assertEqual([step["reward_valid"] for step in steps], [True, True, False])
            self.assertEqual(steps[-1]["is_last"], True)
            self.assertEqual(steps[-1]["is_terminal"], False)
            self.assertEqual(steps[-1]["truncated"], True)
            manifest = json.loads((root / "projection_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["claim_boundary"],
                "development-only RLDS-shaped interchange; no TFDS or formal-episode claim",
            )

    def test_terminal_and_truncated_are_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RldsProjectionError, "cannot both"):
                project_state_action_to_rlds(
                    Path(temporary) / "rlds",
                    episode_id="fixture-0001",
                    source_values=_source(),
                    rewards=[1.0, 2.0],
                    source_provenance={"source_revision": "b" * 40},
                    terminal=True,
                    truncated=True,
                    allow_initial_command_drop=True,
                )
            with self.assertRaisesRegex(RldsProjectionError, "one of terminal or truncated"):
                project_state_action_to_rlds(
                    Path(temporary) / "rlds-no-end",
                    episode_id="fixture-0001",
                    source_values=_source(),
                    rewards=[1.0, 2.0],
                    source_provenance={"source_revision": "b" * 40},
                    terminal=False,
                    truncated=False,
                    allow_initial_command_drop=True,
                )

    def test_missing_reward_and_implicit_drop_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RldsProjectionError, "allow_initial_command_drop"):
                project_state_action_to_rlds(
                    Path(temporary) / "rlds",
                    episode_id="fixture-0001",
                    source_values=_source(),
                    rewards=[1.0, 2.0],
                    source_provenance={"source_revision": "b" * 40},
                )
            with self.assertRaisesRegex(RldsProjectionError, "rewards must be finite"):
                project_state_action_to_rlds(
                    Path(temporary) / "rlds",
                    episode_id="fixture-0001",
                    source_values=_source(),
                    rewards=[1.0],
                    source_provenance={"source_revision": "b" * 40},
                    allow_initial_command_drop=True,
                )

    def test_prefix_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "rlds"
            project_state_action_to_rlds(
                root,
                episode_id="fixture-0001",
                source_values=_source(),
                rewards=[1.0, 2.0],
                source_provenance={"source_revision": "b" * 40},
                allow_initial_command_drop=True,
            )
            path = root / "episode.jsonl"
            lines = path.read_text(encoding="utf-8").splitlines()
            altered = json.loads(lines[1])
            altered["reward"] = 99.0
            lines[1] = json.dumps(altered, sort_keys=True, separators=(",", ":"))
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RldsProjectionError, "projection manifest does not bind|prefix hash mismatch"):
                verify_rlds_interchange(root)


if __name__ == "__main__":
    unittest.main()
