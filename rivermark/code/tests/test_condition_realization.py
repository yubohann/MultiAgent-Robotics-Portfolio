from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.condition_realization import (  # noqa: E402
    SUPPORTED_CONDITION_VALUES,
    condition_request_from_protocol,
    evaluate_condition_realization,
    validate_condition_request,
)


def _protocol() -> dict[str, object]:
    conditions = dict(SUPPORTED_CONDITION_VALUES)
    return {
        "cells": [
            {
                "cell_id": "train-baseline-0",
                "split": "train",
                "conditions": conditions,
            }
        ]
    }


class ConditionRealizationTests(unittest.TestCase):
    def test_request_is_public_and_binding_scoped(self) -> None:
        request = condition_request_from_protocol(
            _protocol(),
            protocol_id="citylite-coverage-v1",
            protocol_sha256="a" * 64,
            cell_id="train-baseline-0",
        )
        self.assertEqual(validate_condition_request(request), ())
        self.assertEqual(request["status"], "pending_independent_check")
        self.assertEqual(set(request["conditions"]), set(SUPPORTED_CONDITION_VALUES))
        self.assertNotIn("lighting", request["axis_support"])
        inconsistent = dict(request)
        inconsistent["axis_support"] = dict(request["axis_support"], layout="unavailable")
        self.assertIn(
            "condition_request_support",
            {issue["code"] for issue in validate_condition_request(inconsistent)},
        )
        binding = {
            "protocol_id": "citylite-coverage-v1",
            "protocol_sha256": "a" * 64,
            "cell_id": "train-baseline-0",
        }
        self.assertEqual(validate_condition_request(request, binding=binding), ())
        binding["cell_id"] = "validation-baseline-0"
        self.assertTrue(validate_condition_request(request, binding=binding))

    def test_raw_evidence_verifies_declared_supported_axes(self) -> None:
        request = condition_request_from_protocol(
            _protocol(),
            protocol_id="citylite-coverage-v1",
            protocol_sha256="a" * 64,
            cell_id="train-baseline-0",
        )
        state = {
            "root_pos_w_m": np.asarray(
                [[[0.0, 0.0, 10.0]] * 8, [[1.0, 0.0, 10.0]] * 8],
                dtype=np.float32,
            ),
            "command_time_ns": np.asarray([0, 5_000_000], dtype=np.int64),
            "effective_time_ns": np.asarray([5_000_000, 10_000_000], dtype=np.int64),
        }
        messages = {
            "sender_agent_id": np.tile(np.arange(8, dtype=np.int64), (2, 1)),
            "message_flags": np.ones((2, 8), dtype=np.uint8),
        }
        report = evaluate_condition_realization(
            request,
            receipt={
                "command": {"dt_s": 0.005},
                "physics": {"cf2x_hover_trim": {"hover_thrust_per_rotor_n": 0.06935, "initial_hover_rps": 263.742}},
            },
            scene={
                "environment_id": "RIVERMARK_CITY_LITE_v1",
                "fresh_stage": True,
                "search_object_prim_count": 4,
                "initial_root_poses_wxyz": [[0.0] * 7] * 8,
            },
            public_task={"agent_count": 8, "route_conditioning": "public_only", "routes_w_m": [[[0.0, 0.0, 10.0]] * 3] * 8, "nominal_object_count": 4},
            state=state,
            messages=messages,
            checks={"literal_fleet_spawn_verified": True},
        )
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["unavailable_axes"], [])
        self.assertEqual(report["axes"]["communication"]["status"], "verified")
        self.assertEqual(report["axes"]["control_latency"]["status"], "verified")

    def test_declared_unimplemented_axis_still_fails_closed(self) -> None:
        protocol = {
            "cells": [
                {
                    "cell_id": "train-baseline-0",
                    "split": "train",
                    "conditions": {"lighting": "lighting-nominal"},
                }
            ]
        }
        request = condition_request_from_protocol(
            protocol,
            protocol_id="citylite-coverage-v1",
            protocol_sha256="a" * 64,
            cell_id="train-baseline-0",
        )
        self.assertEqual(validate_condition_request(request), ())
        report = evaluate_condition_realization(
            request,
            receipt={},
            scene=None,
            public_task=None,
            state={},
            messages={},
            checks={},
        )
        self.assertEqual(report["status"], "unavailable")
        self.assertEqual(report["unavailable_axes"], ["lighting"])
        self.assertEqual(report["axes"]["lighting"]["status"], "unavailable")

    def test_route_realization_rejects_ragged_or_nonfinite_public_routes(self) -> None:
        request = condition_request_from_protocol(
            _protocol(),
            protocol_id="citylite-coverage-v1",
            protocol_sha256="a" * 64,
            cell_id="train-baseline-0",
        )
        state = {
            "root_pos_w_m": np.asarray([[[0.0, 0.0, 10.0]] * 8], dtype=np.float32),
        }
        messages = {
            "sender_agent_id": np.tile(np.arange(8, dtype=np.int64), (1, 1)),
            "message_flags": np.ones((1, 8), dtype=np.uint8),
        }
        public_task = {
            "agent_count": 8,
            "route_conditioning": "public_only",
            "routes_w_m": [[[0.0, 0.0, 10.0], [1.0, 0.0, 10.0]]] * 7
            + [[[0.0, 0.0, float("nan")], [1.0, 0.0, 10.0]]],
            "nominal_object_count": 4,
        }
        report = evaluate_condition_realization(
            request,
            receipt={},
            scene={"environment_id": "RIVERMARK_CITY_LITE_v1", "fresh_stage": True, "search_object_prim_count": 4},
            public_task=public_task,
            state=state,
            messages=messages,
            checks={},
        )
        self.assertEqual(report["axes"]["route"]["status"], "unavailable")
        self.assertTrue(any(issue["path"].endswith(".route") for issue in report["issues"]))


if __name__ == "__main__":
    unittest.main()
