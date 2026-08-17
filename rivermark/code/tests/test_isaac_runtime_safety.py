from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.citylite_scene import AABB
from rivermark_benchmark.isaac_runtime_safety import (
    CF2X_RUNTIME_GUARD_RADIUS_M,
    CONTACT_ABORT_FORCE_N,
    CONTACT_ABORT_FORCE_FLOAT32_CUTOFF_N,
    INTER_AGENT_MINIMUM_CENTER_SEPARATION_M,
    INTER_AGENT_PAIR_COUNT,
    RUNTIME_SAFETY_FRAME_OUTCOME_CODES,
    RuntimeSafetyAbort,
    RuntimeSafetyCheck,
    evaluate_runtime_safety,
    finalize_runtime_safety_guard,
    record_runtime_safety_abort,
    record_runtime_safety_check,
    physics_time_ns,
    runtime_safety_receipt_template,
)


def _positions(x: float = 0.0) -> list[list[float]]:
    return [[x + float(agent_id) * 2.0, 0.0, 11.0] for agent_id in range(8)]


def _forces() -> list[list[list[float]]]:
    return [[[0.0, 0.0, 0.0]] for _ in range(8)]


class RuntimeSafetyTests(unittest.TestCase):
    def test_physics_time_contract_is_decimal_deterministic_and_validated(self) -> None:
        self.assertEqual(physics_time_ns(0, 0.005), 0)
        self.assertEqual(physics_time_ns(1, 0.005), 5_000_000)
        self.assertEqual(physics_time_ns(3, 0.005), 15_000_000)
        self.assertEqual(physics_time_ns(1, 5.0e-10), 0)
        self.assertEqual(physics_time_ns(3, 5.0e-10), 2)
        self.assertEqual(RUNTIME_SAFETY_FRAME_OUTCOME_CODES, {"passed": 0, "aborted": 1})
        for step, dt_s in ((-1, 0.005), (True, 0.005), (0, 0.0), (0, math.inf)):
            with self.assertRaises(ValueError):
                physics_time_ns(step, dt_s)

    def test_safe_post_reset_and_swept_step_pass(self) -> None:
        boxes = (AABB((-40.0, -40.0, 0.0), (-39.0, -39.0, 1.0)),)
        start = _positions()
        post_reset = evaluate_runtime_safety(
            None, start, _forces(), boxes, phase="post_reset", physics_step=0
        )
        self.assertEqual(post_reset.agent_center_checks, 8)
        self.assertEqual(post_reset.initial_point_geometry_checks, 8)
        self.assertEqual(post_reset.swept_segments_checked, 0)
        self.assertEqual(post_reset.inter_agent_pair_checks, INTER_AGENT_PAIR_COUNT)
        self.assertGreater(
            post_reset.minimum_inter_agent_swept_separation_m,
            INTER_AGENT_MINIMUM_CENTER_SEPARATION_M,
        )
        current = _positions(0.01)
        step = evaluate_runtime_safety(
            start, current, _forces(), boxes, phase="rollout", physics_step=1
        )
        self.assertEqual(step.swept_segments_checked, 8)
        self.assertEqual(step.inter_agent_pair_checks, INTER_AGENT_PAIR_COUNT)
        self.assertEqual(step.max_contact_force_n, 0.0)

    def test_volume_is_checked_with_the_cf2x_body_radius(self) -> None:
        boxes = (AABB((-40.0, -40.0, 0.0), (-39.0, -39.0, 1.0)),)
        safe = _positions()
        safe[0][0] = 46.0 - CF2X_RUNTIME_GUARD_RADIUS_M
        evaluate_runtime_safety(None, safe, _forces(), boxes, phase="post_reset", physics_step=0)
        unsafe = _positions()
        unsafe[0][0] = 46.0 - CF2X_RUNTIME_GUARD_RADIUS_M + 1.0e-6
        with self.assertRaises(RuntimeSafetyAbort) as raised:
            evaluate_runtime_safety(None, unsafe, _forces(), boxes, phase="post_reset", physics_step=0)
        self.assertEqual(raised.exception.violation["kind"], "flight_volume_violation")

    def test_swept_aabb_guard_rejects_tunneling(self) -> None:
        boxes = (AABB((-0.1, -0.1, 10.0), (0.1, 0.1, 12.0), source_prim="/World/Test"),)
        previous = _positions(-4.0)
        current = _positions(-4.0)
        previous[0] = [-2.0, 0.0, 11.0]
        current[0] = [2.0, 0.0, 11.0]
        with self.assertRaises(RuntimeSafetyAbort) as raised:
            evaluate_runtime_safety(previous, current, _forces(), boxes, phase="rollout", physics_step=7)
        violation = raised.exception.violation
        self.assertEqual(violation["kind"], "structural_aabb_clearance_violation")
        self.assertEqual(violation["agent_id"], 0)
        self.assertEqual(violation["source_prim"], "/World/Test")

    def test_synchronized_relative_sweep_rejects_midstep_agent_crossing(self) -> None:
        boxes = (AABB((-40.0, -40.0, 0.0), (-39.0, -39.0, 1.0)),)
        previous = _positions()
        current = _positions()
        previous[0] = [-1.0, 0.0, 11.0]
        previous[1] = [1.0, 0.0, 11.0]
        current[0] = [1.0, 0.0, 11.0]
        current[1] = [-1.0, 0.0, 11.0]
        with self.assertRaises(RuntimeSafetyAbort) as raised:
            evaluate_runtime_safety(
                previous, current, _forces(), boxes, phase="rollout", physics_step=7
            )
        violation = raised.exception.violation
        self.assertEqual(violation["kind"], "inter_agent_swept_separation_violation")
        self.assertEqual((violation["left_agent_id"], violation["right_agent_id"]), (0, 1))
        self.assertEqual(violation["closest_segment_time"], 0.5)
        self.assertEqual(violation["minimum_center_separation_m"], 0.0)

    def test_contact_data_is_exact_shape_finite_and_thresholded(self) -> None:
        boxes = (AABB((-40.0, -40.0, 0.0), (-39.0, -39.0, 1.0)),)
        with self.assertRaises(RuntimeSafetyAbort) as malformed:
            evaluate_runtime_safety(None, _positions(), [[[0.0, 0.0, 0.0]]], boxes, phase="post_reset", physics_step=0)
        self.assertEqual(malformed.exception.violation["kind"], "invalid_contact_sensor_data")
        nonfinite = _forces()
        nonfinite[2][0][1] = math.nan
        with self.assertRaises(RuntimeSafetyAbort) as bad_number:
            evaluate_runtime_safety(None, _positions(), nonfinite, boxes, phase="post_reset", physics_step=0)
        self.assertEqual(bad_number.exception.violation["kind"], "invalid_contact_sensor_data")
        hit = _forces()
        hit[3][0][2] = CONTACT_ABORT_FORCE_N
        with self.assertRaises(RuntimeSafetyAbort) as contact:
            evaluate_runtime_safety(None, _positions(), hit, boxes, phase="post_reset", physics_step=0)
        violation = contact.exception.violation
        self.assertEqual(violation["kind"], "contact_force_violation")
        self.assertEqual(violation["agent_id"], 3)

        float32_boundary = _forces()
        float32_boundary[5][0][2] = float(np.float32(CONTACT_ABORT_FORCE_N))
        self.assertLess(float32_boundary[5][0][2], CONTACT_ABORT_FORCE_N)
        self.assertEqual(
            float32_boundary[5][0][2], CONTACT_ABORT_FORCE_FLOAT32_CUTOFF_N
        )
        with self.assertRaises(RuntimeSafetyAbort) as float32_contact:
            evaluate_runtime_safety(
                None,
                _positions(),
                float32_boundary,
                boxes,
                phase="post_reset",
                physics_step=0,
            )
        self.assertEqual(
            float32_contact.exception.violation["kind"], "contact_force_violation"
        )
        self.assertEqual(
            float32_contact.exception.violation["force_abort_float32_cutoff_n"],
            CONTACT_ABORT_FORCE_FLOAT32_CUTOFF_N,
        )

    def test_receipt_template_freezes_geometry_and_contact_contract(self) -> None:
        boxes = (AABB((-40.0, -40.0, 0.0), (-39.0, -39.0, 1.0)),)
        receipt = runtime_safety_receipt_template(
            boxes,
            contact_prim_expression="/World/Swarm/Agent_.*/Robot/body",
            physics_dt_s=0.005,
        )
        self.assertTrue(receipt["enabled"])
        self.assertTrue(receipt["fail_closed"])
        self.assertEqual(receipt["agent_center_radius_m"], CF2X_RUNTIME_GUARD_RADIUS_M)
        self.assertEqual(receipt["swept_aabb_clearance_m"], 0.85)
        self.assertEqual(receipt["contact"]["force_abort_threshold_n"], CONTACT_ABORT_FORCE_N)
        self.assertEqual(
            receipt["contact"]["force_abort_float32_cutoff_n"],
            CONTACT_ABORT_FORCE_FLOAT32_CUTOFF_N,
        )
        self.assertTrue(receipt["contact"]["every_physics_step"])
        self.assertEqual(receipt["evidence"]["path"], "sensors/runtime_safety.npz")

    def test_receipt_accounting_and_abort_are_explicit(self) -> None:
        boxes = (AABB((-40.0, -40.0, 0.0), (-39.0, -39.0, 1.0)),)
        receipt = runtime_safety_receipt_template(
            boxes,
            contact_prim_expression="/World/Swarm/Agent_.*/Robot/body",
            physics_dt_s=0.005,
        )
        check = RuntimeSafetyCheck(8, 8, 0, INTER_AGENT_PAIR_COUNT, 2.0, 1, 0.0)
        record_runtime_safety_check(receipt, check, phase="post_reset")
        record_runtime_safety_check(
            receipt,
            RuntimeSafetyCheck(8, 0, 8, INTER_AGENT_PAIR_COUNT, 1.5, 1, 0.002),
            phase="rollout",
        )
        checks = receipt["checks"]
        self.assertEqual(checks["post_reset_agent_center_checks"], 8)
        self.assertEqual(checks["rollout_physics_steps_checked"], 1)
        self.assertEqual(checks["swept_segments_checked"], 8)
        self.assertEqual(checks["inter_agent_pair_checks"], 2 * INTER_AGENT_PAIR_COUNT)
        self.assertEqual(checks["minimum_inter_agent_swept_separation_m"], 1.5)
        self.assertEqual(checks["max_contact_force_n"], 0.002)

        aborted = runtime_safety_receipt_template(
            boxes,
            contact_prim_expression="/World/Swarm/Agent_.*/Robot/body",
            physics_dt_s=0.005,
        )
        record_runtime_safety_abort(
            aborted,
            RuntimeSafetyAbort({"kind": "contact_force_violation", "physics_step": 1}),
        )
        self.assertEqual(aborted["status"], "aborted")
        self.assertEqual(aborted["checks"]["contact_abort_count"], 1)

        finalize_runtime_safety_guard(receipt, trace_sha256="a" * 64, physics_frame_count=2)
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(receipt["evidence"]["physics_frame_count"], 2)


if __name__ == "__main__":
    unittest.main()
