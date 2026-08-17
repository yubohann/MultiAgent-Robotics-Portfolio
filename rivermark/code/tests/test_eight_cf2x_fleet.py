from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ImportError as error:
    raise unittest.SkipTest("PyTorch is required for EightCF2XFleet tests") from error


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rivermark_benchmark.eight_cf2x_fleet import EightCF2XFleet


PRIM_EXPRESSION = "/World/Swarm/Agent_.*/Robot"
LITERAL_PATHS = tuple(f"/World/Swarm/Agent_{agent_id}/Robot" for agent_id in range(8))
ALLOCATION = torch.tensor(
    (
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0, 1.0),
        (-0.046, 0.046, 0.046, -0.046),
        (-0.046, -0.046, 0.046, 0.046),
        (0.006, -0.006, 0.006, -0.006),
    ),
    dtype=torch.float32,
)


class _FakeData:
    def __init__(self, agent_id: int) -> None:
        base = float(agent_id)
        self.root_pos_w = torch.tensor([[base, base + 0.1, 9.0 + base]])
        self.root_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        self.root_lin_vel_w = torch.tensor([[base, base + 1.0, base + 2.0]])
        self.root_ang_vel_b = torch.tensor([[base + 3.0, base + 4.0, base + 5.0]])
        self.thrust_target = torch.full((1, 4), base)
        self.computed_thrust = torch.full((1, 4), base + 0.25)
        self.applied_thrust = torch.full((1, 4), base + 0.5)
        self.default_thruster_rps = torch.full((1, 4), 263.34388)
        self.thruster_names = ["m1_prop", "m2_prop", "m3_prop", "m4_prop"]
        self.unbatchable_scalar = torch.tensor(base)


class _FakeRobot:
    def __init__(self, agent_id: int, events: list[tuple[int, str]]) -> None:
        self.agent_id = agent_id
        self.events = events
        self.is_initialized = True
        self.num_instances = 1
        self.num_thrusters = 4
        self.device = "cpu"
        self.cfg = SimpleNamespace(prim_path=LITERAL_PATHS[agent_id])
        self.body_names = ["body", "m1_prop", "m2_prop", "m3_prop", "m4_prop"]
        self.allocation_matrix = ALLOCATION.clone()
        self.data = _FakeData(agent_id)
        self.received_targets: list[torch.Tensor] = []

    def reset(self) -> None:
        self.events.append((self.agent_id, "reset"))

    def set_thrust_target(self, target: torch.Tensor) -> None:
        self.received_targets.append(target.clone())
        self.data.thrust_target = target.clone()
        self.events.append((self.agent_id, "target"))

    def write_data_to_sim(self) -> None:
        self.events.append((self.agent_id, "write"))

    def update(self, dt: float) -> None:
        if dt <= 0.0:
            raise AssertionError("test fake requires a positive physics dt")
        self.events.append((self.agent_id, "update"))


def _fleet() -> tuple[EightCF2XFleet, tuple[_FakeRobot, ...], list[tuple[int, str]]]:
    events: list[tuple[int, str]] = []
    robots = tuple(_FakeRobot(agent_id, events) for agent_id in range(8))
    return (
        EightCF2XFleet(
            robots,
            prim_expression=PRIM_EXPRESSION,
            literal_prim_paths=LITERAL_PATHS,
        ),
        robots,
        events,
    )


class EightCF2XFleetTests(unittest.TestCase):
    def test_data_reads_are_batch_ordered_snapshots(self) -> None:
        fleet, robots, _events = _fleet()
        self.assertEqual(fleet.num_instances, 8)
        self.assertEqual(fleet.num_thrusters, 4)
        self.assertEqual(fleet.device, "cpu")
        self.assertEqual(fleet.cfg.prim_path, PRIM_EXPRESSION)
        self.assertEqual(fleet.cfg.literal_prim_paths, LITERAL_PATHS)
        self.assertEqual(tuple(fleet.data.root_pos_w.shape), (8, 3))
        self.assertTrue(torch.equal(fleet.data.root_pos_w[:, 0], torch.arange(8.0)))
        self.assertEqual(tuple(fleet.data.applied_thrust.shape), (8, 4))
        self.assertEqual(fleet.data.thruster_names, ["m1_prop", "m2_prop", "m3_prop", "m4_prop"])
        self.assertEqual(fleet.body_names, ["body", "m1_prop", "m2_prop", "m3_prop", "m4_prop"])

        snapshot = fleet.data.thrust_target
        snapshot[0, 0] = 999.0
        self.assertEqual(float(robots[0].data.thrust_target[0, 0]), 0.0)
        copied_allocation = fleet.allocation_matrix
        copied_allocation[2, 0] = 999.0
        self.assertEqual(float(fleet.allocation_matrix[2, 0]), 1.0)

    def test_full_batch_command_and_step_lifecycle_are_delegated_in_order(self) -> None:
        fleet, robots, events = _fleet()
        command = torch.arange(32.0, dtype=torch.float32).reshape(8, 4)
        fleet.set_thrust_target(command, thruster_ids=None, env_ids=None)
        fleet.write_data_to_sim()
        fleet.update(0.005)
        fleet.reset()

        self.assertEqual(
            events,
            [(agent_id, phase) for phase in ("target", "write", "update", "reset") for agent_id in range(8)],
        )
        for agent_id, robot in enumerate(robots):
            self.assertEqual(len(robot.received_targets), 1)
            self.assertTrue(torch.equal(robot.received_targets[0], command[agent_id : agent_id + 1]))

    def test_partial_command_requests_fail_closed(self) -> None:
        fleet, _robots, _events = _fleet()
        command = torch.zeros((8, 4), dtype=torch.float32)
        with self.assertRaises(NotImplementedError):
            fleet.set_thrust_target(command, env_ids=[0])
        with self.assertRaises(NotImplementedError):
            fleet.set_thrust_target(command, thruster_ids=[0])
        with self.assertRaises(ValueError):
            fleet.set_thrust_target(torch.zeros((8, 3), dtype=torch.float32))
        with self.assertRaises(NotImplementedError):
            fleet.reset(env_ids=[0])

    def test_literal_order_initialization_and_allocation_drift_are_rejected(self) -> None:
        events: list[tuple[int, str]] = []
        robots = tuple(_FakeRobot(agent_id, events) for agent_id in range(8))
        wrong_paths = tuple(reversed(LITERAL_PATHS))
        with self.assertRaisesRegex(RuntimeError, "construction order"):
            EightCF2XFleet(robots, prim_expression=PRIM_EXPRESSION, literal_prim_paths=wrong_paths)

        robots = tuple(_FakeRobot(agent_id, events) for agent_id in range(8))
        robots[5].allocation_matrix[2, 0] = 0.5
        with self.assertRaisesRegex(RuntimeError, "allocation matrices"):
            EightCF2XFleet(robots, prim_expression=PRIM_EXPRESSION, literal_prim_paths=LITERAL_PATHS)

        robots = tuple(_FakeRobot(agent_id, events) for agent_id in range(8))
        robots[2].is_initialized = False
        with self.assertRaisesRegex(RuntimeError, "initialized"):
            EightCF2XFleet(robots, prim_expression=PRIM_EXPRESSION, literal_prim_paths=LITERAL_PATHS)

        robots = tuple(_FakeRobot(agent_id, events) for agent_id in range(8))
        robots[3].body_names = ["body", "m2_prop", "m1_prop", "m3_prop", "m4_prop"]
        with self.assertRaisesRegex(RuntimeError, "body orders"):
            EightCF2XFleet(robots, prim_expression=PRIM_EXPRESSION, literal_prim_paths=LITERAL_PATHS)

    def test_unbatchable_data_and_root_relocation_surface_are_absent(self) -> None:
        fleet, _robots, _events = _fleet()
        with self.assertRaisesRegex(AttributeError, r"not \[1, \.\.\.\]"):
            _ = fleet.data.unbatchable_scalar
        self.assertFalse(hasattr(fleet, "write_root_pose_to_sim"))
        self.assertFalse(hasattr(fleet, "write_root_velocity_to_sim"))


if __name__ == "__main__":
    unittest.main()
