"""A read-batched facade over eight literal one-instance IsaacLab CF2X assets.

IsaacLab initializes PhysX handles only after ``SimulationContext.reset()``.
The facade is therefore constructed after reset, while the eight literal
``MultirotorCfg`` objects author their independent initial states before it.
It intentionally exposes command delegation but no root-state rewrite API.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any, Sequence


class _FleetData:
    """Snapshot-batch compatible data access for one-instance Multirotors."""

    def __init__(self, robots: tuple[Any, ...]) -> None:
        self._robots = robots

    def __getattr__(self, name: str) -> Any:
        values = [getattr(robot.data, name) for robot in self._robots]
        torch = importlib.import_module("torch")
        if all(torch.is_tensor(value) for value in values):
            if any(value.ndim < 1 or value.shape[0] != 1 for value in values):
                raise AttributeError(
                    f"fleet data field {name!r} is not [1, ...] per CF2X"
                )
            return torch.cat(values, dim=0)
        if name == "thruster_names" and all(
            value == values[0] for value in values[1:]
        ):
            return list(values[0])
        raise AttributeError(f"unsupported non-batched fleet data field {name!r}")


class EightCF2XFleet:
    """A fixed-order eight-agent facade without an unsafe relocation surface."""

    def __init__(
        self,
        robots: Sequence[Any],
        *,
        prim_expression: str,
        literal_prim_paths: Sequence[str],
    ) -> None:
        self._robots = tuple(robots)
        expected_paths = tuple(str(path) for path in literal_prim_paths)
        if len(self._robots) != 8 or len(expected_paths) != 8:
            raise RuntimeError("fleet requires exactly eight literal CF2X assets")
        if len(set(expected_paths)) != 8:
            raise RuntimeError("literal CF2X prim paths must be unique")
        if not all(robot.is_initialized for robot in self._robots):
            raise RuntimeError("fleet members must be initialized after simulation reset")
        if any(
            robot.num_instances != 1 or robot.num_thrusters != 4
            for robot in self._robots
        ):
            raise RuntimeError("every fleet member must be one four-thruster CF2X")
        actual_paths = tuple(str(robot.cfg.prim_path) for robot in self._robots)
        if actual_paths != expected_paths:
            raise RuntimeError("literal CF2X construction order does not match the fleet contract")

        self._device = self._robots[0].device
        self._allocation_matrix = self._robots[0].allocation_matrix.detach().clone()
        torch = importlib.import_module("torch")
        for robot in self._robots[1:]:
            if str(robot.device) != str(self._device):
                raise RuntimeError("fleet members use different devices")
            if not torch.equal(robot.allocation_matrix, self._allocation_matrix):
                raise RuntimeError("fleet members use different allocation matrices")

        self.cfg = SimpleNamespace(
            prim_path=str(prim_expression), literal_prim_paths=expected_paths
        )
        self._data = _FleetData(self._robots)
        body_names = tuple(str(name) for name in getattr(self._robots[0], "body_names", ()))
        if not body_names:
            raise RuntimeError("CF2X fleet members expose no body names")
        if any(
            tuple(str(name) for name in getattr(robot, "body_names", ())) != body_names
            for robot in self._robots[1:]
        ):
            raise RuntimeError("fleet members use different CF2X body orders")
        self._body_names = body_names

    @property
    def data(self) -> _FleetData:
        return self._data

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_initialized(self) -> bool:
        return all(robot.is_initialized for robot in self._robots)

    @property
    def num_instances(self) -> int:
        return 8

    @property
    def num_thrusters(self) -> int:
        return 4

    @property
    def allocation_matrix(self) -> Any:
        return self._allocation_matrix.clone()

    @property
    def body_names(self) -> list[str]:
        """Return the shared literal CF2X link order for parent-frame audits."""

        return list(self._body_names)

    def reset(self, env_ids: Any = None) -> None:
        if env_ids is not None:
            raise NotImplementedError("City-Lite fleet resets all eight agents together")
        for robot in self._robots:
            robot.reset()

    def set_thrust_target(
        self,
        target: Any,
        thruster_ids: Any = None,
        env_ids: Any = None,
    ) -> None:
        """Delegate one complete [8, 4] force command in stable agent order.

        The named optional parameters match ``Multirotor.set_thrust_target`` so
        callers can pass its default values explicitly. Partial selections have
        no safe batch meaning across eight independent PhysX views, so reject
        them rather than silently changing the command's agent ordering.
        """
        if thruster_ids is not None or env_ids is not None:
            raise NotImplementedError("City-Lite fleet accepts only full-batch thrust commands")
        if tuple(target.shape) != (8, 4):
            raise ValueError("fleet expects one full [8,4] thrust-target tensor")
        for agent_id, robot in enumerate(self._robots):
            robot.set_thrust_target(target[agent_id : agent_id + 1])

    def write_data_to_sim(self) -> None:
        for robot in self._robots:
            robot.write_data_to_sim()

    def update(self, dt: float) -> None:
        for robot in self._robots:
            robot.update(dt)
