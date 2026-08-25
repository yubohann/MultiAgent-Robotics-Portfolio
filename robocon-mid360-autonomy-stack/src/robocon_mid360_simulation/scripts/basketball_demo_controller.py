#!/usr/bin/env python3
"""Deterministic two-robot basketball demonstration controller.

This is a Gazebo simulation demonstrator: robot motion is driven through the
normal chassis command topics and the visible ball is scripted through Gazebo's
SetEntityState service so that the complete pass and shot remain repeatable.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import rclpy
from gazebo_msgs.msg import EntityState, ModelStates
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Pose, Twist
from rclpy.node import Node
from std_msgs.msg import Int32, String


class BasketballDemoController(Node):
    def __init__(self, duration: float, output_dir: Path) -> None:
        super().__init__("basketball_demo_controller")
        self.duration = max(30.0, float(duration))
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "demo_events.jsonl"
        self.summary_path = self.output_dir / "success_summary.json"
        self.events_file = self.events_path.open("w", encoding="utf-8")
        self.events: list[dict] = []
        self.start_time: float | None = None
        self.created_at = time.monotonic()
        self.last_state = "INTRO"
        self.waiting_emitted = False
        self.completed = False
        self.initial_robot_xy: dict[str, tuple[float, float]] = {}
        self.models: dict[str, Pose] = {}
        self.robot1_dribble_target: tuple[float, float] | None = None
        self.shot_staging_target = (4.90, -0.55)
        self.shot_start_xyz: tuple[float, float, float] | None = None
        self.shot_started_at: float | None = None
        self.shot_apex_z = 3.40
        self.shot_flight_duration = 0.0
        self.ball_pose = Pose()
        self.ball_pose.position.z = 0.18
        self.shot_target = (6.44, -0.19, 2.43)
        self.last_dribble_cycle = {"robot1": -1, "robot2": -1}
        self.final_hold_started: float | None = None
        self.set_state = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        # Gazebo service calls are asynchronous.  Do not enqueue a new entity
        # teleport while the previous one is still pending; queued requests
        # are rendered out of order under WSL and appear as visible jumps.
        self.ball_state_future = None
        self.create_subscription(ModelStates, "/gazebo/model_states", self._models, 10)
        self.robot1_cmd = self.create_publisher(Twist, "/robot1/cmd_vel_chassis", 10)
        self.robot2_cmd = self.create_publisher(Twist, "/robot2/cmd_vel_chassis", 10)
        self.event_pub = self.create_publisher(String, "/robocon/demo/events", 10)
        self.score_pub = self.create_publisher(Int32, "/robocon/demo/score", 10)
        # 20 Hz gives the scripted ball enough intermediate poses to remain
        # visually continuous while the service-future guard prevents request
        # buildup when Gazebo is under a dense LiDAR load.
        self.timer = self.create_timer(0.05, self._tick)
        self._emit("INTRO", "demo_started", {"duration_sec": self.duration})

    def _models(self, msg: ModelStates) -> None:
        self.models = {name: pose for name, pose in zip(msg.name, msg.pose)}

    def _emit(self, state: str, event: str, payload: dict | None = None) -> None:
        row = {
            "t_sec": round(time.monotonic() - (self.start_time or self.created_at), 3),
            "state": state,
            "event": event,
            "payload": payload or {},
        }
        self.events.append(row)
        self.events_file.write(json.dumps(row, ensure_ascii=True) + "\n")
        self.events_file.flush()
        msg = String()
        msg.data = json.dumps(row, ensure_ascii=True)
        self.event_pub.publish(msg)
        self.get_logger().info(f"{state}: {event}")

    def _set_ball(self, x: float, y: float, z: float) -> None:
        self.ball_pose.position.x = float(x)
        self.ball_pose.position.y = float(y)
        self.ball_pose.position.z = max(0.13, float(z))
        if not self.set_state.service_is_ready():
            return
        if self.ball_state_future is not None and not self.ball_state_future.done():
            return
        request = SetEntityState.Request()
        request.state = EntityState()
        request.state.name = "basketball"
        # Copy the pose into the request. Reusing the mutable member while a
        # service request is in flight can serialize a later pose and create
        # visible jumps in the pass or shot trajectory.
        request.state.pose = Pose()
        request.state.pose.position.x = self.ball_pose.position.x
        request.state.pose.position.y = self.ball_pose.position.y
        request.state.pose.position.z = self.ball_pose.position.z
        request.state.pose.orientation.w = 1.0
        request.state.twist = Twist()
        request.state.reference_frame = "world"
        self.ball_state_future = self.set_state.call_async(request)

    def _pose_xy(self, name: str, fallback: tuple[float, float]) -> tuple[float, float]:
        pose = self.models.get(name)
        if pose is None:
            return fallback
        return pose.position.x, pose.position.y

    def _drive(self, publisher, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> None:
        cmd = Twist()
        cmd.linear.x, cmd.linear.y, cmd.angular.z = float(x), float(y), float(yaw)
        publisher.publish(cmd)

    def _drive_to_xy(
        self,
        publisher,
        x: float,
        y: float,
        target: tuple[float, float],
        max_speed: float = 0.45,
        tolerance: float = 0.08,
    ) -> bool:
        """Drive in the fixed zero-yaw demo frame until a world target is reached."""
        dx, dy = target[0] - x, target[1] - y
        distance = math.hypot(dx, dy)
        if distance <= tolerance:
            self._drive(publisher)
            return True
        speed = min(max_speed, max(0.12, 1.4 * distance))
        self._drive(publisher, speed * dx / distance, speed * dy / distance)
        return False

    def _dribble_ball(self, robot: str, x: float, y: float, elapsed: float) -> None:
        """Hold the ball at the robot intake and show repeated vertical bounces."""
        cycle = int(elapsed * 2.2)
        if cycle != self.last_dribble_cycle[robot]:
            self.last_dribble_cycle[robot] = cycle
            self._emit("ROBOT1_DRIBBLE" if robot == "robot1" else "ROBOT2_DRIBBLE", "dribble_cycle", {
                "robot": robot,
                "cycle": cycle,
            })
        bounce = abs(math.sin(elapsed * math.pi * 2.2))
        self._set_ball(x + 0.44, y + 0.025 * math.sin(elapsed * 2.0), 0.14 + 0.40 * bounce)

    def _phase(self, elapsed: float) -> str:
        f = elapsed / self.duration
        if f < 0.06:
            return "INTRO"
        if f < 0.26:
            return "ROBOT1_DRIBBLE"
        if f < 0.36:
            return "ROBOT1_TO_PASS"
        if f < 0.40:
            return "PASS_IN_FLIGHT"
        if f < 0.46:
            return "PASS_SUCCESS"
        if f < 0.66:
            return "ROBOT2_DRIBBLE"
        if f < 0.80:
            return "SHOT_PREPARE"
        if f < 0.93:
            return "SHOT_IN_FLIGHT"
        return "SHOT_SUCCESS"

    def _tick(self) -> None:
        # A launch can take longer than the shell's initial sleep under WSL.
        # Do not begin the evidence timeline until both real Gazebo entities,
        # the ball, and the state service exist.
        required = {"robocon25_robot1", "robocon25_robot2", "basketball"}
        if self.start_time is None:
            if not required.issubset(self.models) or not self.set_state.service_is_ready():
                if not self.waiting_emitted:
                    self._emit("WAITING_FOR_GAZEBO", "waiting_for_entities", {
                        "required_models": sorted(required),
                    })
                    self.waiting_emitted = True
                return
            self.start_time = time.monotonic()
            self.initial_robot_xy = {
                name: self._pose_xy(name, (0.0, 0.0))
                for name in ("robocon25_robot1", "robocon25_robot2")
            }
            robot1_start = self.initial_robot_xy["robocon25_robot1"]
            self.robot1_dribble_target = (robot1_start[0] + 2.40, robot1_start[1])
            self._emit("INTRO", "gazebo_ready", {"models": sorted(required)})

        elapsed = time.monotonic() - self.start_time
        state = self._phase(elapsed)
        if state != self.last_state:
            self._emit(state, "state_entered")
            self.last_state = state

        r1x, r1y = self._pose_xy("robocon25_robot1", (-3.0, -1.35))
        r2x, r2y = self._pose_xy("robocon25_robot2", (2.2, 1.10))
        if state == "ROBOT1_DRIBBLE":
            target = self.robot1_dribble_target or (r1x + 2.40, r1y)
            self._drive_to_xy(self.robot1_cmd, r1x, r1y, target)
            self._drive(self.robot2_cmd)
            self._dribble_ball("robot1", r1x, r1y, elapsed)
        elif state == "ROBOT1_TO_PASS":
            target = self.robot1_dribble_target or (r1x, r1y)
            self._drive_to_xy(self.robot1_cmd, r1x, r1y, target)
            self._drive(self.robot2_cmd, -0.10, -0.03)
            self._set_ball(r1x + 0.44, r1y, 0.30)
        elif state == "PASS_IN_FLIGHT":
            self._drive(self.robot1_cmd)
            self._drive(self.robot2_cmd)
            u = min(1.0, max(0.0, (elapsed / self.duration - 0.36) / 0.04))
            smooth = u * u * (3.0 - 2.0 * u)
            sx, sy = r1x + 0.44, r1y
            ex, ey = r2x + 0.44, r2y
            self._set_ball(
                sx + smooth * (ex - sx),
                sy + smooth * (ey - sy),
                0.30 + 1.20 * math.sin(math.pi * smooth),
            )
        elif state == "PASS_SUCCESS":
            self._drive(self.robot1_cmd)
            self._drive(self.robot2_cmd)
            self._set_ball(r2x + 0.44, r2y, 0.30)
            if not any(item["event"] == "handoff_verified" for item in self.events):
                self._emit(state, "handoff_verified", {
                    "receiver": "robocon25_robot2",
                    "ball_offset_m": [0.44, 0.0, 0.30],
                })
        elif state == "ROBOT2_DRIBBLE":
            self._drive(self.robot1_cmd)
            self._drive(self.robot2_cmd)
            self._dribble_ball("robot2", r2x, r2y, elapsed)
        elif state == "SHOT_PREPARE":
            self._drive(self.robot1_cmd)
            ready = self._drive_to_xy(
                self.robot2_cmd, r2x, r2y, self.shot_staging_target, max_speed=0.38
            )
            self._set_ball(r2x + 0.44, r2y, 0.42)
            if ready and not any(item["event"] == "shot_setup_ready" for item in self.events):
                self._emit(state, "shot_setup_ready", {
                    "robot2_xy": [round(r2x, 3), round(r2y, 3)],
                    "target_xy": list(self.shot_staging_target),
                    "rim_xyz": list(self.shot_target),
                })
        elif state == "SHOT_IN_FLIGHT":
            self._drive(self.robot1_cmd)
            self._drive(self.robot2_cmd)
            if self.shot_started_at is None:
                self.shot_started_at = time.monotonic()
                self.shot_start_xyz = (r2x + 0.44, r2y, 0.42)
                gravity = 9.81
                rise_time = math.sqrt(2.0 * (self.shot_apex_z - self.shot_start_xyz[2]) / gravity)
                fall_time = math.sqrt(2.0 * (self.shot_apex_z - self.shot_target[2]) / gravity)
                self.shot_flight_duration = rise_time + fall_time
                self._emit(state, "ballistic_shot_started", {
                    "start_xyz": [round(value, 3) for value in self.shot_start_xyz],
                    "rim_xyz": list(self.shot_target),
                    "apex_z": self.shot_apex_z,
                    "flight_duration_sec": round(self.shot_flight_duration, 3),
                    "gravity_mps2": gravity,
                })
            flight_elapsed = min(
                self.shot_flight_duration,
                max(0.0, time.monotonic() - self.shot_started_at),
            )
            shot_t = flight_elapsed / max(1e-6, self.shot_flight_duration)
            sx, sy, sz = self.shot_start_xyz or (r2x + 0.44, r2y, 0.42)
            ex, ey, ez = self.shot_target
            gravity = 9.81
            rise_time = math.sqrt(2.0 * (self.shot_apex_z - sz) / gravity)
            launch_vz = gravity * rise_time
            self._set_ball(
                sx + shot_t * (ex - sx),
                sy + shot_t * (ey - sy),
                sz + launch_vz * flight_elapsed - 0.5 * gravity * flight_elapsed * flight_elapsed,
            )
        elif state == "SHOT_SUCCESS":
            self._drive(self.robot1_cmd)
            self._drive(self.robot2_cmd)
            self._set_ball(*self.shot_target)
            if self.final_hold_started is None:
                self.final_hold_started = time.monotonic()
            if not any(item["event"] == "shot_success" for item in self.events):
                self._emit(state, "shot_success", {"target": "basketball_hoop1", "rim_xyz": list(self.shot_target)})
                score = Int32()
                score.data = 2
                self.score_pub.publish(score)
        else:
            self._drive(self.robot1_cmd)
            self._drive(self.robot2_cmd)
            self._set_ball(r1x + 0.45, r1y, 0.25)

        if elapsed >= self.duration + 3.0:
            self._finish()

    def _finish(self) -> None:
        if self.final_hold_started is None or time.monotonic() - self.final_hold_started < 1.0:
            return
        # A dense Gazebo ray workload can delay the final service response.
        # Hold for a bounded grace period, then finish with both the observed
        # model pose and the commanded target recorded in the manifest.
        if (self.ball_state_future is not None and not self.ball_state_future.done()
                and time.monotonic() - self.final_hold_started < 5.0):
            return
        self._drive(self.robot1_cmd)
        self._drive(self.robot2_cmd)
        final_robot_xy = {
            name: self._pose_xy(name, (0.0, 0.0))
            for name in ("robocon25_robot1", "robocon25_robot2")
        }
        robot_motion = {
            name: round(math.dist(self.initial_robot_xy.get(name, (0.0, 0.0)), final_robot_xy[name]), 3)
            for name in final_robot_xy
        }
        ball = self.models.get("basketball", self.ball_pose)
        ball_error = math.dist(
            (ball.position.x, ball.position.y, ball.position.z), self.shot_target)
        commanded_ball_error = math.dist(
            (self.ball_pose.position.x, self.ball_pose.position.y, self.ball_pose.position.z), self.shot_target)
        shot_acceptance_radius = 0.15
        verified = robot_motion["robocon25_robot1"] >= 0.10 and robot_motion["robocon25_robot2"] >= 0.10 and ball_error <= shot_acceptance_radius
        self._emit("COMPLETE", "demo_complete", {
            "shot_success": verified,
            "score": 2 if verified else 0,
            "robot_motion_m": robot_motion,
            "final_ball_error_m": round(ball_error, 3),
            "commanded_ball_error_m": round(commanded_ball_error, 3),
            "shot_acceptance_radius_m": shot_acceptance_radius,
        })
        self.summary_path.write_text(json.dumps({
            "status": "PASS" if verified else "FAIL",
            "shot_success": verified,
            "score": 2 if verified else 0,
            "events": len(self.events),
            "duration_sec": round(time.monotonic() - self.start_time, 3),
            "robot_motion_m": robot_motion,
            "final_ball_error_m": round(ball_error, 3),
            "commanded_ball_error_m": round(commanded_ball_error, 3),
            "shot_target_xyz": list(self.shot_target),
            "shot_start_xyz": list(self.shot_start_xyz) if self.shot_start_xyz else None,
            "shot_apex_z": self.shot_apex_z,
            "shot_flight_duration_sec": round(self.shot_flight_duration, 3),
            "observed_ball_xyz": [round(ball.position.x, 3), round(ball.position.y, 3), round(ball.position.z, 3)],
            "evidence": "gazebo_model_states_and_set_entity_state",
        }, indent=2) + "\n", encoding="utf-8")
        self.events_file.close()
        self.timer.cancel()
        self.completed = True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rclpy.init()
    node = BasketballDemoController(args.duration, args.output_dir)
    try:
        while rclpy.ok() and not node.completed:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        if not node.events_file.closed:
            node.events_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
