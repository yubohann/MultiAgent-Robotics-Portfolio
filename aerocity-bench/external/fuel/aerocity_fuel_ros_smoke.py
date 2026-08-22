"""Exercise FUEL's ROS 1 planner boundary without starting its controller.

This helper executes only inside the locked GPL container.  It starts a private
ROS master, launches FUEL's exploration node through ``algorithm.xml``, and
feeds a synthetic *public-style* local point cloud, sensor pose, odometry, and
route trigger.  Its only planner output is ``/planning/bspline``.  The FUEL
trajectory server and all position-command topics are deliberately absent, so
this script cannot command an AeroCityBench vehicle or bypass the shared CF2X
controller.

It is an interface smoke test, not a benchmark episode: the synthetic cloud is
not an AeroCity city, no target/evaluator data exists in this process, and a
route emission does not imply G1-U or G2-I compatibility.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Any

import rospy
from bspline.msg import Bspline
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2

TRIGGER_WARMUP_S = 3.0


@dataclass
class SmokeState:
    bspline_message_count: int = 0
    first_bspline_control_point_count: int | None = None
    first_bspline_order: int | None = None

    def record_bspline(self, message: Bspline) -> None:
        self.bspline_message_count += 1
        if self.first_bspline_control_point_count is None:
            self.first_bspline_control_point_count = len(message.pos_pts)
            self.first_bspline_order = int(message.order)


def _child(command: list[str], log_path: str) -> subprocess.Popen[bytes]:
    log = open(log_path, "wb")
    try:
        return subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log.close()


def _log_tail(path: str) -> str:
    try:
        return open(path, encoding="utf-8", errors="replace").read()[-2_000:].strip()
    except OSError:
        return "no child log was produced"


def _stop(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=2.0)
    except (OSError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass


def _odometry(stamp: rospy.Time) -> Odometry:
    message = Odometry()
    message.header.stamp = stamp
    message.header.frame_id = "world"
    message.child_frame_id = "cf2x_body"
    message.pose.pose.position.z = 1.5
    message.pose.pose.orientation.w = 1.0
    return message


def _pose(stamp: rospy.Time) -> PoseStamped:
    message = PoseStamped()
    message.header.stamp = stamp
    message.header.frame_id = "world"
    message.pose.position.z = 1.5
    message.pose.orientation.w = 1.0
    return message


def _cloud(stamp: rospy.Time) -> PointCloud2:
    """Create a bounded public-style obstacle cloud around the vehicle.

    The free center leaves a valid local take-off space.  The surrounding low
    walls give FUEL a nonempty occupancy update and unknown-frontier boundary
    without encoding any benchmark target or private scene geometry.
    """

    points: list[tuple[float, float, float]] = []
    for coordinate in range(-6, 7):
        for height in (0.0, 0.5, 1.0, 1.5, 2.0):
            points.append((-6.0, float(coordinate), height))
            points.append((6.0, float(coordinate), height))
            points.append((float(coordinate), -6.0, height))
            points.append((float(coordinate), 6.0, height))
    header = rospy.Header(stamp=stamp, frame_id="world")
    return point_cloud2.create_cloud_xyz32(header, points)


def _trigger(stamp: rospy.Time) -> Path:
    message = Path()
    message.header.stamp = stamp
    message.header.frame_id = "world"
    pose = PoseStamped()
    pose.header = message.header
    pose.pose.position.z = 1.5
    pose.pose.orientation.w = 1.0
    message.poses.append(pose)
    return message


def _launch_arguments() -> list[str]:
    return [
        "roslaunch",
        "exploration_manager",
        "algorithm.xml",
        "map_size_x_:=20.0",
        "map_size_y_:=20.0",
        "map_size_z_:=5.0",
        "box_min_x:=-9.0",
        "box_min_y:=-9.0",
        "box_min_z:=0.0",
        "box_max_x:=9.0",
        "box_max_y:=9.0",
        "box_max_z:=4.0",
        "odometry_topic:=/aerocity_fuel_smoke/odom",
        "sensor_pose_topic:=/aerocity_fuel_smoke/sensor_pose",
        "depth_topic:=/aerocity_fuel_smoke/depth_unused",
        "cloud_topic:=/aerocity_fuel_smoke/cloud",
        "cx:=160.0",
        "cy:=120.0",
        "fx:=160.0",
        "fy:=160.0",
        "max_vel:=1.5",
        "max_acc:=1.5",
    ]


def run_smoke(duration_s: float) -> dict[str, Any]:
    if duration_s <= TRIGGER_WARMUP_S + 2.0:
        raise ValueError("duration_s must include FUEL's frontier-update warmup and route window")

    os.environ.setdefault("ROS_MASTER_URI", "http://127.0.0.1:11311")
    os.environ.setdefault("ROS_IP", "127.0.0.1")
    os.environ.setdefault("ROS_HOME", "/tmp/aerocity-fuel-ros-home")
    os.environ.setdefault("ROS_LOG_DIR", "/tmp/aerocity-fuel-ros-log")
    master_log = "/tmp/aerocity-fuel-ros-master.log"
    planner_log = "/tmp/aerocity-fuel-ros-planner.log"
    master = _child(["roscore"], master_log)
    planner: subprocess.Popen[bytes] | None = None
    state = SmokeState()
    try:
        deadline = time.monotonic() + 4.0
        while time.monotonic() < deadline:
            if master.poll() is not None:
                raise RuntimeError(
                    "FUEL ROS smoke master exited during startup: " + _log_tail(master_log)
                )
            time.sleep(0.1)

        planner = _child(_launch_arguments(), planner_log)
        rospy.init_node("aerocity_fuel_smoke", anonymous=True, disable_signals=True)
        odom_publisher = rospy.Publisher("/aerocity_fuel_smoke/odom", Odometry, queue_size=4)
        pose_publisher = rospy.Publisher(
            "/aerocity_fuel_smoke/sensor_pose", PoseStamped, queue_size=4
        )
        cloud_publisher = rospy.Publisher(
            "/aerocity_fuel_smoke/cloud", PointCloud2, queue_size=4
        )
        trigger_publisher = rospy.Publisher(
            "/waypoint_generator/waypoints", Path, queue_size=1
        )
        rospy.Subscriber("/planning/bspline", Bspline, state.record_bspline, queue_size=4)

        graph_deadline = time.monotonic() + 5.0
        while time.monotonic() < graph_deadline:
            if planner.poll() is not None:
                raise RuntimeError(
                    "FUEL exploration node exited during graph startup: " + _log_tail(planner_log)
                )
            if (
                odom_publisher.get_num_connections() > 0
                and pose_publisher.get_num_connections() > 0
                and cloud_publisher.get_num_connections() > 0
            ):
                break
            time.sleep(0.1)

        graph_ready = (
            odom_publisher.get_num_connections() > 0
            and pose_publisher.get_num_connections() > 0
            and cloud_publisher.get_num_connections() > 0
        )
        if not graph_ready:
            raise RuntimeError("FUEL exploration node did not subscribe to the public ROS inputs")

        trigger_sent = False
        started_at = time.monotonic()
        end_at = started_at + duration_s
        rate = rospy.Rate(10.0)
        while time.monotonic() < end_at and not rospy.is_shutdown():
            if planner.poll() is not None:
                raise RuntimeError(
                    "FUEL exploration node exited while receiving public inputs: "
                    + _log_tail(planner_log)
                )
            stamp = rospy.Time.now()
            odom_publisher.publish(_odometry(stamp))
            pose_publisher.publish(_pose(stamp))
            cloud_publisher.publish(_cloud(stamp))
            # FUEL's frontier timer skips its first five 0.5-second callbacks.
            # Triggering sooner makes its FSM enter FINISH before the online map
            # yields a candidate frontier, even though inputs are correctly wired.
            if not trigger_sent and time.monotonic() - started_at >= TRIGGER_WARMUP_S:
                trigger_publisher.publish(_trigger(stamp))
                trigger_sent = True
            rate.sleep()

        return {
            "schema": "org.aerocity.bench.fuel-ros-smoke.v1",
            "status": "ROUTE_EMITTED" if state.bspline_message_count else "GRAPH_READY_NO_ROUTE",
            "duration_s": duration_s,
            "trigger_warmup_s": TRIGGER_WARMUP_S,
            "public_input_topics": [
                "/aerocity_fuel_smoke/odom",
                "/aerocity_fuel_smoke/sensor_pose",
                "/aerocity_fuel_smoke/cloud",
                "/waypoint_generator/waypoints",
            ],
            "planner_output_topic": "/planning/bspline",
            "position_command_topics_started": False,
            "graph_ready": graph_ready,
            "route_trigger_sent": trigger_sent,
            "bspline_message_count": state.bspline_message_count,
            "first_bspline_control_point_count": state.first_bspline_control_point_count,
            "first_bspline_order": state.first_bspline_order,
            "master_log_tail": _log_tail(master_log),
            "planner_log_tail": _log_tail(planner_log),
            "target_truth_exposed": False,
            "benchmark_score_claimed": False,
        }
    finally:
        _stop(planner)
        _stop(master)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-s", type=float, default=12.0)
    arguments = parser.parse_args(argv)
    try:
        result = run_smoke(arguments.duration_s)
    except (OSError, RuntimeError, ValueError, rospy.ROSException) as exc:
        print(
            json.dumps(
                {
                    "schema": "org.aerocity.bench.fuel-ros-smoke.v1",
                    "status": "SMOKE_FAILED",
                    "reason": str(exc),
                    "target_truth_exposed": False,
                    "benchmark_score_claimed": False,
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ROUTE_EMITTED" else 3


if __name__ == "__main__":
    raise SystemExit(main())
