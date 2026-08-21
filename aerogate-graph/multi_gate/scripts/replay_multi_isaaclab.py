"""Render a saved multi-agent replay inside IsaacLab with real scene assets."""

from __future__ import annotations

import argparse
import json
import sys
import traceback


def _bootstrap_imports() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def main() -> None:
    _bootstrap_imports()
    from shared.visualization.scene_isaaclab import ensure_project_and_source_paths

    ensure_project_and_source_paths()

    from scripts.rl.kit_runtime_utils import ensure_writable_kit_runtime
    from scripts.rl.sim_shutdown_utils import close_simulation_app_with_timeout

    _app_import_error = None
    try:
        from isaaclab.app import AppLauncher
    except ModuleNotFoundError as exc:
        AppLauncher = None
        _app_import_error = exc

    parser = argparse.ArgumentParser(description="Replay one multi-agent aerogate_graph trajectory in IsaacLab.")
    parser.add_argument("--trajectory", type=str, required=True)
    parser.add_argument("--report", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--mp4-path", type=str, default=None)
    parser.add_argument("--config-name", type=str, default="variable")
    parser.add_argument("--scene-mode", type=str, default=None)
    parser.add_argument("--render-real-gate", type=int, default=None)
    parser.add_argument("--render-real-drone-shell", type=int, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--hold-initial-frames", type=int, default=18)
    parser.add_argument("--hold-final-frames", type=int, default=12)
    parser.add_argument(
        "--camera-mode",
        type=str,
        default="picture_in_picture",
        choices=("global", "follow", "picture_in_picture", "height_audit", "top_global", "top_centroid_follow"),
    )
    parser.add_argument("--follow-agent-index", type=int, default=None)
    parser.add_argument(
        "--route-waypoints-json",
        type=str,
        default=None,
        help="Optional JSON list of route waypoint [x, y] pairs to render as IsaacSim beacons.",
    )
    parser.add_argument(
        "--route-waypoint-names-json",
        type=str,
        default=None,
        help="Optional JSON list of waypoint names matching --route-waypoints-json.",
    )
    parser.add_argument("--real-time", action="store_true", default=False)
    if AppLauncher is not None:
        AppLauncher.add_app_launcher_args(parser)
    if AppLauncher is None:
        args_cli, _ignored_unknown = parser.parse_known_args()
    else:
        args_cli = parser.parse_args()

    if args_cli.mp4_path is not None:
        args_cli.enable_cameras = True

    if AppLauncher is None:
        missing = getattr(_app_import_error, "name", "isaacsim")
        parser.error(
            f"Failed to import IsaacLab AppLauncher because '{missing}' is unavailable. "
            "Activate the Isaac Sim / IsaacLab Python environment to run this script."
        )

    ensure_writable_kit_runtime(args_cli, app_name="aerogate_graph_multi_replay")
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    exit_code = 0
    try:
        from multi_gate.configs import get_multi_experiment_config, override_multi_scene_config
        from multi_gate.isaac_replay import render_multi_trajectory_isaaclab

        experiment_config = get_multi_experiment_config(args_cli.config_name)
        if (
            args_cli.scene_mode is not None
            or args_cli.render_real_gate is not None
            or args_cli.render_real_drone_shell is not None
        ):
            experiment_config = override_multi_scene_config(
                experiment_config,
                scene_mode=args_cli.scene_mode,
                render_real_gate=(
                    None if args_cli.render_real_gate is None else bool(int(args_cli.render_real_gate))
                ),
                render_real_drone_shell=(
                    None
                    if args_cli.render_real_drone_shell is None
                    else bool(int(args_cli.render_real_drone_shell))
                ),
            )
        print(
            f"[INFO] Starting multi-agent IsaacLab replay: camera_mode={args_cli.camera_mode}, "
            f"output_dir={args_cli.output_dir or '<trajectory_dir>'}",
            flush=True,
        )
        summary = render_multi_trajectory_isaaclab(
            trajectory_path=args_cli.trajectory,
            report_path=args_cli.report,
            output_dir=args_cli.output_dir,
            mp4_path=args_cli.mp4_path,
            fps=args_cli.fps,
            resolution=(args_cli.width, args_cli.height),
            hold_initial_frames=args_cli.hold_initial_frames,
            hold_final_frames=args_cli.hold_final_frames,
            real_time=args_cli.real_time,
            camera_mode=args_cli.camera_mode,
            follow_agent_index=args_cli.follow_agent_index,
            route_waypoints_xy=(
                None
                if args_cli.route_waypoints_json is None
                else json.loads(args_cli.route_waypoints_json)
            ),
            route_waypoint_names=(
                None
                if args_cli.route_waypoint_names_json is None
                else json.loads(args_cli.route_waypoint_names_json)
            ),
            experiment_config=experiment_config,
        )
        print("multi-agent IsaacLab replay complete", flush=True)
        for key, value in summary.items():
            print(f"{key}={value}", flush=True)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        exit_code = 1
    finally:
        close_simulation_app_with_timeout(
            simulation_app,
            timeout_s=30.0 if args_cli.mp4_path else 10.0,
            label="multi_replay.simulation_app",
            wait_for_replicator=args_cli.mp4_path is None,
        )

    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

