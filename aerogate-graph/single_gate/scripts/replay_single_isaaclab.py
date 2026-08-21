"""Render a saved single-agent replay inside IsaacLab with real scene assets."""

from __future__ import annotations

import argparse
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

    parser = argparse.ArgumentParser(description="Replay one single-agent aerogate_graph trajectory in IsaacLab.")
    parser.add_argument("--trajectory", type=str, required=True)
    parser.add_argument("--report", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--mp4-path", type=str, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--hold-initial-frames", type=int, default=18)
    parser.add_argument("--hold-final-frames", type=int, default=12)
    parser.add_argument(
        "--camera-mode",
        type=str,
        default="picture_in_picture",
        choices=("global", "follow", "picture_in_picture"),
    )
    parser.add_argument("--enable-follow-overlay", action="store_true", default=False)
    parser.add_argument("--disable-follow-overlay", action="store_true", default=False)
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

    ensure_writable_kit_runtime(args_cli, app_name="aerogate_graph_single_replay")
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    exit_code = 0
    try:
        from single_gate.isaac_replay import render_single_trajectory_isaaclab

        print(
            f"[INFO] Starting single-agent IsaacLab replay: camera_mode={args_cli.camera_mode}, "
            f"output_dir={args_cli.output_dir or '<trajectory_dir>'}",
            flush=True,
        )
        summary = render_single_trajectory_isaaclab(
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
            overlay_follow_view=bool(args_cli.enable_follow_overlay) and not bool(args_cli.disable_follow_overlay),
        )
        print("single-agent IsaacLab replay complete", flush=True)
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
            label="single_replay.simulation_app",
            wait_for_replicator=args_cli.mp4_path is None,
        )

    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

