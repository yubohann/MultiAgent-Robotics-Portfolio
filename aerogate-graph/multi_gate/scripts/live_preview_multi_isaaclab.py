"""Live IsaacLab preview sidecar for Experiment 3 training snapshots."""

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

    parser = argparse.ArgumentParser(description="Show live Experiment 3 training snapshots in IsaacLab.")
    parser.add_argument("--snapshot", type=str, required=True)
    parser.add_argument("--config-name", type=str, required=True)
    parser.add_argument("--scene-mode", type=str, default=None)
    parser.add_argument("--render-real-gate", type=int, default=None)
    parser.add_argument("--render-real-drone-shell", type=int, default=None)
    parser.add_argument("--poll-interval", type=float, default=0.1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--camera-mode",
        type=str,
        default="picture_in_picture",
        choices=("global", "follow", "picture_in_picture"),
    )
    parser.add_argument("--follow-agent-index", type=int, default=0)
    if AppLauncher is not None:
        AppLauncher.add_app_launcher_args(parser)
    if AppLauncher is None:
        args_cli, _ignored_unknown = parser.parse_known_args()
    else:
        args_cli = parser.parse_args()

    if AppLauncher is None:
        missing = getattr(_app_import_error, "name", "isaacsim")
        parser.error(
            f"Failed to import IsaacLab AppLauncher because '{missing}' is unavailable. "
            "Activate the Isaac Sim / IsaacLab Python environment to run this live preview."
        )

    ensure_writable_kit_runtime(args_cli, app_name="aerogate_graph_multi_live_preview")
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    exit_code = 0
    try:
        from multi_gate.configs import get_multi_experiment_config, override_multi_scene_config
        from multi_gate.live_preview import run_live_snapshot_preview
        import isaaclab.sim as sim_utils

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
        sim = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(dt=max(float(args_cli.poll_interval), 0.02), render_interval=1)
        )
        run_live_snapshot_preview(
            snapshot_path=args_cli.snapshot,
            experiment_config=experiment_config,
            sim=sim,
            poll_interval_s=args_cli.poll_interval,
            camera_mode=args_cli.camera_mode,
            follow_agent_index=args_cli.follow_agent_index,
        )
    except KeyboardInterrupt:
        pass
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        exit_code = 1
    finally:
        close_simulation_app_with_timeout(
            simulation_app,
            timeout_s=10.0,
            label="multi_live_preview.simulation_app",
            wait_for_replicator=False,
        )

    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

