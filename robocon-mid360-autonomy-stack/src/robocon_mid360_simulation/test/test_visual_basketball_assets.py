import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class VisualBasketballAssetTests(unittest.TestCase):
    def test_depth_visualizer_and_dual_launch_are_wired(self):
        launch = (PACKAGE_ROOT / "launch" / "gazebo_mid360_dual_mapping.launch.py").read_text(encoding="utf-8")
        cmake = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        visualizer = (PACKAGE_ROOT / "scripts" / "depth_visualizer.py").read_text(encoding="utf-8")
        self.assertIn('"publish.tf": True', launch)
        self.assertIn('"use_odometry": True', launch)
        self.assertIn('"log_odds_decay_per_cloud": 0', launch)
        self.assertEqual(launch.count('executable="depth_visualizer.py"'), 2)
        self.assertIn('"/robot1/simulated_rgbd_camera/depth/image_visualized"', launch)
        self.assertIn('"/robot2/simulated_rgbd_camera/depth/image_visualized"', launch)
        self.assertIn("visualized.encoding = \"mono8\"", visualizer)
        self.assertIn("scripts/depth_visualizer.py", cmake)

    def test_both_rgbd_configs_use_their_robot_namespace(self):
        robot1 = (PACKAGE_ROOT / "config" / "rviz_basketball_rgbd.rviz").read_text(encoding="utf-8")
        robot2 = (PACKAGE_ROOT / "config" / "rviz_basketball_rgbd_robot2.rviz").read_text(encoding="utf-8")
        self.assertIn("/robot1/simulated_rgbd_camera/image_raw", robot1)
        self.assertIn("/robot1/simulated_rgbd_camera/depth/image_visualized", robot1)
        self.assertIn("/robot2/simulated_rgbd_camera/image_raw", robot2)
        self.assertIn("/robot2/simulated_rgbd_camera/depth/image_visualized", robot2)
        self.assertEqual(robot1.count("Class: rviz_default_plugins/Image"), 2)
        self.assertEqual(robot2.count("Class: rviz_default_plugins/Image"), 2)

    def test_controller_uses_staging_and_ballistic_trajectory(self):
        controller = (PACKAGE_ROOT / "scripts" / "basketball_demo_controller.py").read_text(encoding="utf-8")
        self.assertIn("self.shot_staging_target = (4.90, -0.55)", controller)
        self.assertIn("self.robot1_dribble_target = (robot1_start[0] + 2.40", controller)
        self.assertIn("def _drive_to_xy(", controller)
        self.assertIn("gravity = 9.81", controller)
        self.assertIn("ballistic_shot_started", controller)
        self.assertNotIn("2.01 * math.sin(math.pi * smooth)", controller)

    def test_quality_recorder_defaults_to_full_packet_density(self):
        runner = (PACKAGE_ROOT.parent.parent / "tools" / "run_robocon_basketball_real_gui.sh").read_text(encoding="utf-8")
        self.assertIn('WORKSPACE="${WORKSPACE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"', runner)
        self.assertNotIn("C:/Users/Administrator/AppData", runner)
        self.assertIn("LIDAR_SAMPLES=${2:-50000}", runner)
        self.assertIn("LIDAR_DOWNSAMPLE=${5:-1}", runner)

    def test_image_only_rgbd_viewer_has_no_empty_rviz_render_panel(self):
        viewer = (PACKAGE_ROOT.parent.parent / "tools" / "run_rgbd_image_view.sh").read_text(encoding="utf-8")
        self.assertIn("rqt_image_view", viewer)
        self.assertIn("robot1:depth", viewer)
        self.assertIn("robot2:depth", viewer)
        self.assertIn("depth/image_visualized", viewer)


if __name__ == "__main__":
    unittest.main()
