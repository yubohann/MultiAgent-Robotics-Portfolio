"""Static validation for the checked-in Gazebo simulation profile."""

import csv
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT / "scripts"))

from run_contract import StreamReadiness


class SimulationAssetTests(unittest.TestCase):
    def test_motion_readiness_requires_valid_packets_from_both_streams(self):
        readiness = StreamReadiness(required_lidar_packets=3, required_imu_packets=2)
        readiness.observe_lidar(False)
        readiness.observe_lidar(True)
        readiness.observe_lidar(True)
        readiness.observe_imu(True)
        self.assertFalse(readiness.ready)

        readiness.observe_lidar(True)
        self.assertFalse(readiness.ready)
        readiness.observe_imu(True)
        self.assertTrue(readiness.ready)
        self.assertEqual(readiness.snapshot()["accepted_lidar_packets"], 3)

    def test_motion_readiness_rejects_nonpositive_thresholds(self):
        with self.assertRaises(ValueError):
            StreamReadiness(required_lidar_packets=0, required_imu_packets=1)
        with self.assertRaises(ValueError):
            StreamReadiness(required_lidar_packets=1, required_imu_packets=0)

    def test_mapping_profile_publishes_registered_cloud_without_changing_smoke_profile(self):
        mapping = (PACKAGE_ROOT / "config" / "fast_lio_mapping_simulation.yaml").read_text(encoding="utf-8")
        smoke = (PACKAGE_ROOT / "config" / "fast_lio_simulation.yaml").read_text(encoding="utf-8")
        self.assertIn("scan_publish_en: true", mapping)
        self.assertIn("dense_publish_en: true", mapping)
        self.assertIn("scan_publish_en: false", smoke)

    def test_sparse_coverage_mapping_is_explicitly_diagnostic(self):
        launch = (PACKAGE_ROOT / "launch" / "gazebo_mid360_mapping.launch.py").read_text(encoding="utf-8")
        runner = (PACKAGE_ROOT / "scripts" / "run_simulation_mapping.sh").read_text(encoding="utf-8")
        self.assertIn('"sparse_coverage"', launch)
        self.assertIn('radius_filter": 0.0', launch)
        self.assertIn('"z_min": -0.6', launch)
        self.assertIn('"z_max": 4.5', launch)
        self.assertIn("SPARSE_COVERAGE", runner)
        self.assertIn("mapping-sparse-coverage-diagnostic", runner)

    def test_map_manifest_records_promotion_profile_and_density_contract(self):
        runner = (PACKAGE_ROOT / "scripts" / "run_simulation_mapping.sh").read_text(encoding="utf-8")
        self.assertIn('"lidar_samples_requested": requested_rays', runner)
        self.assertIn('"lidar_downsample": downsample', runner)
        self.assertIn('"scene": scene', runner)
        self.assertIn('requested_rays >= 30000', runner)
        self.assertIn('downsample == 1', runner)
        self.assertIn('scene == "indoor_competition_candidate"', runner)
        self.assertIn('"eligible_for_fixed_map": eligible', runner)

    def test_one_command_dispatcher_does_not_mix_failed_mapping_with_archived_map(self):
        dispatcher = (PACKAGE_ROOT.parent.parent / "tools" / "run_experiments.sh").read_text(encoding="utf-8")
        self.assertIn('if [[ "$mode" == "all" || "$mode" == "quality" ]]', dispatcher)
        self.assertIn('selected_map="$run_root/02_mapping_dense/frozen_map.pcd"', dispatcher)
        self.assertIn('! map_is_eligible "$selected_map"', dispatcher)
        self.assertIn('run_rgbd || failed=1', dispatcher)
        self.assertIn('MIN_FREE_SPACE_GB', dispatcher)
        self.assertIn('pcd_sha256 == actual_sha256', dispatcher)
        self.assertIn('[[ "$dry_run" == true ]] || require_free_space', dispatcher)
        self.assertIn('[[ "$dry_run" == false ]] && ! map_is_eligible', dispatcher)

    def test_fixed_map_interlock_publishes_expiring_peer_heartbeat(self):
        driver = (PACKAGE_ROOT / "scripts" / "fixed_map_interlock_integration.py").read_text(encoding="utf-8")
        self.assertIn('"/robocon/team/message"', driver)
        self.assertIn('"message_type": "heartbeat"', driver)
        self.assertIn('"task_id": "gazebo-fixed-map-interlock"', driver)
        self.assertIn('"expires_at_ns"', driver)
        self.assertIn('Parameter("use_sim_time", Parameter.Type.BOOL, False)', driver)

    def test_fixed_map_smoke_keeps_map_provenance_and_isolation(self):
        runner = (PACKAGE_ROOT / "scripts" / "run_fixed_map_localization_smoke.sh").read_text(encoding="utf-8")
        probe = (PACKAGE_ROOT / "scripts" / "fixed_map_localization_smoke.py").read_text(encoding="utf-8")
        self.assertIn("ROS_DOMAIN_ID", runner)
        self.assertIn("map_input.sha256", runner)
        self.assertIn("/cloud_registered", runner)
        self.assertIn("map_locked_seen", probe)
        self.assertIn("gazebo_simulation", probe)

    def test_mid360_scan_pattern_supports_monotonic_uint32_offsets(self):
        scan_file = PACKAGE_ROOT / "scan_mode" / "mid360.csv"
        with scan_file.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            sample_ticks = [float(next(reader)["Time/s"]) for _ in range(30_000)]
        relative_ns = [round((value - min(sample_ticks)) * 1.0e3) for value in sorted(sample_ticks)]
        self.assertEqual(relative_ns, sorted(relative_ns))
        self.assertGreater(relative_ns[-1], 0)
        self.assertLessEqual(relative_ns[-1], 2**32 - 1)
        self.assertLess(relative_ns[-1], 100_000_000)

    def test_mid360_scan_pattern_wrap_preserves_ray_order_and_frame_duration(self):
        pattern_period_ticks = 800_000
        ray_order_ticks = [799_990, 799_999, 800_000, 1, 2, 10]
        relative_ns = [
            ((tick - ray_order_ticks[0]) % pattern_period_ticks) * 1_000
            for tick in ray_order_ticks
        ]
        self.assertEqual(relative_ns, sorted(relative_ns))
        self.assertLess(relative_ns[-1], 30_000_000)

    def test_public_field_asset_retains_rulebook_and_license_provenance(self):
        world = (PACKAGE_ROOT / "worlds" / "robocon25_candidate.world").read_text(encoding="utf-8")
        notice = (PACKAGE_ROOT / "LICENSE_UPSTREAM_FIELD_ASSET.md").read_text(encoding="utf-8")
        self.assertIn("RoboCon25_Field", world)
        self.assertIn("fynngwu/gazebo_simulation", notice)
        self.assertIn("MIT License", notice)
        self.assertIn("rulebook_geometry_subset_verified", notice)
        self.assertIn("mid360_simulation_return_plane", world)
        self.assertIn("official ROBOCON field geometry", world)

    def test_rulebook_geometry_subset_matches_verified_dimensions(self):
        field_model = PACKAGE_ROOT / "models" / "RoboCon25_Field" / "model.sdf"
        root = ET.parse(field_model).getroot()
        rings = root.findall(".//collision[@name='collision_hoop']/pose")
        self.assertEqual(len(rings), 2)
        self.assertEqual([ring.text.split()[2] for ring in rings], ["2.43", "2.43"])
        boards = root.findall(".//collision[@name='collision_backboard']/geometry/box/size")
        self.assertEqual(len(boards), 2)
        self.assertEqual([board.text.split() for board in boards], [["0.02", "1.8", "1.05"]] * 2)
        self.assertIn("<size>15 0.05 0.1</size>", field_model.read_text(encoding="utf-8"))
        self.assertIn("<size>22.5 11.5</size>", field_model.read_text(encoding="utf-8"))
        ball_world = (PACKAGE_ROOT / "worlds" / "robocon25_candidate.world").read_text(encoding="utf-8")
        self.assertIn("<mass>0.6</mass>", ball_world)
        self.assertIn("<radius>0.12</radius>", ball_world)

    def test_plugin_preserves_custommsg_time_contract(self):
        source = (PACKAGE_ROOT / "src" / "livox_points_plugin.cpp").read_text(encoding="utf-8")
        self.assertIn("pp_livox.timebase", source)
        self.assertIn("point.offset_time", source)
        self.assertIn("PatternOffsetNanoseconds", source)
        self.assertIn("scanPatternPeriodTicks", source)
        self.assertNotIn("std::sort(valid_points.begin(), valid_points.end()", source)
        self.assertNotIn("boost::chrono", source)

    def test_plugin_caches_pattern_directions_and_skips_unused_cloud_mirror(self):
        source = (PACKAGE_ROOT / "src" / "livox_points_plugin.cpp").read_text(encoding="utf-8")
        header = (PACKAGE_ROOT / "include" / "robocon_mid360_simulation" / "livox_points_plugin.h").read_text(
            encoding="utf-8"
        )
        self.assertIn("ignition::math::Vector3d direction", header)
        self.assertIn("rotate_info.direction", source)
        self.assertIn("cloud2_pub->get_subscription_count() > 0", source)
        self.assertIn("if (publish_cloud_mirror)", source)

    def test_plugin_limits_each_packet_to_one_sensor_update_window(self):
        source = (PACKAGE_ROOT / "src" / "livox_points_plugin.cpp").read_text(encoding="utf-8")
        self.assertIn("PacketPatternSamples(scanPeriodSeconds, maxPointSize)", source)
        self.assertIn("PatternStartIndex(", source)
        self.assertIn("PatternIndexForRay(", source)
        self.assertIn("requested_ray_count > packet_pattern_samples", source)
        self.assertNotIn("currStartIndex += samplesStep", source)

    def test_stream_probe_requires_non_overlapping_livox_packets(self):
        probe = (PACKAGE_ROOT / "scripts" / "probe_simulation_streams.py").read_text(encoding="utf-8")
        self.assertIn('"offset_time_monotonic"', probe)
        self.assertIn('"packets_non_overlapping"', probe)
        self.assertIn("timebase >= self.previous_lidar_end_time", probe)
        self.assertIn('parser.add_argument("--lidar-topic"', probe)

    def test_launch_exposes_bounded_wsl_and_upstream_density_profiles(self):
        launch = (PACKAGE_ROOT / "launch" / "gazebo_mid360_candidate.launch.py").read_text(encoding="utf-8")
        robot = (PACKAGE_ROOT / "urdf" / "robocon25_mid360_robot.xacro").read_text(encoding="utf-8")
        self.assertIn('default_value="2000"', launch)
        self.assertIn("lidar_samples", launch)
        self.assertIn('default="2000"', robot)
        self.assertIn("samples=\"$(arg lidar_samples)\"", robot)
        self.assertIn('name="enable_rgbd" default="false"', robot)
        self.assertIn('DeclareLaunchArgument(\n            "enable_rgbd"', launch)

    def test_optional_rgbd_interface_is_explicit_and_disabled_by_default(self):
        robot = (PACKAGE_ROOT / "urdf" / "robocon25_mid360_robot.xacro").read_text(encoding="utf-8")
        self.assertIn('<xacro:if value="$(arg enable_rgbd)">', robot)
        self.assertIn('name="camera_rgbd_link"', robot)
        self.assertIn('sensor name="simulated_rgbd_camera" type="depth"', robot)
        self.assertIn('filename="libgazebo_ros_camera.so"', robot)
        self.assertIn('/camera/depth/image_raw', robot)

    def test_ground_truth_interface_is_explicit_and_disabled_by_default(self):
        launch = (PACKAGE_ROOT / "launch" / "gazebo_mid360_candidate.launch.py").read_text(encoding="utf-8")
        robot = (PACKAGE_ROOT / "urdf" / "robocon25_mid360_robot.xacro").read_text(encoding="utf-8")
        self.assertIn('name="enable_ground_truth" default="false"', robot)
        self.assertIn('<xacro:if value="$(arg enable_ground_truth)">', robot)
        self.assertIn('filename="libgazebo_ros_p3d.so"', robot)
        self.assertIn('ground_truth/odom', robot)
        self.assertIn('"enable_ground_truth"', launch)

    def test_rgbd_quality_and_systemd_harnesses_are_bounded(self):
        probe = (PACKAGE_ROOT.parent.parent / "tools" / "rgbd_quality_probe.py").read_text(encoding="utf-8")
        rgbd_runner = (PACKAGE_ROOT.parent.parent / "tools" / "run_rgbd_quality_smoke.sh").read_text(encoding="utf-8")
        systemd_runner = (PACKAGE_ROOT.parent.parent / "tools" / "test_systemd_simulation.sh").read_text(encoding="utf-8")
        self.assertIn("depth_valid_ratio", probe)
        self.assertIn("max-wall-sec", probe)
        self.assertIn("RGBD_MAX_WALL_SEC", rgbd_runner)
        self.assertIn("TimeoutStopSec=2s", systemd_runner)

    def test_base_ray_sensor_is_only_a_low_cost_update_trigger(self):
        sensor = (PACKAGE_ROOT / "urdf" / "mid360.xacro").read_text(encoding="utf-8")
        self.assertIn("custom\n                Livox collision shape", sensor)
        self.assertIn("<always_on>true</always_on>", sensor)
        self.assertEqual(sensor.count("<samples>1</samples>"), 2)

    def test_indoor_profile_uses_single_field_asset_with_two_internal_hoops(self):
        world = (PACKAGE_ROOT / "worlds" / "indoor_competition_candidate.world").read_text(encoding="utf-8")
        self.assertIn("model://RoboCon25_Field", world)
        self.assertNotIn("<name>indoor_hoop_east</name>", world)
        self.assertNotIn("<name>indoor_hoop_west</name>", world)
        self.assertNotIn("indoor_north_wall", world)
        self.assertNotIn("sideline_rail_north", world)
        self.assertIn("venue_north_wall", world)
        self.assertIn("venue_south_wall", world)
        field = (PACKAGE_ROOT / "models" / "RoboCon25_Field" / "model.sdf").read_text(encoding="utf-8")
        self.assertEqual(field.count('<model name="basketball_hoop'), 2)


if __name__ == "__main__":
    unittest.main()
