import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class TemplateTests(unittest.TestCase):
    def test_driver_template_is_valid_and_contains_no_real_network_value(self):
        path = ROOT / "config" / "MID360_config.template.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        host = data["MID360"]["host_net_info"]
        lidar = data["lidar_configs"][0]
        self.assertIn("REPLACE_WITH_TARGET_COMPUTER_NIC_IP", host["cmd_data_ip"])
        self.assertIn("REPLACE_WITH_MID360_IP", lidar["ip"])
        self.assertEqual(lidar["pcl_data_type"], 1)

    def test_contract_yaml_keeps_static_tf_disabled(self):
        text = (ROOT / "config" / "competition.yaml").read_text(encoding="utf-8")
        self.assertIn("enabled: false", text)
        self.assertIn("calibration_ready: false", text)
        self.assertIn("max_pose_silence_sec: 0.30", text)


if __name__ == "__main__":
    unittest.main()
