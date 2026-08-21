from glob import glob
from setuptools import setup


package_name = "mid360_localization_contract"


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/docs", ["README.md", "INTERFACE_CONTRACT.md"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        (
            "share/" + package_name + "/config",
            glob("config/*.yaml") + glob("config/*.template.json"),
        ),
        ("share/" + package_name + "/systemd", glob("systemd/*.service")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ROBOCON2025 team",
    maintainer_email="replace-before-deployment@example.invalid",
    description="ROS 2 interface, validation, and localization-anchor nodes for a Livox MID-360 deployment.",
    license="Proprietary",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mid360_input_guard = mid360_localization_contract.input_guard:main",
            "mid360_pose_bridge = mid360_localization_contract.pose_bridge:main",
            "mid360_map_odom_anchor = mid360_localization_contract.map_odom_anchor:main",
            "mid360_static_sensor_frames = mid360_localization_contract.static_sensor_frames:main",
            "mid360_preflight = mid360_localization_contract.preflight:main",
        ],
    },
)
