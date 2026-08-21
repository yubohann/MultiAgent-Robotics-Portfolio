from setuptools import setup

package_name = "robocon_pose_command_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "README.md", "UPSTREAM_NOTICE.md"]),
        (f"share/{package_name}/config", ["config/pose_command_bridge.yaml"]),
        (f"share/{package_name}/launch", ["launch/pose_command_bridge.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "pose_command_bridge = robocon_pose_command_bridge.pose_command_bridge:main",
        ],
    },
)
