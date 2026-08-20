from glob import glob
from setuptools import setup


package_name = "robocon_game_supervisor"


setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/docs", ["README.md"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ROBOCON2025 team",
    maintainer_email="replace-before-deployment@example.invalid",
    description="Deterministic ROBOCON task, safety-gate, and dual-robot protocol core.",
    license="Proprietary",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "robocon_game_supervisor = robocon_game_supervisor.ros_node:main",
            "robocon_action_simulator = robocon_game_supervisor.simulator:main",
            "robocon_synthetic_inputs = robocon_game_supervisor.synthetic_inputs:main",
            "robocon_run_manifest = robocon_game_supervisor.run_manifest:main",
            "robocon_rulebook_scenario = robocon_game_supervisor.rulebook_scenario:main",
        ],
    },
)
