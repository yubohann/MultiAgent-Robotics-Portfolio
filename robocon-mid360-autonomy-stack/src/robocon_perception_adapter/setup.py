from glob import glob
from setuptools import setup


package_name = "robocon_perception_adapter"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ROBOCON2025 team",
    maintainer_email="replace-before-deployment@example.invalid",
    description="Evidence-gated perception contract for the ROBOCON competition stack.",
    license="Proprietary",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "target_gate = robocon_perception_adapter.target_gate:main",
            "synthetic_target_source = robocon_perception_adapter.synthetic_target_source:main",
        ],
    },
)
