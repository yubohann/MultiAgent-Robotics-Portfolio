# Third-Party Notices

This project is packaged as an original ROS2 integration and migration project. It depends on standard ROS2 ecosystem packages and documents the historical references used during migration.

Runtime dependencies expected from the ROS2 distribution.

- ROS2 Jazzy core packages
- Nav2
- slam_toolbox
- robot_state_publisher
- xacro
- OpenCV
- cv_bridge
- sensor_msgs
- geometry_msgs
- std_srvs
- robot_localization

Historical ROS1 baseline packages reviewed during migration.

- Slamtec `rplidar_ros`
- WaterPlus `waterplus_map_tools`
- ROS1 `move_base`, `gmapping`, `amcl`
- ROS1 `apriltag_ros`

Robot geometry reference.

- `description/meshes/zoo/base_link.stl`
- `description/meshes/zoo/laser_link.stl`

These meshes are adapted from the local Zoo robot description package included with the project material. They serve as CAD reference assets, with collision geometry, sensor frames, ROS2 package interfaces and runtime behavior maintained in this portfolio workspace.

The ROS2 workspace references those ROS1 repositories externally. For hardware drivers on a specific robot, install the vendor-provided ROS2 driver separately and document the exact version used.
