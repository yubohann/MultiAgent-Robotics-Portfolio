# LiDAR-Inertial Localization and Shooting-Parameter Integration for ROBOCON Robot Basketball

**LiDAR and Perception Lead** | **Jun. 2024 - Oct. 2025**  
**Methods & Systems:** ROS 2 Humble; C++17; Python; Linux; FAST-LIO2; Livox MID-360 with built-in IMU; PCL; Eigen; TF2; Nav2; Gazebo; YOLO; RGB-D; UART; ESP-NOW; STM32; CAN.

- Integrated Livox MID-360 and built-in IMU with FAST-LIO2 on physical competition robots subject to rapid acceleration, chassis vibration, mechanism-induced disturbance, and partial field occlusion. Calibrated sensor installation and timing, then established mapping and fixed-map localization workflows for navigation and shooting.
- Unified LiDAR odometry, fixed-map localization, and navigation through ROS 2 pose interfaces with freshness, map-state, and jump checks, providing consistent pose status for target alignment and chassis control.
- Built a pose-aware shooting interface that computes range and heading error from robot pose and hoop coordinates, maps range to launch-speed parameters, and transmits commands through UART to STM32 shooting and chassis-control hardware.
- Developed an offline ball-trajectory model with gravity, aerodynamic drag, and Magnus-force terms. Used simulated annealing to search initial launch velocities and supplied the physics-based result as a calibration reference for the online range-to-speed mapping.
- Integrated YOLO and RGB-D target detection into ROS 2, normalized confidence, depth, stability, and timestamp fields, and gated shooting on target validity and observation freshness. Connected ESP-NOW position synchronization to the dual-robot task sequence with acknowledgements, heartbeats, timeout recovery, and emergency-stop handling.
- Built Gazebo MID-360 simulation with Livox scan timing and IMU inputs, reusing ROS 2 interfaces before transfer to physical robots.
- Led field integration across LiDAR localization, perception, navigation, communication, UART, and STM32 control, contributing to National Second Prizes in both Robot Basketball and Robot Basketball Shooting at the 24th China University Robot Competition (ROBOCON).
