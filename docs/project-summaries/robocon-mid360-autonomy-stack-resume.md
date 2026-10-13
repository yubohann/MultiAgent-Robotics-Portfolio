# LiDAR-Inertial Localization and Competition Autonomy for ROBOCON Robot Basketball

**LiDAR and Perception Lead** | **Jun. 2024 - Oct. 2025**  
**Methods & Systems:** ROS 2; C++; Python; FAST-LIO2; Livox MID-360 with built-in IMU; TF2; Nav2; Gazebo; YOLO/RGB-D perception; UART; ESP-NOW; STM32; CAN; Linux.

- Deployed LiDAR-inertial localization on physical ROBOCON basketball robots, integrating Livox MID-360 and built-in IMU data with FAST-LIO2 and fixed-map localization to support navigation, target alignment, and shooting under rapid motion and chassis vibration.
- Built the pose-to-shot interface that converted robot pose and basket coordinates into distance, heading-error, and launch-speed parameters, then transmitted commands through UART to STM32 shooting and chassis-control firmware.
- Connected YOLO/RGB-D target observations to ROS 2 action logic with confidence, depth, stability, and freshness gates; integrated a rule-aware dual-robot supervisor with acknowledgements, heartbeats, action expiry, recovery, and emergency-stop handling.
- Developed Gazebo MID-360 profiles that exercised the same Livox timing, IMU, TF, localization, perception, and control contracts used for the simulation-to-physical-robot transfer.
- Led field integration across localization, perception, navigation, communication, and embedded control, contributing to National Second Prizes in both the Robot Basketball Competition and the Robot Basketball Shooting Competition at the 24th China University Robot Competition (ROBOCON).
