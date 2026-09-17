# Competition Rules Summary

This project targets the 2025 China Robot Competition and RoboCup China visual challenge rule set.

## Field

- The arena measures 3m x 3m.
- Outer and inner fences stand 0.5m high.
- Start zones span 0.5m x 0.5m in blue and yellow.
- Base zones span 0.5m x 0.5m in blue and yellow.
- Two obstacles measure 0.3m x 0.3m x 0.3m each.
- Targets include eight normal targets plus one base target in each base.
- Each base target is protected by four armor plates.

## Visual Targets

- AprilTag family Tag36h11.
- Normal target ID 1.
- Yellow base target ID 2.
- Blue base target ID 3.
- Tag size 5cm x 5cm.
- Tag bottom height spans 6.5cm to 7.5cm from the floor.
- Base targets are placed with a 45-degree tilt.

## Elimination Match Logic

- Two robots start from their own start zones.
- Each robot must search for and shoot opponent targets.
- The project safety gate rejects own-target shots.
- A normal target hit removes one opponent base armor plate in order and awards 5 points.
- After any normal target hit, the robot may keep clearing normal targets or directly attempt the opponent base target.
- The first robot to knock down the opponent base target wins.
- Accidentally knocking down its own base target loses the match.
- If time expires at 3 minutes with both base targets standing, the higher score wins.
- If a robot collision knocks down a target, the score is awarded to the non-contacting side according to the target value.

## Sensors And Shooter Constraints

- Allowed sensing includes lidar, depth camera, monocular camera, ultrasonic and ToF range sensors, bumper and contact sensors, and IMU.
- The shooter is modeled as a fixed low-power red laser module.
- The real system keeps the shooter behind ROS2 services and applies an opponent-target safety check before firing.
