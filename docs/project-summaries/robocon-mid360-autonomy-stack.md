# Robocon MID-360 Autonomy Stack：300 字概括

该项目面向 ROBOCON“飞身上篮”比赛，把 Livox MID-360 官方驱动、内置 IMU、FAST-LIO2、固定地图定位、RGB-D/YOLO 感知、双车通信、比赛总控和 STM32/CAN 执行边界连成一条真实可运行的闭环。系统保留 `CustomMsg` 的 `timebase` 与逐点 `offset_time`，局部里程计和固定地图通过唯一 `map -> odom -> base_link` TF 接口输出；质量状态、位姿新鲜度、失锁/重定位、ACK/心跳/过期和投篮互锁共同决定高层动作。我的工作覆盖传感器安装、时间与外参标定、ROS 2 联调、视觉调参、投篮参数标定、双车通信、STM32 接口和真实竞赛机器人现场验证，团队获得机器人篮球和投篮专项赛全国二等奖。
