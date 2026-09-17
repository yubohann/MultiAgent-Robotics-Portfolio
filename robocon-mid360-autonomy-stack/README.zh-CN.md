# Robocon MID-360 Autonomy Stack

这是一个面向竞赛移动机器人的仿真优先 ROS 2 自主系统。它连接 Livox `CustomMsg` 和 FAST-LIO2，并集成固定地图定位和感知门控以及动作仲裁和比赛总控。

英文展示入口是 [README.md](README.md)。ROS 2 包在 `src/`，验证工具在 `tools/`，仿真和 Gazebo 资产在 `src/robocon_mid360_simulation/`。

目标环境是 Ubuntu 22.04 和 ROS 2 Humble。依赖无关检查命令如下。

```bash
python3 tools/validate_project.py
python3 tools/run_python_contract_tests.py
```
