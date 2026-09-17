# RoboCup CBG-WM

CBG-WM 是一个面向规则约束多机器人视觉导航的 ROS 2 和 IsaacLab 研究项目。项目包含不确定性 belief graph，也包含对象交互动力学和风险 MPC，还包含规则屏蔽以及回放计分。

英文展示入口是 [README.md](README.md)。ROS 2 工作区在 `crc_robocup_vision_ws/`，IsaacLab 和 RL 代码在 `isaaclab_sim/`，架构说明和能力清单在 `docs/`，展示图片和视频在 `assets/readme/` 和 `docs/media/`。

CPU 规则 smoke 命令如下。

```powershell
python -m pytest tests -q
python isaaclab_sim/rl/evaluate_selfplay.py --episodes 2 --max-steps 8
```

本仓库同时保留仿真阶段证据和 1v1 实机实验记录。第三方模型和资产遵循各自许可证。
