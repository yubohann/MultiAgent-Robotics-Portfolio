# RoboCup CBG-WM

CBG-WM 是一个面向规则约束多机器人视觉导航的 ROS 2 + IsaacLab 研究项目，包含不确定性 belief graph、对象交互动力学、风险 MPC、规则屏蔽和回放评测。

英文展示入口：[README.md](README.md)。ROS 2 工作区在 `crc_robocup_vision_ws/`，IsaacLab/RL 在 `isaaclab_sim/`，架构和能力边界在 `docs/`，展示图片和视频在 `assets/readme/` 与 `docs/media/`。

CPU 规则 smoke：

```powershell
python -m pytest tests -q
python isaaclab_sim/rl/evaluate_selfplay.py --episodes 2 --max-steps 8
```

仿真证据不等价于真实机器人部署证据；第三方模型和资产遵循各自许可证。
