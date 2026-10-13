# RoboCup CBG-WM：项目技术说明

## 300 字项目概括

RoboCup CBG-WM 是一个将 ROS 2 Jazzy、IsaacLab、对象中心世界模型和规则约束多机器人战术结合起来的研究工程。系统以两个对抗机器人为主验证规模，传感器侧包含轮速里程计、IMU、2D LiDAR、RGB/深度图像、ToF/碰撞触点和固定激光模块；策略侧把对象观测编码为带不确定性的 belief token，构造带类型的对象交互图，由概率集成世界模型预测对象状态、奖励、终止和规则风险，再用 Flow-proposal CVaR MPC 在短时域内筛选联合动作，最后经过几何安全 shield 执行。旧版 SAC Flow/PolicyFlow 保留为必须比较的基线，规则环境、严格 replay、IsaacLab 三视角回放和 ROS 2 dry-run 共同构成验证链。项目还实现可推动刚体障碍物、凹入式目标、遮挡和激光 dwell/range 规则，并将 Sim2Real 的接口、标定和随机化步骤写成文档。当前公开数字对应已审计的 legacy baseline 或选定回放，不能直接当作 CBG-WM 的 SOTA 结果；这一区分体现了项目对实验归因和安全边界的重视。

## 项目定位

- **问题**：在规则约束和对抗干扰下进行多机器人视觉导航与目标交互。
- **当前主线**：两机器人黄蓝对抗；50v50 是规则级规模扩展，不等于真实机器人验证。
- **工程栈**：Ubuntu 24.04、ROS 2 Jazzy、Nav2、slam_toolbox、IsaacLab、Python/C++。
- **方法主线**：CBG-WM + PolicyFlow 风格 actor + CVaR MPC + action shield；legacy SAC Flow 是基线。

## 技术架构

```text
RGB/depth/AprilTag + lidar/IMU/odom/contact
              -> object-centric belief tokens
              -> typed interaction graph
              -> probabilistic ensemble world model
              -> Flow proposal candidates
              -> CVaR risk MPC and rule gates
              -> geometry-aware action shield
              -> ROS 2/Nav2/shooter or IsaacLab actuator
```

世界模型的预测头同时覆盖短期状态、奖励、终止和四类规则风险；评测报告 1/5/10-step prediction、风险校准、CVaR、OOD 和成对 counterfactual 方向。动作 shield 负责把策略输出约束到碰撞、射击距离、视线、dwell 和规则可行域内。

## 核心代码结构

| 路径 | 作用 |
| --- | --- |
| `sim/rl/` | 规则环境、SAC Flow/PolicyFlow、CBG-WM 训练和评估 |
| `sim/rl/evaluate_selfplay.py` | 规则环境 self-play 和批量评估入口 |
| `ros_ws/src/` | ROS 2 包、Nav2、slam_toolbox、视觉目标和 shooter service |
| `docs/cbg_wm.md` | 当前方法的 belief、图动力学、ensemble 和 MPC 合同 |
| `docs/capabilities.md` | 已验证规模、指标和 Sim2Real 边界 |
| `docs/figures/`、`docs/rl_data/` | 论文图、训练曲线、回放和审计导出 |
| `scripts/run_isaaclab_project.ps1` | Windows 上隔离 IsaacLab 配置、日志和扩展缓存 |

ROS 2 侧以 launch 文件组合导航、目标文件、团队颜色和 shooter dry-run；IsaacLab 侧使用同一套比赛规则和几何约束生成 replay。严格回放会重新检查静态/箱体穿透、机器人接触、目标归属和 base-rush 规则，避免只看训练日志。

## 开发过程

1. 先把比赛规则、传感器和执行接口写成可测试的 rule environment。
2. 保留并审计 legacy SAC Flow，形成可重复 baseline，再接入 belief graph 和 world model。
3. 为障碍箱、凹入目标、激光范围/dwell、碰撞和 OOD 风险补充 counterfactual 与 safety tests。
4. 用 ROS 2 dry-run、IsaacLab 三视角回放和严格 replay 对齐仿真与机器人接口。
5. 再开展 Sim2Real 标定、域随机化和 1v1 实机实验；规模扩展单独报告。

## 涉及知识点

- ROS 2 Jazzy、Nav2、slam_toolbox、launch 和服务接口
- 对象中心表征、typed graph dynamics、概率集成世界模型
- SAC Flow、Flow proposal、CVaR MPC 和 OOD 风险评估
- 多机器人博弈、规则状态机、目标交互和可推动刚体
- 视觉 AprilTag、LiDAR/IMU/轮速融合与 Sim2Real
- 几何安全屏蔽、碰撞审计、严格回放和实验归因

## 测试与验证

```bash
python -m pip install -e ".[dev]"
python -m pytest tests -q
python -m pip install -r sim/rl/requirements.txt
cd sim/rl
python evaluate_selfplay.py --episodes 8
```

无硬件 ROS 2 冒烟：

```bash
ros2 launch rcvrl_bringup competition.launch.py start_navigation:=false shooter_dry_run:=true auto_start:=false
```

IsaacLab 预览：`.\scripts\run_isaaclab_project.ps1 -Headless -DemoFlow -Duration 120`。README 中的 128-episode、strict replay 和 50v50 表格必须按其标注的 baseline、replay 或 simulation-stage 解释，不能统称为 CBG-WM 或实机 SOTA。

## 学术申请中的项目价值

该项目适合申请机器人学习、世界模型、多智能体强化学习和具身智能方向。亮点在于把学习策略嵌入规则和安全约束，而不是只展示一条成功轨迹；同时能说明如何用 legacy baseline、counterfactual、OOD、replay audit 和 Sim2Real contract 支撑可复核研究。

## 运行入口

- Python 规则评估：`sim/rl/evaluate_selfplay.py`
- ROS 2 比赛入口：`ros_ws` 中的 `rcvrl_bringup/competition.launch.py`
- IsaacLab 项目包装器：`scripts/run_isaaclab_project.ps1`
- 方法与边界：`docs/cbg_wm.md`、`docs/capabilities.md`

## 证据边界

项目公开结果包含规则环境、IsaacLab 回放和后续 1v1 实机覆盖，但完整 rosbag 和硬件统计未作为公开数据集发布。50v50 仍是仿真阶段。任何方法数字都必须回到对应 run、replay、配置和图表文件；不能把 legacy SAC Flow 数字改写成 CBG-WM 的结果，也不能把仿真演员数量改写成真实机器人数量。

## 源码审计记录

| 审计层 | 代表文件/目录 | 需要核对的实现事实 |
| --- | --- | --- |
| ROS 2 bringup | `ros_ws/src/rcvrl_bringup/` | launch 是否按 team color、target file、dry-run、auto-start 和 navigation 参数组合节点。 |
| 导航与定位 | `rcvrl_navigation/`、Nav2 参数、slam_toolbox 配置 | frame、costmap、地图/定位模式和目标文件是否与比赛状态机一致。 |
| 视觉 | 相机节点、AprilTag/Tag36h11 适配 | 检测结果如何转换成目标 belief；检测失败是否进入安全/重试状态。 |
| 执行 | shooter service、motion/controller 节点 | 发射距离、dwell、line-of-sight 和对手/己方目标规则是否由执行端复核。 |
| 规则环境 | `sim/rl/` | 场地、目标、遮挡、可推动箱体、碰撞、HP、base damage、终止和奖励。 |
| 学习模型 | CBG-WM 方法模块、legacy SAC Flow | belief token、typed graph、ensemble head、Flow proposal、CVaR 风险和 baseline 开关。 |
| 回放 | IsaacLab replay、strict audit、`docs/rl_data/` | 是否重新计算接触、穿透、目标归属、规则违规，而不是只回放 policy log。 |
| 证据 | `docs/capabilities.md`、`docs/admissions_project_brief.md` | 1v1、50v50、实机、Isaac 和规则环境的证据层级与未公开数据边界。 |

## 一次策略执行的详细路径

1. 传感器输入先经过时间/坐标整理，视觉对象和地图状态形成 object-centric belief。
2. belief token 按对象类型、关系和可见性组图；图动力学输出未来状态分布及不确定性。
3. ensemble 预测奖励、终止和规则风险；Flow actor 产生多条短期联合动作 proposal。
4. CVaR MPC 对 proposal 的尾部风险排序，筛掉不可解释的碰撞、射击和区域违规。
5. geometry-aware shield 根据障碍、目标、距离和视线再做一次硬约束，剩余动作才交给模拟或 ROS 2 控制器。
6. replay 记录观测、动作、预测、shield 修改和实测状态；strict audit 再独立判定结果。

## 规则状态机复核

比赛状态至少要能区分 BOOT、PREFLIGHT、WAIT_START、ACTIVE、PAUSED、RECOVERY、FAULT、ESTOP 和 FINISHED 等阶段。`ACTIVE` 也不能直接等于“允许射击”：导航完成、目标确认、姿态/距离有效、dwell 达标、己方目标过滤、碰撞/接触安全和 shooter service 在线必须同时满足。暂停或定位不可信时，策略可以继续产生候选，但 action gate 必须拒绝执行或降级为安全动作。

## 结果读取规则

- 128-episode 黄/蓝胜率表属于选定 baseline 的 stochastic evaluation，不能直接标成 CBG-WM 训练结果。
- strict replay 表是小规模审计，重点是 hard violation、warning 和规则闭环，不等同于完整统计评测。
- 50v50 表是 rule-level / Isaac tactical replay 的规模探索，不代表 100 个真实刚体机器人或分布式训练已验证。
- 实机 1v1 覆盖说明接口和实验发生过，但没有公开 rosbag/统计表时，不应声称可独立复核硬件性能。

## 交付检查表

- [ ] legacy baseline 与 CBG-WM 使用相同规则、预算和 replay audit。
- [ ] world-model prediction、CVaR、OOD 和 counterfactual 输出有独立记录。
- [ ] action shield 的修改量、拒绝原因和安全指标可追溯。
- [ ] ROS 2 dry-run、Isaac replay 和实机证据分开存档。
- [ ] 50v50、1v1、simulation-stage 和 real-robot 规模没有混写。

## 原理与理论基础

### 1. 对抗导航的 POMDP 与 belief graph

比赛机器人只能看到局部图像、LiDAR 和自身状态，真实的对手位姿、目标状态和障碍交互是隐藏变量，因此适合用 POMDP 表示。系统将观测编码为对象 token，并构造带类型边的图 `G_t=(V_t,E_t)`；节点表示机器人、目标、箱体、基地等对象，边表示可见、遮挡、推挤、攻击或保护关系。相比把所有像素拼成一个向量，图结构可以把“谁和谁发生什么关系”显式传给动力学模型，也便于在对象数量变化时复用参数。

### 2. 概率世界模型

世界模型学习 `p_theta(z_{t+1}, r_t, d_t, c_t | z_t, a_t)`，其中 `z` 是对象图状态，`r` 是奖励，`d` 是终止，`c` 是规则风险通道。使用 ensemble 的意义不是装饰，而是用多个模型的均值和方差近似 epistemic uncertainty；在训练分布外的推挤、遮挡或目标布局上，模型分歧可作为拒绝高风险 proposal 的信号。训练时要区分预测损失、奖励/终止损失、风险校准损失和策略损失，不能用单一平均 loss 推断真实战术效果。

### 3. Flow proposal 与 CVaR MPC

Flow policy 先生成动作候选分布，而 MPC 在长度为 `H` 的短时域上滚动评估候选。对于随机成本 `J`，CVaR 在置信水平 `alpha` 下定义为 `CVaR_alpha(J) = min_eta [ eta + 1/(1-alpha) E[(J-eta)_+] ]`。它关注尾部风险，不只优化平均奖励；因此候选即使平均得分高，只要碰撞、误射或规则违规的尾部成本过大，也会被拒绝。随后 action shield 用几何和规则硬约束做最后过滤，形成“学习策略提出、风险模型排序、规则层执行”的分层结构。

### 4. 安全与可达性

几何 shield 可以看作对动作集合 `A` 的投影：`a_safe = argmin_{a in A_safe} ||a-a_policy||^2`。`A_safe` 由障碍距离、速度/加速度、激光 range、line-of-sight、dwell、目标归属和碰撞状态共同定义。若集合为空，系统应输出停止/恢复，而不是把非法动作裁剪成看似合理的值。strict replay 重新计算接触和规则状态，就是在验证这个安全集合是否真的被执行。

### 5. 理论假设和失效模式

## 开发过程深拆

### 阶段 A：规则环境先于策略

先实现可确定执行的场地、目标、障碍、射击距离、dwell、碰撞和终止，策略只在这个环境上提出动作。这样可以独立测试规则，不会因为 RL 尚未收敛就无法判断是模型问题还是环境问题。

### 阶段 B：baseline 作为回归锚点

legacy SAC Flow/PolicyFlow 是必须保留的控制锚点。新 CBG-WM 只有在相同 episode budget、随机评估、碰撞审计和 replay 规则下比较，才能讨论 world model 是否带来改善。

### 阶段 C：从 belief 到风险动作

对象解析、图构造、ensemble 预测、Flow proposal、CVaR 排序和 shield 每一层都应输出可检查的中间结果。训练时记录预测误差和不确定性，评估时记录候选被拒绝的原因，回放时再用真实状态重算违规。

### 阶段 D：ROS/Isaac/实机三条证据线

Python rule environment 适合单测，IsaacLab 适合视觉和刚体回放，ROS 2 workspace 适合接口与部署 smoke，真实机器人实验只负责硬件证据。三者共享规则和消息语义，但不共享未经声明的结果。

## 源码阅读路线

先读 docs/cbg_wm.md 和 capability matrix，再读 rule environment 的 reset/step/termination，然后跟踪 world-model forward、proposal ranking、shield 和 strict replay。最后读 ROS launch 与 Sim2Real 文档，核对 frame、topic、service、参数和执行器接口。

方法假设对象检测和状态估计至少有可用的新鲜度，训练/评测规则和地图保持一致，ensemble 的不确定性在 OOD 区域具有排序意义。视觉丢帧、world-model 崩溃、地图漂移、对手规则变化或 shield 过度保守都会使策略退化；因此公开结果必须同时展示 baseline、OOD/counterfactual、replay hard violations 和失败原因。
