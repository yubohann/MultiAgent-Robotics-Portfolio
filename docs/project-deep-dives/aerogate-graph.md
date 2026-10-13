# AeroGate Graph：项目技术说明

## 300 字项目概括

AeroGate Graph 是一个以固定高度二维动力学为核心的模块化多无人机竞速研究环境。项目将场景几何、运动约束、碰撞检测、图观测、全局 A* 路由、虚拟结构编队、动作安全屏蔽和强化学习训练拆成可组合模块；单机、可变规模多机、动态门密度和内部规划基线共用同一套几何语义。核心环境只依赖 NumPy，因而可以在没有 PyTorch、Isaac Lab 或 GPU 的机器上执行 smoke、回放和确定性报告；Graph-SAC、Graph-MASAC、behavior cloning、DAgger 和 FlashSAC 作为可选训练路径，Isaac Lab 适配器只负责把同一任务几何用于可视化/回放，不替代核心测试。每个 seed 会被 reset/step 两次，报告记录奖励、终止、clearance、队形误差、路径和运行时版本；测试覆盖布局、配置、导入、公开 API、团队几何、单/多门环境和复现。项目因此适合作为多智能体路径规划与安全控制的可复核基线，但二维固定高度结果不能直接外推到真实飞行器、感知或执行器安全。

## 项目定位

- **抽象**：固定高度、速度/加速度受限的二维无人机竞速。
- **任务族**：single-static、multi-static、multi-dynamic；另有 gate-density 和内部 planner 基线。
- **主要方法**：图观测、A* 路由、虚拟结构编队、Graph-SAC/Graph-MASAC、安全 shield。
- **可选依赖**：PyTorch 用于训练，Isaac Lab 用于渲染/回放；基础环境不强依赖二者。

## 技术架构

```text
scenario config -> gate layout / obstacle map
                -> Kinematics2DUpdater
                -> route planner + virtual structure
                -> graph observation + action shield
                -> reward / termination / diagnostics
                -> policy training or IsaacLab replay adapter
```

`Kinematics2DUpdater` 先按轴或向量裁剪速度，再按加速度限幅，最后积分位置并按速度更新 yaw；`GateObstacleMap2D` 对圆形门柱提供点、局部查询和线段碰撞检查。多机环境维护 active mask、队形 slot、中心线路径、动态门缓存、规划器延迟和 shield 诊断，并把这些信息放入 `info`/snapshot。

## 核心代码结构

| 路径 | 作用 |
| --- | --- |
| `aerogate/` | 稳定 public API、CLI、scenario registry 和 smoke/reproduce 命令 |
| `shared/core/` | 2D 数学、运动学、碰撞、动态门、团队几何和高度不变量 |
| `single_gate/` | 单机环境、观测、奖励、Graph-SAC 和 replay |
| `multi_gate/` | 多机环境、编队、A*、guidance、safety shield、Graph-MASAC |
| `density_single/`、`density_multi/` | 门密度 curriculum、评测和论文表格工具 |
| `internal/` | planner baseline、arbitration、method evaluation 和 ablation |
| `assets/`、`shared/visualization/` | 门/无人机资产、布局和 IsaacLab 适配 |
| `tests/`、`evaluation_artifacts/` | CPU 回归、复现 JSON、manifest、报告和表格 |

## 开发过程

1. 先建立与 Isaac Lab 无关的二维合同和最小门布局，保证环境可独立测试。
2. 将单机运动学、碰撞和观测扩展为可变规模团队，再引入虚拟结构和全局路径。
3. 把动态门、密度 curriculum、规划器和 shield 作为独立 profile，避免改动基础环境语义。
4. 加入 Graph-SAC/MASAC、BC/DAgger 和 FlashSAC 训练脚本；评测脚本统一输出 manifest。
5. 通过 IsaacLab replay adapter 提供可视化，仍以 NumPy core 的合同测试作为最低准入。

## 涉及知识点

- 二维受限运动学、积分和 yaw 计算
- 圆形障碍、线段碰撞、clearance 和动态门
- 图节点/邻接/active mask 观测
- A* 路由、lookahead guidance 和虚拟结构编队
- MARL、Graph-SAC、Graph-MASAC、BC、DAgger
- 动作安全 shield、碰撞/队形奖励和可变团队规模
- seed 控制、JSON-safe diagnostics、IsaacLab replay

## 测试与验证

```powershell
uv sync --extra dev
uv run python -m aerogate info
uv run python -m aerogate smoke --scenario single-static --steps 8
uv run python -m aerogate smoke --scenario multi-static --agents 4 --steps 8
uv run python -m aerogate reproduce --scenario multi-static --agents 4 --seeds 3 7 11 --steps 8 --output artifacts/reproducibility/multi-static.json
uv run python -m pytest
uv run ruff check aerogate tests
```

复现命令两次执行同一 seed 的零动作轨迹，并比较公开诊断；它证明的是当前运行时 core 的一致性，不是训练策略质量。`evaluation_artifacts/` 中的结果也应按 scenario、seed、依赖和脚本版本解释。

## 学术申请中的项目价值

该项目适合展示“把研究想法拆成可测模块”的能力：运动学、几何、规划、编队、策略和可视化有明确边界，既能做算法实验，也能做安全和复现审计。相比只提交训练曲线，图观测合同、动态难度和 planner baseline 更容易支撑研究讨论。

## 运行入口

- CLI：`aerogate/__main__.py`、`aerogate/cli.py`
- 单机：`single_gate/scripts/train_single.py`、`eval_single.py`
- 多机：`multi_gate/scripts/train_multi.py`、`eval_multi.py`
- IsaacLab 回放：`single_gate/scripts/replay_single_isaaclab.py`、`multi_gate/scripts/replay_multi_isaaclab.py`

## 证据边界

核心环境是二维固定高度抽象。通过 pytest、smoke 或 reproduce 不代表真实飞行动力学、感知、执行器、通信延迟或物理安全已经验证。PyTorch 训练、IsaacLab 渲染和保留 checkpoint 需要单独披露硬件、依赖、seed、预算和结果来源。

## 源码审计记录

| 层次 | 文件群 | 复核内容 |
| --- | --- | --- |
| 公共入口 | `aerogate/cli.py`、`scenarios.py`、`reproducibility.py` | scenario 名称、参数解析、JSON-safe 输出、版本/NumPy/Python provenance。 |
| 数学/运动 | `shared/core/math_2d.py`、`kinematics_2d.py`、`fixed_height.py` | 向量范数、yaw、速度/加速度轴限幅、dt 积分和固定高度不变量。 |
| 几何/碰撞 | `collision_2d.py`、`dynamic_gate_density_2d.py`、`team_geometry.py` | 圆形门柱距离、线段碰撞、动态门、corridor、队形 slot 和 clearance。 |
| 单机 | `single_gate/env/`、`observation_single.py`、`rewards/` | action/observation shape、奖励、终止、Graph-SAC 和 replay。 |
| 多机 | `multi_gate/env/`、`formation/`、`planners/`、`guidance/` | active agent、virtual structure、A* waypoint、lookahead heading、动态场景和 shield。 |
| 策略 | `multi_gate/graph_rl/`、`single_gate/graph_rl/` | replay buffer、Graph-SAC/MASAC、FlashSAC 的输入输出与 mask。 |
| 训练/评测 | `multi_gate/scripts/`、`gate_density_*/scripts/` | curriculum、BC/DAgger、planner baseline、评测和结果导出。 |
| 资产/回放 | `assets/`、`shared/visualization/`、`*_isaaclab.py` | USD/URDF 布局、视觉配置和 Isaac adapter 是否改变 core 语义。 |
| 回归 | `tests/`、`evaluation_artifacts/` | 布局、配置、导入、API、复现和指标表是否保持一致。 |

## 一次多机 step 的调用链

1. `reset(seed, num_agents)` 建立随机源、动态门布局、初始状态、route 和虚拟结构 slot。
2. 输入 action 先按 active mask 解析，再由速度/加速度模型生成期望命令。
3. global planner 给出 centerline waypoint；guidance 产生 lookahead heading；formation 将中心线变成每个 agent 的 slot。
4. graph observation 将 agent node、gate/obstacle node、邻接矩阵、node mask 和 action mask 组装成固定 shape。
5. safety shield 对速度、门柱、动态门、队形崩溃和边界做修正；环境积分并计算 clearance、slot error、goal progress。
6. reward/termination 组合通过、碰撞、超时、corridor miss 和队形失败，`info` 写入 planner call count、latency 和失败原因。

## 关键算法的代码对应

- **运动学**：`Kinematics2DUpdater.step` 先裁剪 desired velocity，再限制 `dv`，最后积分 `x/y/t`；yaw 只在速度超过 deadband 时更新。
- **碰撞**：`GateObstacleMap2D.min_signed_distance` 用障碍半径和 drone radius 计算 clearance，`segment_collides` 检查整段运动而非只检查终点。
- **编队**：`VirtualStructure2D` 根据中心线和 heading 生成 slot，`slot_error_stats` 把平均/最大误差传入奖励和诊断。
- **路线**：`GlobalRoutePlanner2D` 在障碍 map 上提供 waypoint，动态 gate 场景使用独立缓存和 corridor 状态。
- **策略输入**：graph policy 必须尊重 node/action mask，不能把 padding agent 当作真实成员。

## 研究实验如何避免混淆

核心 reproduce 只比较同一 scenario、seed、零动作和运行时 diagnostics；训练结果要另外报告 policy checkpoint、训练预算、设备、随机种子和评测脚本。single-static、multi-static、multi-dynamic 不能只用“成功率”一项合并，至少应同时保留 reward、clearance、pair separation、formation error、planner latency 和失败原因。

## 交付检查表

- [ ] 运动学、碰撞和固定高度合同先通过 CPU 测试。
- [ ] 可变团队的 mask、slot、route 和 action shape 有回归。
- [ ] dynamic gate profile 的缓存、碰撞和 corridor 状态可解释。
- [ ] shield 修正和拒绝原因进入 `info`，不是静默覆盖 action。
- [ ] IsaacLab replay 与 NumPy core 使用同一几何参数和 scenario manifest。

## 原理与理论基础

### 1. 受限二维动力学

环境用速度控制的离散模型近似固定高度飞行：v(t+1)=clip(v(t)+clip(v_cmd-v(t), a_max*dt), v_max)，p(t+1)=p(t)+v(t+1)dt。轴向限速和向量限速是两种不同模型，不能在配置中混用而不记录。yaw 在速度超过 deadband 时按 atan2(v_y,v_x) 更新，避免无人机静止时的航向抖动。该抽象的意义是让规划、队形和安全算法可重复，而不是模拟完整姿态、推力和气动。

### 2. A* 与几何安全

全局规划把门柱/障碍地图离散或抽象成可查询几何，A* 使用 f(n)=g(n)+h(n) 在候选 waypoint 上搜索低代价路线。heuristic 必须不高估剩余代价才保持最优性；动态门场景则需要在时间或每步重规划中处理移动障碍。segment_collides 检查运动线段，min_signed_distance 给出带 drone radius 的 clearance，二者共同避免“终点没撞但中间穿过门柱”。

### 3. 图观测与多智能体策略

多机观测可表示为节点特征矩阵 X、邻接矩阵 A、节点 mask 和 action mask。图策略通过消息传递 H(l+1)=sigma(A H(l) W(l)) 聚合邻居，天然适配可变 team size；padding 节点必须被 mask，否则网络会把不存在的无人机当作真实信息。Graph-SAC/MASAC 仍优化熵正则目标，差别在于 actor/critic 如何共享或分离 agent、图和全局状态。

### 4. 虚拟结构编队与奖励

虚拟结构把团队中心线和 heading 转成 slot，控制目标可写为 e_slot=sum_i ||p_i-p_i_desired||^2。奖励通常同时考虑 route progress、goal distance、slot error、pair separation、clearance、action smoothness 和 collision penalty；只奖励到达会鼓励个体抢路，只有队形奖励又可能停滞。因此 snapshot 中记录 mean/max slot error、goal distance 和 clearance，帮助区分规划失败、编队塌缩和碰撞。

### 5. 理论假设和失效模式

该环境假设所有 agent 位于同一固定高度平面，门柱可以用圆形障碍表示，动作在离散时间内以速度命令执行，团队可以共享完整的几何状态。由此得到的结论只适用于规划层、编队层和安全接口的比较。典型失效包括：A* 在障碍膨胀后找不到可行路径；动态门移动导致旧 waypoint 过期；slot 目标在边界或门柱附近不可同时满足；shield 过于保守造成停滞；padding 或 mask 错误使策略读取不存在的 agent。每一类失效都应以 `done_reason`、几何诊断或回放片段记录，而不是只用总 reward 概括。

## 开发过程深拆

### 阶段 A：几何核心

先实现 math_2d、kinematics_2d、collision_2d 和固定高度检查，让零动作、单门和静态地图可重复。所有上层算法都依赖这些几何语义，因此 core 不应导入 PyTorch 或 Isaac Lab。

### 阶段 B：单机到多机

单机环境先确定 action/observation/reward/termination；多机环境再增加 active agent、slot、route、邻接、mask 和 team metrics。这样发生 shape 错误时可以定位到扩展层。

### 阶段 C：动态难度和安全

动态门密度、moving posts、corridor、height escape、side bypass 和 formation collapse 作为独立状态；每一种失败都写入 info。shield 必须记录修改前后 action、最小 clearance 和 veto 原因。

### 阶段 D：策略和回放

Graph-SAC/MASAC、BC/DAgger 与 planner baseline 使用相同环境和评测字段；训练脚本只负责 checkpoint/metric，IsaacLab adapter 读取任务级状态做可视化。

## 源码阅读路线

从 aerogate/cli.py 进入，追到 scenario factory；再读 Kinematics2DUpdater.step、obstacle map、planner、virtual structure、observation、reward/termination 和 shield；最后读 Graph policy、training script、replay adapter 和 tests。

环境假设平面、固定高度、圆形障碍和受限速度足以回答研究问题；它不包含风、姿态耦合、真实感知或执行器延迟。A* 在动态密度变化、slot 无可行解或 guidance 过期时可能无路；shield 过强会导致停滞，过弱会放行碰撞。必须把这些情况当作可分类的终止原因，而不是只报平均 reward。

---

# 深入技术说明：从源码到实验设计

## 阅读范围和证据等级

本节是在逐文件阅读项目源码、配置、测试和评估附件后整理的技术说明。为了避免把“代码里存在”误写成“实验已经证明”，全文使用三种证据等级：

1. **源码事实**：可以在当前文件、函数或数据结构中直接定位的行为，例如 `MultiGate2DEnv.step` 的调用顺序、图节点的字段、检查点元数据字段。
2. **历史附件证据**：位于 `evaluation_artifacts/` 的旧报告、CSV 或 JSON。它们能够说明曾经运行过某个版本，但不自动代表当前源码、当前配置或论文最终结果。
3. **建议性实验**：为了形成论文级结论而提出的对照、消融或统计方案。它们是下一步工作，不应写成已经取得的结果。

这一区分很重要。AeroGate Graph 的核心环境是可执行的二维固定高度抽象；训练、Isaac Lab 回放和历史评估各自有独立依赖与证据链。下面所有“应当”“建议”均属于实验设计，不等于当前仓库已经完成。

## 1. 研究问题：项目真正要回答什么

项目名称中包含 Graph、Gate 和 Aero，容易让读者以为它是完整的飞行器仿真器。更准确的理解是：它把“多智能体在有门柱和移动门约束的平面中协同通过”压缩成一个可复核的研究环境，用来研究以下问题。

### 1.1 图结构是否能承载可变规模团队信息

团队规模从 2 到 12 变化时，逐个拼接固定编号的状态向量会遇到两个问题：一是输入维度和 agent 身份绑定，二是填充位置容易被策略误当成真实成员。项目把 agent、期望 slot、路线 waypoint、障碍物和全局节点放入统一图中，以 `node_mask` 表示有效节点，以 `action_mask` 表示有控制权的 agent。研究问题不是“图网络一定更好”，而是：在相同几何和相同控制接口下，图消息传递能否减少对固定团队规模的依赖。

### 1.2 全局路线、编队 slot 与局部安全修正如何分工

全局 A* 只规划团队中心线；虚拟结构把中心线变成每个 agent 的 slot；策略根据图观测产生速度动作；安全屏蔽在动作执行前修正潜在碰撞、越界和编队风险。这个分层允许分别做消融：只用全局规划、只用局部策略、加入 slot、加入 shield，然后观察成功率、最小净空、队形误差和停滞率的变化。

### 1.3 闸门密度和动态运动对协作的影响

静态闸门能验证几何和编队的基本闭环，但不能检验策略对时变障碍的响应。`dynamic_gate_density_2d.py` 使用 seed 生成闸门布局，再以 lateral、antiphase 或 lissajous 模式更新门中心，并在每个时间步计算移动门柱速度、实时净空和 swept 净空。这里要研究的是困难度变化是否导致可解释的失败类型，而不是单纯追求一条漂亮轨迹。

### 1.4 项目的非目标

当前代码没有证明以下命题：

- 没有证明真实无人机的六自由度动力学、姿态稳定、推力分配或电机饱和；
- 没有证明真实相机、激光雷达、定位、通信延迟或丢包条件下仍能安全；
- 没有证明 Isaac Lab 的 USD 外观与二维几何具有物理一致性；
- 没有证明某个 Graph-SAC 变体在论文意义上优于所有经典规划器；
- 没有把历史附件中的少量 episode 当作统计充分的正式基准。

把非目标写清楚并不会削弱项目，反而能让读者知道这个环境适合回答什么问题、不能回答什么问题。

## 2. 代码仓库的完整地图

### 2.1 公共入口层：`aerogate/`

`aerogate/cli.py` 是用户面对的命令行入口，负责解析 `info`、`smoke` 和 `reproduce` 等子命令，建立场景并把诊断转换为可序列化结构。`aerogate/scenarios.py` 是场景注册表：将 `single-static`、`multi-static` 和 `multi-dynamic` 映射到具体环境工厂，同时集中处理团队规模和步数参数。`aerogate/reproducibility.py` 记录 Python、NumPy、运行时版本和场景配置，并执行同 seed 的重复 rollout。

这一层的设计价值是稳定 API。上层训练脚本可以更换实现，下游使用者仍然通过统一场景名称启动 smoke 或复现检查。公共 API 不应依赖 Isaac Lab，也不应在导入时强制加载 PyTorch，否则最基本的几何测试会被可选依赖阻塞。

### 2.2 几何核心层：`shared/core/`

- `math_2d.py`：向量范数、单位化、角度、投影等小型纯函数；
- `kinematics_2d.py`：速度命令、速度/加速度限幅、离散状态积分；
- `collision_2d.py`：圆形门柱、障碍地图、点碰撞、线段碰撞和 signed clearance；
- `fixed_height.py`：固定高度和高度带的约束检查；
- `dynamic_gate_density_2d.py`：动态闸门生成、运动、非重叠投影、corridor 和 swept collision；
- `team_geometry.py`：队形统计、横向 band、slot error 和团队中心几何。

这些模块只依赖标准库和 NumPy。它们共同构成一个“几何合同”：上层策略只能在这个合同上工作，不能偷偷改变半径、边界、时间步或碰撞定义。

### 2.3 配置和运行时：`shared/configs/` 与 `shared/runtime/`

`shared/configs/global_config.py` 提供全局默认值，例如固定高度、时间步、最大团队规模和可复现实验的默认场景。`shared/runtime/` 负责产物目录、TensorBoard、训练控制、向量化环境辅助和实验运行名。运行时模块不应该改变物理语义；它的职责是把配置、日志、检查点和评估结果组织起来。

### 2.4 单机路径：`single_gate/`

`single_gate/env/single_gate_env.py` 是单机版本的最小环境，适合先验证动作映射、单机奖励和终止。`observation_single.py` 构造单机观测；`rewards/single_agent_rewards.py` 计算单机奖励；`graph_rl/graph_sac.py` 和 `graph_flashsac.py` 提供单机图策略训练；`replay.py` 是基于规则或启发式的回放控制器。单机路径是多机扩展前的回归基线，不应和多机指标直接混为一张表。

### 2.5 多机路径：`multi_gate/`

多机目录将环境、观测、奖励、编队、规划、guidance、屏蔽、策略和训练拆开：

- `env/multi_gate_env.py`：维护团队状态并串联一个时间步；
- `env/observation_multi.py`：把几何状态编码为固定尺寸图；
- `env/observation_runtime.py`：提供环境调用所需的观测辅助和序列化逻辑；
- `env/reward_runtime.py`：集中生成 `info`、快照和诊断字段；
- `env/safety_shields.py`：在速度空间修正动作；
- `formation/virtual_structure.py`：生成局部 slot 并投影到世界坐标；
- `planners/global_route_planner.py`：A* 中心线规划和 waypoint 简化；
- `graph_rl/graph_policy.py`：共享参数图编码器和 actor 头；
- `graph_rl/replay_buffer.py`：保存图观测、联合动作、失败标签和安全代价；
- `imitation.py`、`dagger.py`：专家轨迹、行为克隆和风险状态纠正；
- `training.py`：向量环境采样、更新、评估、检查点和安全恢复。

### 2.6 密度实验和内部基线

`density_single/` 与 `density_multi/` 是面向闸门数量、运动速度和课程学习的实验入口。它们复用核心几何，但有更专门的任务配置和评估脚本。`internal/` 中的 `classic_planners.py`、`strong_planners.py`、`arbitration.py` 和 `ablation.py` 用于比较 A*、Theta*、RRT*、启发式或强规划器以及安全仲裁策略。这里的“内部”表示实验基线组织方式，不表示这些规划器已经完成公平的论文对比。

### 2.7 资产、回放和验证

`assets/gate_scene_layouts.py` 描述二维布局到三维闸门变换的映射；`assets/gate_scene_builder.py` 创建地面、光照、闸门 USD 和无人机外壳；`shared/visualization/scene_isaaclab.py`、`single_gate/isaac_replay.py` 和 `multi_gate/isaac_replay.py` 提供可选回放。`tests/` 覆盖导入、配置、布局、公开 API、单/多机 smoke、团队几何和复现。`evaluation_artifacts/` 保存历史结果、manifest 和报告，不能作为实时运行目录使用。

## 3. 配置如何变成一次可执行实验

项目没有把参数散落在训练脚本中，而是通过嵌套数据类组合实验。一次多机运行可以按以下顺序理解：

1. `MultiExperimentConfig` 指定实验编号、最小/默认/最大团队规模、场景、规划器、观测、算法、失败回放、团队规模抽样、DAgger 和评估门槛；
2. `MultiGateEnvConfig` 指定时间步、速度/加速度、边界、门柱/无人机半径、目标半径、奖励系数、终止条件和 shield 开关；
3. `MultiGraphObservationConfig` 根据最大 agent、waypoint、障碍和 guidance 数量计算固定 `max_nodes`；
4. `MultiPlannerConfig` 指定网格分辨率、障碍膨胀半径、安全裕度、A* 最大搜索次数和 waypoint 简化步长；
5. `MultiFormationConfig` 指定横向/纵向 slot 间距、列数、目标 slot 容差和可选 bootstrap 形状；
6. `DynamicGateDensity2DConfig` 指定闸门数量、空间范围、门柱半径、动态幅值/速度、corridor 和固定高度合同；
7. 环境构造时读取这些配置，创建障碍图、虚拟结构、全局规划器、运动学更新器和 guidance 引擎；
8. 训练器再依据同一个 `observation_shapes` 创建 actor、critic 和回放缓冲。

配置嵌套的直接好处是评估时能保存完整实验签名；风险是“只改一个字段”可能改变很多行为。因此每次正式运行都应保存 resolved config、代码版本、依赖版本、seed、设备和命令行参数。

## 4. 一次 `reset` 和 `step` 的完整执行顺序

### 4.1 `reset` 的职责

`MultiGate2DEnv.reset` 先重置随机源、步数、规划器计数、动态闸门缓存、shield 诊断和失败状态。随后依据显式路径或随机范围选择起点与目标中心，计算团队规模对应的横向安全 padding，并调用 `_resolve_safe_start_setup` 确认起点 slot 不越界、不落在障碍内部。

起点和目标确定后，环境创建每个 agent 的 `KinematicState2D`，初始化速度、偏航和时间，建立初始中心线、slot、目标距离和 route guidance。最后构造第一份图观测和 `info`。因此第一次 `reset` 返回的 `info` 不代表一次成功，而是包含当前场景合同、团队规模、目标和诊断的初始快照。

### 4.2 `step` 的职责

一个时间步严格遵循以下顺序：

1. 将输入 action 变成 `(max_agents_soft, 2)`，截取 active agent，并在必要时调用 safety shield；
2. 将归一化 action 转为平面期望速度，再交给 `Kinematics2DUpdater` 限制速度变化并积分状态；
3. 用上一帧和当前帧位置检查门柱线段碰撞；动态闸门同时传入起止门柱位置，以计算相对运动的 swept clearance；
4. 计算 agent 间碰撞、边界越界、团队中心、当前 path index、slot error、目标距离和 guidance tracking error；
5. 读取动态闸门高度合同、corridor 状态和 formation line collapse 状态；
6. 调用 `evaluate_multi_agent_termination`，确定本步是否终止或截断，并给出唯一的 `done_reason`；
7. 调用 `compute_multi_agent_reward`，将进度、slot、净空、分离、动作平滑和终止项组合为标量奖励，同时保留分项；
8. 更新 previous action、previous slot error、previous goal distance 和 episode 状态；
9. 生成下一份图观测和 JSON-safe `info`。

这个顺序有一个关键含义：奖励和终止读取的是“动作积分之后”的几何结果；shield 只在积分前改变动作，并通过 `action_shield_*` 字段留下诊断。若将碰撞检查放在积分前，或者只检查终点而不检查线段，就会改变任务定义。

## 5. 几何和运动学：为什么二维抽象仍然有研究价值

### 5.1 速度控制离散模型

每个 agent 的状态可以写成

\[
 s_t=(x_t,y_t,v^x_t,v^y_t,\psi_t,t),
\]

动作先被解释为归一化平面速度命令，再经由轴向或向量上限转换为 `PlanarVelocityCommand2D`。更新器执行两层限幅：

\[
 \tilde{v}_{t+1}=v_t+\operatorname{clip}(v^{cmd}_t-v_t, -a_{max}\Delta t, a_{max}\Delta t),
\]
\[
 v_{t+1}=\operatorname{clip}(\tilde{v}_{t+1},v_{max}),\qquad
 p_{t+1}=p_t+\Delta t\,v_{t+1}.
\]

代码还允许分别配置前向和侧向速度/加速度上限。偏航只在速度超过 deadband 时用 `atan2(v_y,v_x)` 更新，因此静止时不会因数值噪声反复旋转。该模型回答的是“在速度和加速度受限的平面中，协同控制是否能通过门柱”，不是“真实飞行器的姿态响应是什么”。

### 5.2 门柱和线段碰撞

一个闸门由两个圆形门柱近似。对位置 `p`，考虑无人机半径后的 signed clearance 可写成

\[
 d_{clear}(p)=\min_j\|p-c_j\|-r_{post}-r_{agent}.
\]

`d_clear < 0` 表示穿入碰撞半径。规划器的 `segment_collides` 检查从当前点到候选点的整段运动，因此能发现“起点和终点都安全、但中间穿过门柱”的情况。动态场景再把 agent 和门柱都视为在一个时间步内线性运动，计算相对位置轨迹的最小距离。这是连续碰撞的离散近似，不是高精度连续碰撞检测器。

### 5.3 固定高度合同和 corridor

虽然核心状态只有 `(x,y)`，动态闸门配置仍保存固定高度、开口底部、开口顶部、门中心高度、上下安全边界和最大允许高度。`validate_height_and_corridor_invariants` 检查：固定高度是否等于几何中心、无人机外壳是否位于开口内、最大允许高度是否低于顶部余量，以及 corridor 是否窄于世界边界。二维模拟中的“height escape”因此表现为合同或配置失败，而不是实际 z 轴运动。

`corridor_region_status` 在闸门区域内检查 active agent 的绝对 y 是否超出 corridor 半宽；`center_has_completed_corridor` 要求团队中心越过闸门区域右边界且仍在 corridor 内。这样可以把“从侧面绕开所有门”的策略从普通到达中区分出来。

### 5.4 动态闸门密度

`generate_gate_layout` 根据 `gate_count` 和 seed 生成基础门中心及 lane/column 元数据。`live_gate_centers` 根据时间、幅值和速度计算运动中心：

- `lateral` 只沿横向摆动；
- `antiphase` 根据列号反向摆动；
- `lissajous` 同时使用较小的纵向分量和横向分量。

每次更新后 `_project_gate_non_overlap` 调整中心，防止不同闸门的门柱相互重叠。`live_gate_velocities` 用相邻时间的中心差分得到速度。几何 sanity 函数会检查闸门数量、中心是否真的运动、速度是否非零、门间是否重叠，以及合成碰撞位置能否被 live/swept 查询检测到。

## 6. 全局路线与虚拟结构编队

### 6.1 A* 中心线

`GlobalRoutePlanner2D.plan` 先将起点和目标映射到网格，再寻找最近的自由 cell。A* 采用八邻域，代价是相邻网格中心的欧氏距离，启发函数也是欧氏距离。障碍判断使用门柱地图加上 `inflation_radius_m`，边界附近的 cell 被视为占用。规划完成后把网格路径转回世界坐标，并用 `segment_collides` 做路径简化；缓存键包含起点 cell、目标 cell 和膨胀半径。

这里的 A* 输出是团队中心线，不是每个 agent 的轨迹。A* 的优点是给出可解释的全局通道；局限是网格分辨率会产生量化误差，动态门移动后旧路径可能过时，且它没有直接优化编队形状或动作平滑。

### 6.2 slot 生成

`VirtualStructure2D` 在局部坐标中生成 slot。默认规则是：1 个 agent 为单列，2--4 个使用两列，5--12 个使用三列；横向默认间距为 2.4 m，纵向默认间距为 3.0 m。局部 offset 为 `(longitudinal, lateral)`，再通过

\[
 p_i^{slot}=c+ h\,o_i^x + (-h_y,h_x)\,o_i^y
\]

投影到世界坐标，其中 `c` 是中心，`h` 是单位 heading。这样 slot 会随着路线方向旋转，而不会固定在世界 x/y 轴。

配置开启 bootstrap 模板时，8 机还可以使用 line、triangle、rectangle、diamond、circle 等局部形状；默认路径仍是规则行列布局。`slot_error_stats` 返回平均和最大 slot 误差，分别用于奖励、终止条件和评估表。

### 6.3 route、slot 和 guidance 的边界

路线决定团队应该往哪里走，slot 决定团队保持什么相对几何，guidance 可以提供一个额外的 lookahead 目标或方向。三者不是同一件事：把 guidance 直接当作真值会掩盖规划失败，把 slot 误差直接当作路径误差也会误诊。实验报告应分别记录 path index、goal distance、guidance tracking error、mean/max slot error 和 formation collapse。

## 7. 图观测：从物理状态到固定张量

### 7.1 固定尺寸设计

`build_multi_graph_observation` 创建四个数组：

| 数组 | 默认含义 |
| --- | --- |
| `node_features` | `(max_nodes, node_feature_dim)`，默认特征维度为 16 |
| `adjacency` | `(max_nodes, max_nodes)`，表示节点间距离关系 |
| `node_mask` | `(max_nodes,)`，1 表示真实节点，0 表示 padding |
| `action_mask` | `(max_agents_soft,)`，1 表示当前团队有控制权的 agent |

`max_nodes` 由最大 agent 节点、slot 节点、lookahead waypoint、局部障碍、guidance 节点和一个 global/team center 节点相加得到。这样同一个 actor 可以处理不同团队规模；代价是小团队会带有更多 padding，必须在编码器和动作头中正确屏蔽。

### 7.2 节点类型和特征

当前实现按固定区域分配节点：

1. **agent 节点**：相对团队中心的位置、速度、slot error、无人机半径、局部净空、进度比例、团队规模和有效标志；
2. **slot 节点**：期望位置、当前 heading、slot 序号、进度比例和团队规模；
3. **waypoint 节点**：路线 lookahead 相对位置、heading、顺序和进度；
4. **障碍/动态门节点**：相对位置、动态速度、预测位置、门柱半径、最近 agent 净空和动态标志；
5. **guidance 节点**：可选的远程目标或风险字段；
6. **global/team 节点**：全局尺度、中心状态或任务状态。

代码通过 `_pad_feature` 将每个列表截断或补零到固定维度。数值使用 50 m 位置尺度、最大速度尺度和 10/20 m 误差尺度做归一化。动态门节点会用 0.75 s 预测位置，但这个预测只进入观测，不等于环境把门固定在预测位置。

### 7.3 邻接和 mask

邻接权重按节点世界位置之间的距离衰减，距离过远的节点影响较小。真实节点之外的 padding 节点虽然在数组中存在，但 `node_mask=0`；策略还使用 `action_mask` 将非 active agent 的动作置零。任何新增节点类型都必须同时更新：节点容量、节点特征、世界坐标表、邻接构造、mask 和测试，否则会出现 silent shape bug。

### 7.4 图观测的理论意义

令节点特征矩阵为 `X`，邻接矩阵为 `A`，有效节点 mask 为 `M`。图编码器的一个消息传递层可以写成

\[
 H^{(l+1)}=\sigma\left(W_l[H^{(l)}\,\|\,D^{-1}AH^{(l)}]\right)\odot M.
\]

其中 `D` 是 masked adjacency 的度矩阵，`||` 表示拼接。因为节点顺序只在固定区域内承担索引作用，策略可以在团队规模改变时共享参数；但这并不自动保证对 agent 置换不变。若实验要声称置换鲁棒性，需要额外做 slot permutation、agent order permutation 和跨团队规模测试。

## 8. 动作安全屏蔽：它是什么，不是什么

### 8.1 处理顺序

`_apply_action_safety_shield` 先将 action 转为速度，再依次执行：

1. 动态闸门已经在团队后方且目标尚未到达时的前向速度下限；
2. agent pair 的预测接近、排斥和制动；
3. 世界边界的内向修正；
4. guidance corridor 的横向修正；
5. 动态门 channel 的横向引导和减速；
6. 静态/动态障碍物的排斥和制动；
7. 将修正后的速度映射回归一化 action。

每一步都使用配置中的 margin、gain、time horizon 和 team-size limit。shield 不修改环境状态，只修改即将执行的动作。

### 8.2 pairwise shield 的基本逻辑

对 agent i、j，代码同时考虑当前距离和短时预测距离，取两者较小值作为风险距离。如果风险距离低于安全距离加激活 margin，计算 0--1 的 closeness。随后可以沿 pair axis 加排斥速度、沿前向减少速度，或在小团队时沿期望 lateral slot 施加外向偏置。动态闸门场景还会混合纵向 heading，避免所有 agent 只在横向互相推开而失去前进方向。

### 8.3 诊断和安全边界

shield 返回 `mean_intervention_norm`、pair/boundary/obstacle/guidance/gate-channel closeness 以及 cruise boost。评估时应报告 shield 介入率和平均修正量；否则一个“成功率高”的策略可能只是频繁被规则层接管。shield 也不是形式化安全证明：它使用离散时间、局部几何和启发式增益，不能替代控制屏障函数、可达性分析或真实硬件验证。

## 9. 奖励、终止和指标解释

### 9.1 奖励组成

`compute_multi_agent_reward` 将以下项分别计算并放入 `reward_terms`：

- route progress：团队中心目标距离的改善；
- goal proximity：接近目标的连续奖励（若配置启用）；
- slot improvement：平均 slot error 改善；
- slot error / max slot error：队形偏离惩罚；
- guidance tracking：偏离 lookahead 或外部 guidance 的惩罚；
- boundary proximity：接近世界边界的惩罚；
- clearance：接近门柱的惩罚；
- separation：接近其他 agent 的惩罚；
- formation line collapse：横向 band 不足或队形塌缩惩罚；
- action L2 和 smoothness：抑制过大、过抖动作；
- survival 与 terminal：存活奖励以及成功/碰撞/越界/超时的终止项。

进度奖励不是无条件发放。代码会根据 slot、最大 slot、边界、pair separation、guidance 等安全 gate 对 progress 进行调制，从而减少“某个 agent 冲到目标、其他 agent 落后”带来的虚假成功。报告中要同时保存 `ungated progress` 和实际 `progress gate`，否则很难解释奖励下降究竟来自路线变差还是安全约束增强。

### 9.2 终止优先级

当前环境先计算几何失败，再交给统一终止函数。实际分析时建议使用如下优先级解释：高度合同失败、侧向绕行、corridor miss、门柱碰撞、agent 碰撞、队形线塌缩、到达目标、越界和超时。具体优先级以当前 `evaluate_multi_agent_termination` 返回的 `reason` 为准，不能根据奖励符号推测。

成功至少应同时满足：`done_reason=goal_reached`（或明确允许 timeout 的配置）、高度合同通过、未发生 bypass/corridor miss/formation collapse、团队中心完成 corridor，且 slot 容差和最小净空条件满足。只看 `terminated=True` 或只看目标距离，都不能作为正式成功定义。

### 9.3 指标应该如何读

建议最少保留以下维度：

| 维度 | 指标 | 解释 |
| --- | --- | --- |
| 任务 | success、done reason、episode steps、goal distance | 是否完成任务以及如何结束 |
| 安全 | min clearance、min pair distance、collision、safety violation | 是否接近或穿入风险区域 |
| 协作 | mean/max slot error、lateral span、line collapse | 团队是否保持目标结构 |
| 规划 | path length、path index、planner calls、planner latency | A* 和 guidance 的开销 |
| 控制 | action norm、smoothness、shield intervention | 策略是否依赖大动作或规则接管 |
| 动态 | gate count、gate speed、motion range、corridor completion | 困难度和动态障碍实际是否生效 |

## 10. Graph-SAC / Graph-MASAC 的实现和理论

### 10.1 编码器

`GraphEncoder` 先用两层线性网络把每个节点嵌入到 hidden space，再进行指定次数的消息传递。每层先用 `node_mask` 清除无效节点，再用 masked adjacency 做批量矩阵乘法，按有效度归一化邻居聚合，最后把自身嵌入和聚合结果拼接。图级向量是有效节点嵌入的平均池化；因此 padding 数量变化不会直接改变 pooled 的尺度。

### 10.2 共享 actor

`GraphPolicy` 取前 `max_agents_soft` 个节点作为 agent embedding，同时拼接图级 pooled embedding，经过共享 backbone 为每个 agent 生成均值和 log standard deviation。默认 actor head 是 `single`；`modular` 模式则分别生成 path、slot、avoid 三个 head，再用 softmax gate 混合。动作从高斯分布采样后经过 `tanh`，并乘以 `action_mask`。log probability 也只对 active agent 求和。

共享参数带来团队规模泛化的可能，但也有明确假设：不同 agent 使用同一个策略，只能通过节点特征、slot 和图上下文区分角色。如果任务需要异质角色，应该在观测中加入角色类型或改成角色条件 actor，而不是把失败归咎于训练不稳定。

### 10.3 中央 critic 和熵正则

Graph-MASAC 使用联合动作和 pooled graph embedding 构造中央 critic，通常维护双 Q 和 target Q，以降低过估计。目标可概括为

\[
 y=r+\gamma(1-d)\left(\min_{k=1,2}Q_{\bar\theta_k}(s',a')-\alpha\log\pi(a'|s')\right).
\]

actor 通过最大化 Q 与熵的组合更新，`alpha` 可以根据 target entropy 调整，并被限制在 `min_alpha` 与 `max_alpha` 之间。代码还包含目标值裁剪、梯度裁剪、权重范数投影、reward scale EMA 和可选 safety critic。它们是数值稳定措施，不应被描述成新的理论算法，除非有单独的消融和对照。

### 10.4 FlashSAC 命名的解释

代码和检查点默认命名为 `graph_flashsac`，但当前实现的主要结构仍是图编码器、SAC actor/critic、软更新和训练间隔控制。文档中应把它写成“项目内部的 Graph-FlashSAC 配置/实现路径”，不要仅凭命名声称已经提出了全新的 FlashSAC 理论。论文若要使用该名称，需要明确与标准 SAC 的差异、复杂度、消融和可复现代码位置。

## 11. 回放、失败样本和行为克隆

### 11.1 失败感知回放

`MultiGraphReplayBuffer` 除 observation、next observation、联合 action、reward、done 外，还保存 `failure_tag`、`safety_cost` 和 `failure_reason`。训练器在每个 episode 结束或发生近失事件时，根据最小门柱净空、最小机间距离、slot error spike、硬失败原因和 timeout poor progress 生成标签。

当 `failure_replay` 开启时，采样器按配置比例优先抽取失败或近失 transitions，其余从普通样本抽取。它解决的是稀有安全事件在大容量 replay 中被淹没的问题，不等于风险敏感强化学习；若要声称安全优化，还需要明确 cost objective、约束或风险度量。

### 11.2 专家轨迹和 BC

`imitation.py` 使用 `HeuristicFormationReplayController` 或教师 actor 收集示范。每个 episode 会记录是否碰撞、是否越界、是否高度/侧向违规、是否队形塌缩以及是否完成 corridor，然后执行质量和团队规模平衡筛选。正式专家数据不会把碰撞轨迹直接当成高质量 BC 样本；失败片段可以进入 DAgger 或失败回放，但必须在元数据中保留来源。

行为克隆把图观测到动作当作监督学习，使用训练/验证划分并限制 log standard deviation。它的作用是提供 actor warm start，降低纯随机探索一开始撞击门柱的概率；它不是最终策略性能证据。

### 11.3 DAgger 纠正

`dagger.py` 先用专家数据做初始 BC，再让当前 actor rollout；当净空、机间距离、slot error 或 corridor 状态进入风险阈值时，周期性查询教师动作并追加 correction。每轮追加数据后重新 BC，再进入 RL 微调。评估时需要记录每轮新增样本数、风险原因计数、教师来源和最终策略 checkpoint，否则无法判断改进来自数据覆盖还是训练预算增加。

## 12. 训练、评估和检查点的可复核链

### 12.1 训练采样

`multi_gate/training.py` 创建模板环境以获得 observation shape，再创建 `VectorMultiGate2DEnv` 批量采样。团队规模可固定，也可按 uniform buckets 抽样。每步将 batch transition 写入 replay；达到 `learning_starts` 和 `batch_size` 后按 `updates_per_step` 进行更新。episode 完成后只 reset done 的环境，并记录每个 done reason。

### 12.2 检查点和恢复

检查点不仅保存 actor/critic/target/optimizer/alpha，还保存训练步数、算法名、seed、experiment id、观测形状、失败回放和 size invariance 配置。恢复时严格检查 experiment id 和 observation shapes；不兼容时可以只加载 actor 作为 warm start，并重置优化器和熵状态。这样能避免把不同实验的隐状态误拼在一起。

### 12.3 选择门槛

周期性 checkpoint 可以用独立评估 episode 计算 selection score，并刷新 `best_agent.pt` 别名。配置中的 evaluation gate 支持最小成功率、最大碰撞率、最大硬失败率、最小团队规模桶成功率和平均 episode reward。正式报告要同时给出“最终 checkpoint”和“按门槛选中的 checkpoint”，防止只挑训练末期最好的一次。

### 12.4 复现和训练确定性的区别

`aerogate reproduce` 主要验证同一 seed 的 NumPy core 和零动作 rollout 是否一致。它不覆盖 GPU 浮点、PyTorch dropout/采样、并行环境顺序、checkpoint 恢复或训练曲线。因此论文中应将以下内容分开：

- core deterministic replay；
- 固定 checkpoint 的策略评估；
- 多 seed 训练方差；
- resume 后的状态一致性。

## 13. 经典规划器和公平对照

`internal/planners/` 和 `scripts/run_classic_planner_baselines.py` 允许将 A*、Theta*、RRT*、Informed RRT*、heuristic 或强规划器放入相同几何环境。公平比较至少要固定：场景布局、门数量、起终点、团队规模、seed、时间步、碰撞半径、路径平滑规则、最大计算预算和成功判据。

经典规划器通常输出几何路径，而 Graph policy 输出速度动作；两者若要直接比较，必须经过同一个运动学跟踪器和同一个碰撞/终止检查。否则比较的是“路径生成器”与“闭环控制器”两个不同层级。多机任务还要明确经典基线是否有虚拟结构、是否允许共享全局路线、是否使用相同的 safety shield。

建议把结果拆成三张表：单机规划质量、多机协作质量、动态闸门安全质量。不要用一个综合分数把成功率、净空和计算时间混成无法解释的数字。

## 14. Isaac Lab 回放和资产许可

核心环境不需要 Isaac Lab。回放适配器读取任务级状态、slot 和门布局，再创建地面、光照、闸门 USD 和 `five_in_drone.usd` 外壳。它的目的主要是可视化、演示和回放检查，不会自动把二维速度控制变成真实飞行动力学，也不会自动验证相机、传感器或碰撞材质。

`THIRD_PARTY_NOTICES.md` 记录门和无人机视觉资产来自 `kousheekc/isaac_drone_racer`，许可证为 BSD 3-Clause。任何重新分发视频、USD 或改造资产时都应保留版权和许可证文本。README 和论文应使用“Isaac Lab replay/visualization adapter”这样的准确表述，避免写成“真实飞行仿真”。

## 15. 实验设计：怎样把项目推进到论文级证据

### 15.1 最小可行基线

先建立一个不依赖训练的几何基线：固定场景、固定起终点、固定团队规模、零动作和规则 controller 各运行多个 seed。检查运动学、线段碰撞、动态门运动、corridor 和 info 字段。只有这些合同稳定后，才开始训练策略。

### 15.2 因素矩阵

一个可复核的多因素矩阵可以包含：

| 因素 | 建议水平 |
| --- | --- |
| 场景 | 静态门、动态门、动态密度 |
| 闸门数量 | 低、中、高、极限；具体数值由冻结配置给出 |
| 团队规模 | 2、4、8、12 或项目正式桶 |
| 动态速度 | 0、低、中、高；与幅值一起记录 |
| 控制模块 | A* 跟踪、图策略、图策略加 slot、图策略加 shield |
| 训练初始化 | 随机、BC warm start、BC+DAgger |
| agent 顺序 | 默认顺序、随机置换 |
| seed | 至少 3 个用于 smoke，正式结论应增加到足以估计置信区间 |

不要在同一张主表中混合不同团队规模、不同成功定义或不同场景模式。每个单元都要保存 resolved config、manifest 和原始 episode summary。

### 15.3 消融实验

推荐的单变量消融顺序：

1. 去掉图消息传递，只保留逐节点 MLP；
2. 固定团队规模，比较可变规模配置；
3. 去掉 global route，只保留局部 guidance；
4. 去掉 slot，观察队形误差和碰撞变化；
5. 关闭 action shield，只保留奖励惩罚；
6. 关闭 failure replay；
7. 随机打乱 agent 节点顺序；
8. 去掉动态门预测位置；
9. BC、BC+RL、BC+DAgger+RL 三条初始化路径对照。

每次只改变一个开关，其他配置、seed、训练步数和评估脚本保持不变。shield 消融尤其要报告安全失败率和停滞率，因为关闭 shield 后成功率下降并不意外，但 shield 过强也可能使策略几乎不动。

### 15.4 统计报告

每个方法至少给出每个 seed 的 episode 级原始记录，再计算均值、标准差或置信区间。成功率之外，报告失败类型比例、最小净空分布、最小机间距离分布、slot error、路径长度、episode steps、planner latency 和 shield intervention。对极少数 episode 的“100% 成功”不应使用强结论措辞；应明确样本数和附件日期。

### 15.5 失败样本审计

每次失败都应能从 `done_reason` 追溯到发生时的几何状态：团队位置、动态门中心/速度、clearance、pair distance、slot error、shield 介入和当前 waypoint。建议保存失败前后至少若干步的轨迹窗口，并按 collision、no path、corridor、formation、timeout 分类。这样才能区分算法问题、参数问题和测试合同问题。

## 16. 可复现实验命令和检查顺序

在项目根目录执行：

```powershell
uv sync --extra dev
uv run python -m aerogate info
uv run python -m aerogate smoke --scenario single-static --steps 8
uv run python -m aerogate smoke --scenario multi-static --agents 4 --steps 8
uv run python -m aerogate reproduce --scenario multi-static --agents 4 --seeds 3 7 11 --steps 8
uv run python -m pytest
uv run ruff check aerogate tests
uv run ruff format --check aerogate tests
```

推荐顺序是先 `info`，再单机 smoke，再多机 smoke，再 reproduce，最后执行完整测试和静态检查。训练命令不应替代这些准入检查。需要运行动态闸门时，先执行几何 sanity 或对应的 `multi-dynamic` smoke，确认闸门确实运动、门间不重叠、碰撞查询可用，再启动长训练。

## 17. 调试路径：遇到失败先看哪里

### 17.1 一开始就 `no_path`

先检查起终点是否落在边界或障碍膨胀区域，`planner_config.grid_resolution_m` 是否过粗，`safety_margin_m` 与无人机半径是否过大，以及动态场景是否错误地复用了静态障碍图。不要先调 reward；路线不存在时，策略无法补救。

### 17.2 到达目标但被判失败

读取 `done_reason`、`corridor_completed`、`side_bypass_failure`、`corridor_miss_failure`、`height_contract_passed`、`formation_line_collapse_failure` 和最小净空。常见原因是中心到达了，但有 agent 绕过 corridor、slot 超出容差或门柱净空为负。这个结果说明成功合同比“中心点到目标”严格，不能简单改成只看距离。

### 17.3 成功率高但队形很差

检查 progress gate 是否关闭、`goal_requires_slot_tolerance` 是否关闭、slot reward 是否过小，以及 policy 是否使用了 padding 节点。若是 shield 强行把 agent 推回去，还要看 intervention 率；高成功率可能主要来自规则层，而不是 actor 学会协作。

### 17.4 动态门看起来不动

运行 `validate_dynamic_gate_density_geometry`，比较不同时间的 live center、velocity 和 motion range。检查 speed、amplitude 是否为零，是否把静态布局传给动态 profile，或者只渲染了旧缓存。动态门的“视觉上移动”与环境碰撞查询使用同一份时间状态，不能只改回放画面。

### 17.5 训练不稳定或恢复后指标跳变

查看 reward scale EMA、alpha、梯度范数、replay failure ratio、checkpoint metadata 和 observation shapes。warm start 只能加载 actor 时，应明确优化器、critic 和熵状态被重置。不同 `experiment_id`、团队规模上限或 node feature 维度的 checkpoint 不应强行 resume。

## 18. 研究延伸：哪些方向值得继续做

以下是基于现有接口的研究计划，不是当前已完成结果。

### 18.1 置换等变和异质角色

当前图节点有固定区域和 agent 顺序。可以比较随机置换训练、DeepSets/attention pooling、显式 permutation-equivariant message passing，以及角色条件 actor。评价不仅看成功率，还看同一 episode 在 agent 顺序置换后的动作差异和失败类型。

### 18.2 时间图和风险预测

动态门节点当前提供速度和 0.75 s 预测位置。可以加入短历史窗口或时间编码，比较单帧图、堆叠图和 recurrent graph policy，并记录预测误差、提前减速时间和动态碰撞率。需要把预测模型误差纳入 manifest，不能把预测位置当作真实未来。

### 18.3 安全屏蔽的形式化比较

可以将启发式 shield 与控制屏障函数、速度障碍或可达集近似进行对照。比较应包含干预率、停滞率、最小净空、碰撞率和计算延迟；只报告成功率会掩盖 shield 的过度保守。

### 18.4 从二维到更高保真度

如果未来需要连接真实机器人或高保真仿真，应保持任务层接口不变，把 `Kinematics2DUpdater` 替换为带姿态/推力的适配器，并对传感器观测、通信延迟、执行器限幅和坐标变换逐项增加合同测试。二维结果只能作为规划层和协作层基线，不能直接充当迁移证明。

### 18.5 计算预算和扩展性

固定团队规模的成功率不足以说明图策略可扩展。建议测量团队规模、节点数量、消息传递步数与单步 wall-clock 的关系，分别在 CPU 和 GPU 上报告。A* 搜索迭代、图编码器前向、shield 和环境几何检查应分开计时。

## 19. 面向学术申请的准确表述

可以这样介绍项目：

> 我实现了一个可复核的二维固定高度多智能体闸门环境，将受限运动学、圆形障碍碰撞、A* 中心线规划、虚拟结构编队、固定尺寸图观测和动作安全修正拆成独立模块，并用 Graph-SAC/MASAC、行为克隆和 DAgger 形成可选训练路径。项目同时保存失败原因、净空、队形误差、规划延迟和实验 manifest，便于做可变团队规模和动态闸门消融。

不建议写成“完成真实无人机集群飞行”“提出通用安全算法”或“证明图强化学习优于所有方法”。如果展示历史附件结果，应写明场景、seed 数、附件生成日期和“需按当前源码重新核验”。如果尚未完成正式训练或论文实验，应使用“实现了实验框架”“建立了可复现实验入口”“正在进行多 seed 评估”，不要把脚本存在写成实验结论。

## 20. 当前证据边界与历史附件说明

仓库中的 `evaluation_artifacts/report/gate_graph_2d_evaluation_report.md` 和 `results_manifest.json` 保存过一组历史结果，包括单机动态门、多机静态门和多机动态门的少量 episode。它们的正确用法是：

- 作为“曾经运行过的附件级证据”展示实验管线和指标字段；
- 结合 manifest 检查当时的配置、seed、脚本和依赖；
- 在当前源码发生变化后重新运行，确认结果是否仍然成立；
- 在论文中报告样本数和不确定性，而不是只摘录成功率。

尤其不能删除或掩盖历史附件中的 `real_eval_failed`、碰撞或 timeout 记录。失败记录对于定位压力边界有价值；是否修复、如何修复、修复后是否改变任务合同，都应通过新 run 和新 manifest 说明。

本次文档更新没有上传 GitHub，也没有把任何新运行结果写入历史报告。需要正式论文结论时，应另建带时间戳的评估目录，保存原始 episode、resolved config、代码哈希、依赖版本、硬件信息和统计脚本。

## 附录 A：源码到概念的索引

| 研究概念 | 主要源码 | 阅读重点 |
| --- | --- | --- |
| 公开场景 | `aerogate/cli.py`、`aerogate/scenarios.py` | 命令参数如何进入场景工厂 |
| 运动学 | `shared/core/kinematics_2d.py` | 速度/加速度限幅和 dt 积分 |
| 静态碰撞 | `shared/core/collision_2d.py` | 点、线段、signed clearance |
| 动态门 | `shared/core/dynamic_gate_density_2d.py` | live center、velocity、corridor、swept clearance |
| 团队几何 | `shared/core/team_geometry.py`、`multi_gate/formation/virtual_structure.py` | slot、横向 band、队形误差 |
| 多机环境 | `multi_gate/env/multi_gate_env.py` | reset/step 的调用顺序 |
| 图观测 | `multi_gate/env/observation_multi.py` | 节点类型、尺度、邻接、mask |
| 安全屏蔽 | `multi_gate/env/safety_shields.py` | 速度空间的排斥、制动和边界修正 |
| 奖励终止 | `multi_gate/rewards/multi_agent_rewards.py` | progress gate、终止原因、分项奖励 |
| 图策略 | `multi_gate/graph_rl/graph_policy.py` | 消息传递、池化、共享 actor、tanh 高斯 |
| 回放 | `multi_gate/graph_rl/replay_buffer.py` | failure tag、safety cost 和采样比例 |
| 模仿学习 | `multi_gate/imitation.py`、`multi_gate/dagger.py` | 专家筛选、BC、风险纠正 |
| 训练管理 | `multi_gate/training.py` | 向量采样、checkpoint、评估门槛、恢复 |
| 可视化 | `assets/`、`*_isaaclab.py` | 二维几何到 USD 回放的边界 |
| 质量控制 | `tests/`、`evaluation_artifacts/` | 回归、复现和历史证据 |

## 附录 B：术语表

- **agent**：环境中的一个受控平面运动体；不等同于真实无人机硬件。
- **virtual structure**：把团队中心和 heading 转换为相对 slot 的几何编队模型。
- **slot error**：当前位置与期望 slot 的距离统计。
- **clearance**：考虑门柱和 agent 半径后的带符号净空。
- **swept clearance**：一个时间步内 agent 与移动门柱相对运动的近似最小净空。
- **corridor**：动态闸门区域内限制侧向绕行的横向空间合同。
- **node mask**：图中真实节点与 padding 节点的区分。
- **action mask**：当前团队规模下哪些 agent 的动作有效。
- **failure replay**：按碰撞、近失、队形或进度标签提高风险样本抽样概率。
- **Graph-MASAC**：使用图编码观测和集中式联合动作 critic 的多智能体 SAC 实现路径。
- **Isaac Lab replay**：把任务状态映射到 USD 场景进行可视化或回放，不等于核心环境的物理验证。
