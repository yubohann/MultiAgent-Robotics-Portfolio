# AeroGate Graph: Research Framing and Technical Explanation

## 1. What the project is

**English**

AeroGate Graph is a modular research environment for coordinating multiple planar agents through a field of gates. The core task is a fixed-height, two-dimensional model: each agent moves on the same horizontal plane, receives a planar velocity command, and is subject to speed, acceleration, boundary, gate-post, and inter-agent constraints. I chose this abstraction so that I could change the coordination and policy assumptions while keeping the geometry, success definition, and diagnostics fixed. The project is a controlled testbed for planning, formation coordination, graph-structured observations, and geometric safety intervention. Its clean layer boundaries make it easy to compare design choices and trace a result back to a concrete mechanism.

**中文**

AeroGate Graph 是一个用于研究多个平面运动体如何协同通过闸门场景的模块化研究环境。核心任务是固定高度的二维模型：每个智能体都在同一水平面运动，接收平面速度命令，并受到速度、加速度、边界、门柱和智能体间距离等约束。我选择这个抽象，是为了在保持几何、成功定义和诊断字段不变的情况下，逐项改变协同和策略假设。它是用于研究规划、编队协同、图结构观测和几何安全干预的受控测试平台。清晰的模块边界便于比较设计选择，并把结果追溯到具体机制。

## 2. The research question

**English**

The central question is how a team with a changing number of agents can make progress while preserving a usable formation and respecting gates, boundaries, and teammate separation. I separate four decisions: where the team reference point should travel, where each robot should be relative to that reference, how relational state is represented, and how risky velocity commands are corrected. These decisions define a comparison matrix covering graph observations, desired formation positions, safety filtering, static gates, moving gates, and gate density.

**中文**

核心问题是：当团队规模变化时，团队如何在保持可用编队的同时持续前进，并满足门柱、边界和机间分离约束。我把问题拆成四个决策：团队参考点往哪里走、每个机器人相对参考点应处于什么位置、关系状态如何表示，以及有风险的速度命令如何被修正。由此形成比较矩阵，覆盖图观测、期望编队位置、安全滤波、静态闸门、动态闸门和门密度。

## 3. Overall architecture

**English**

The main execution path is:

```text
scenario configuration
        -> gate layout and obstacle geometry
        -> global route for the team reference point
        -> virtual-structure formation positions
        -> graph-structured observation
        -> policy action
        -> velocity-space safety shield
        -> acceleration and speed limits
        -> position integration
        -> collision, clearance, formation, reward, and termination checks
```

The repository keeps these layers separate. The `shared/core` package contains the two-dimensional geometry and kinematics. `multi_gate` contains the multi-agent environment, planner, formation, observations, guidance, rewards, and safety filter. `multi_gate/graph_rl` contains the graph actor and centralized-critic multi-agent SAC path. The public `aerogate` package provides scenario registration, smoke tests, and deterministic core-environment reproduction. The portfolio includes single-agent experiments and multi-agent experiments with teams from one to eight agents. The dynamic gate-density curriculum uses eight agents. The graph interface also supports larger padded team capacities when a preset requires them.

**中文**

主要执行链是：场景配置生成闸门布局和障碍几何；全局规划器为团队中心生成路线；虚拟结构生成期望编队位置；环境构造图观测；策略输出动作；安全屏蔽在速度空间修正动作；运动学模块施加加速度和速度限制并积分位置；最后检查碰撞、净空、编队、奖励和终止状态。

仓库把这些层次分开。`shared/core` 负责二维几何和运动学；`multi_gate` 负责多机环境、规划、编队、观测、引导、奖励和安全滤波；`multi_gate/graph_rl` 负责图 actor 和集中式 critic 的多智能体 SAC 路径；公共 `aerogate` 包负责场景注册、烟测和核心环境确定性复现。作品集同时包含单机实验，以及 1 到 8 个 agent 的多机团队实验；动态门密度 curriculum 使用 8 个 agent，其他 preset 也支持更大的 padding 团队容量。

## 4. Scene and dynamics

**English**

The default shared configuration uses a time step of `0.1 s`, a planar maximum speed of about `3.5 m/s`, and a planar acceleration limit of about `2.45 m/s^2`. The exact values can change in experiment presets, so an experiment should always record its resolved configuration. The state of one agent is planar position, planar velocity, yaw, and time. The action is converted into a desired planar velocity. The kinematics then clip the command, limit the change in velocity, and integrate the new position. Yaw is aligned with the velocity when the speed is above a small deadband. This gives the project a compact velocity-controlled kinematic model for studying coordination at a fixed flight height.

Static gates are represented by circular gate-post obstacles. Collision checking combines point clearance with the complete movement segment, which gives the environment a consistent geometric collision contract. Dynamic-gate profiles update gate centers and velocities over time and additionally evaluate relative swept clearance during a time step.

**中文**

共享默认配置使用 `0.1 s` 的时间步、约 `3.5 m/s` 的平面最大速度和约 `2.45 m/s²` 的平面加速度限制。具体实验 preset 可能会改这些值，因此正式实验必须记录最终解析后的配置。单个智能体的状态包括平面位置、平面速度、航向角和时间。动作先被转换为期望平面速度，然后经过速度裁剪、速度变化量限制和位置积分；当速度高于一个小阈值时，航向与速度方向对齐。这样，项目就得到一个紧凑的固定高度速度控制运动学模型，用于研究团队协同。

静态闸门用圆形门柱障碍表示。碰撞检查同时使用点净空和完整运动线段，形成一致的几何碰撞合同。动态闸门会随时间更新中心和速度，并在每个时间步计算相对运动的 swept clearance。

## 5. Global route and formation geometry

**English**

The global route planner runs A* on an obstacle-inflated two-dimensional grid with an eight-neighbor connectivity pattern. The heuristic is Euclidean distance. After the grid path is reconstructed, it is converted back to world coordinates and simplified when a longer segment is collision-free. The planner output is a sequence of waypoints for the **team reference point**. In the static-gate track, the obstacle map used by A* contains the inflated gate posts. In the moving-gate density track, the moving posts are updated at runtime, while the route remains a geometric reference. Dynamic-gate response comes from runtime geometry, guidance, and the velocity-level safety filter.

The virtual-structure model provides the second layer. It creates **desired formation positions** around the route reference point and rotates those offsets into the world frame using the current heading. In the code these positions are stored as `slots`; in conversation I call them formation positions. The default spacing is approximately `2.4 m` laterally and `3.0 m` longitudinally, with formation presets available for other layouts. Small teams use fewer columns, while larger teams use more columns. Each agent therefore receives a geometric tracking target, and the global planner stays focused on the team route.

The division of responsibility is important: **A* decides where the team reference point should go; the virtual structure decides how the individual agents should be arranged around that route.** This decomposition turns joint planning into a route problem plus a formation-tracking problem, with a clear diagnostic path for each part.

**中文**

全局路线规划器在经过障碍膨胀的二维网格上运行 A*，使用八邻域连接和欧氏距离启发函数。路径重建后转换回世界坐标；更长线段通过碰撞检查后可用于 waypoint 简化。规划器输出的是**团队参考点**的 waypoint 序列；各个 agent 通过编队和引导层跟踪这些 waypoint。在静态闸门轨道中，A* 使用包含膨胀门柱的障碍地图；在动态门密度轨道中，移动门柱在运行时更新，路线作为几何参考。动态门响应来自运行时几何、引导和速度级安全滤波器。

虚拟结构模型负责第二层。它在路线参考点周围生成**期望相对编队位置**，再根据当前航向把这些偏移旋转到世界坐标。代码中这些位置存储为 `slots`，面试时我会直接称为 formation positions。默认横向间距约为 `2.4 m`，纵向间距约为 `3.0 m`，不同 formation preset 可以使用其他布局。小团队使用较少列数，大团队使用更多列数。这样每个 agent 都有明确的几何跟踪目标，全局规划器专注于团队路线。

两者的分工是理解项目的关键：**A* 决定团队参考点往哪里走，虚拟结构决定各个智能体如何排列在这条路线上。** 这种分解把联合规划转成路线问题和编队跟踪问题，每一部分都有清晰的诊断路径。

## 6. Graph observation design

**English**

The multi-agent observation is a fixed-size graph with four arrays: `node_features`, `adjacency`, `node_mask`, and `action_mask`. The graph is padded to a maximum capacity, and masks distinguish active entities from padding. This gives one policy interface a stable tensor shape across team sizes. The portfolio includes single-agent experiments and teams from one to eight agents. The dynamic gate-density curriculum uses eight agents; other presets can select a larger padded capacity.

The graph contains six node groups:

1. **Agent nodes.** These include position relative to the virtual center, velocity, error to the agent's desired formation position, local obstacle clearance, task progress, and the normalized team size.
2. **Formation-position nodes.** These describe the desired relative positions, their heading context, position index, and progress.
3. **Waypoint nodes.** These describe look-ahead points on the global route, including relative position, heading, waypoint order, and progress.
4. **Obstacle or dynamic-gate nodes.** These include relative position, obstacle radius, dynamic velocity, a short predicted position for moving posts, and a clearance-related feature.
5. **Optional guidance nodes.** These can carry a guidance target, heading, risk or speed information, and confidence.
6. **A global/team node.** This summarizes the virtual center, heading, overall clearance, and progress.

The adjacency is mainly distance-based: active nodes have self-connections, and pairwise weights decay exponentially with distance. The encoder first maps each node feature through an MLP, performs several masked message-passing rounds, and mean-pools valid nodes into a graph embedding. Padding nodes are excluded by `node_mask`. The actor then combines each agent embedding with the pooled team embedding and produces one two-dimensional action per possible agent. `action_mask` removes actions for inactive agents.

Padding and masks give the policy a variable-size **interface**. The current implementation keeps an ordered agent block and maps actions by agent index, which makes the interface straightforward to debug. Team-size and agent-order studies can be organized directly around this interface by selecting training and evaluation buckets.

**中文**

多机观测是一个固定尺寸图，由 `node_features`、`adjacency`、`node_mask` 和 `action_mask` 四个数组组成。图被填充到最大容量，mask 区分 active 实体和 padding 节点，从而让不同团队规模共享稳定的张量形状和策略接口。作品集中包含单机实验，以及 1 到 8 个 agent 的多机团队实验；动态门密度 curriculum 使用 8 个 agent，其他 preset 可以选择更大的 padding 容量。

节点主要分为六类：智能体节点，包含相对团队中心的位置、速度、到期望编队位置的误差、局部障碍净空、任务进度和归一化团队规模；编队位置节点，描述期望相对位置、航向、位置索引和进度；waypoint 节点，描述全局路线上的前视点；障碍物或动态门节点，包含相对位置、半径、动态速度、短时预测位置和净空特征；可选的 guidance 节点，包含引导目标、航向、风险或速度信息和置信度；以及表示团队中心、航向、整体净空和进度的全局节点。

邻接主要由距离决定：有效节点有自连接，节点之间的权重随距离指数衰减。编码器先用 MLP 编码节点，再进行多轮带 mask 的消息传递，并对有效节点做平均池化得到图嵌入。padding 节点通过 `node_mask` 排除。actor 将每个 agent 的节点嵌入与团队级 pooled embedding 拼接，为每个可能的 agent 输出二维动作；`action_mask` 让无效 agent 不产生有效动作。

这里可以把设计理解为：padding 和 mask 提供可变规模的**接口**；当前实现使用有序的 agent block，并按 agent index 对应动作，便于调试，也便于组织团队规模与 agent 顺序实验。

## 7. Observation assumptions

**English**

The current AeroGate core uses simulator geometry and kinematic state as the observation source, including nearby obstacles and route context. This creates a clean privileged-state coordination setting for isolating planning and policy effects. The same observation interface is ready for local, delayed, or noisy sensor features in future perception-driven studies.

**中文**

当前 AeroGate 核心环境以仿真几何和运动学状态作为观测来源，包括附近障碍物和路线信息。这构成了一个干净的特权状态协同设置，便于单独分析规划和策略效果。同一观测接口也为后续加入局部、延迟或带噪传感器特征预留了位置。

## 8. Graph-encoded SAC and centralized-critic training

**English**

The actor is a shared-parameter policy with a graph encoder. Sharing the actor across agents gives every agent the same policy parameters while the graph input carries its local and team context. The policy head outputs a Gaussian mean and log standard deviation for each agent, samples with the reparameterization trick, applies `tanh`, and masks inactive agents. In this project, “Graph-SAC” means SAC equipped with this graph encoder and masked multi-agent interface.

The multi-agent training path uses centralized training with a shared actor and a joint-action critic. Each critic receives the pooled graph state, the masked joint action, and the action mask. The implementation maintains two Q critics and target critics, following the usual SAC pattern. The target value uses the minimum of the two target Q values and the entropy term. Optional safety-critic, behavior-cloning, failure-replay, and DAgger paths extend the same training interface. The setup follows the centralized-training/decentralized-execution pattern: training evaluates joint actions, while execution produces one action for each active agent from the shared graph policy.

The reason for using a centralized critic is credit assignment. During training, the critic evaluates the joint action in the context of the whole team. During execution, each active agent receives its action from the shared graph actor. This lets the project study how relational observations and joint-value estimation interact in the same controlled environment.

**中文**

actor 是带图编码器的共享参数策略。所有 agent 共享 actor，同时图输入携带各自的局部关系和团队上下文。策略头为每个 agent 输出高斯分布的均值和对数标准差，用重参数化采样，再通过 `tanh` 限制动作范围，并屏蔽无效 agent。在本项目中，“Graph-SAC”指配备这一图编码器和带 mask 多机接口的 SAC 实现。

多机训练路径使用共享 actor 和联合动作 critic 的集中式训练。每个 critic 接收 pooled graph state、经过 mask 的联合动作和 action mask。实现中维护两个 Q critic 以及对应的 target critic，遵循 SAC 的常见结构；目标值使用两个 target Q 的较小值并加入熵项。可选的 safety critic、behavior cloning、failure replay 和 DAgger 路径在同一训练接口上扩展。整体采用 centralized training / decentralized execution 模式：训练时评价联合动作，执行时由共享图策略为每个 active agent 产生动作。

使用集中式 critic 的主要理由是改善多智能体 credit assignment：训练时 critic 在整个团队上下文中评价联合动作，执行时每个 active agent 仍由共享图 actor 产生动作。这样可以在同一受控环境中研究关系观测与联合价值估计如何共同影响协同。

## 9. Learning and data pipelines

**English**

The learning side has four connected paths. The first is Graph-SAC for the graph-encoded actor. The multi-agent path uses Graph-MASAC, also exposed in the repository as Graph-FlashSAC. It keeps a shared actor, a centralized joint-action critic, twin Q estimates, target critics, entropy tuning, masked actions, reward-scale tracking, gradient clipping, and soft target updates. `flash_update_interval` controls the update schedule used by the lightweight FlashSAC loop. A modular actor-head option gives separate proposal heads for route following, formation tracking, and avoidance, followed by learned soft gates.

Behavior cloning (BC) provides a supervised actor warm start. `collect_expert_demonstrations` runs the `HeuristicFormationReplayController` or a teacher actor checkpoint, records graph observations and joint actions, and saves episode metrics such as progress, pair distance, formation-position error, clearance, and termination reason. The dataset selector balances team sizes and filters episodes using the recorded quality fields. `run_actor_behavior_cloning` trains the shared actor with masked mean-squared action loss, a validation split, and an optional log-standard-deviation penalty. The output is an actor checkpoint that can seed the reinforcement-learning stage.

DAgger adds corrections on policy-visited states. At each iteration, the current actor rolls out the environment; low clearance, small pair distance, large formation-position error, height-contract events, corridor events, and hard termination reasons trigger a teacher query. The teacher supplies the action for that state, the correction is appended to the BC dataset, and the actor is retrained for a few epochs. The combined routine then fine-tunes the corrected actor with Graph-FlashSAC. This gives the training pipeline a direct way to concentrate data around states where the current policy needs guidance.

The replay buffer stores graph observations, joint actions, rewards, done flags, failure tags, failure reasons, and safety costs. With failure replay enabled, a configured fraction of each batch comes from tagged transitions such as gate-post collision, agent collision, out-of-bounds, formation collapse, near-miss clearance, low pair distance, or a sharp formation-position error. An optional centralized safety critic predicts discounted safety cost and adds a penalty to the actor objective. A behavior-anchor loss can keep the actor close to a reference or demonstration action on selected transitions. The dynamic gate-density curriculum changes gate count, moving-gate speed and amplitude, train/evaluation seeds, and controller parameters stage by stage. The speed schedule spans approximately `1.15–3.50 m/s`, paired with acceleration limits from `0.75–2.45 m/s^2`; the reference multi-agent setting uses eight agents.

**中文**

学习部分有四条相互衔接的路径。第一条是图编码 actor 的 Graph-SAC。多机路径使用 Graph-MASAC，在仓库中也以 Graph-FlashSAC 的形式提供。它包含共享 actor、集中式联合动作 critic、双 Q 估计、target critic、熵系数调节、动作 mask、奖励尺度跟踪、梯度裁剪和软更新；`flash_update_interval` 控制轻量 FlashSAC 循环的更新节奏。模块化 actor head 可以分别生成路线跟踪、编队跟踪和避障倾向，再通过学习到的 soft gate 进行组合。

Behavior cloning（BC）用于 actor 的监督式 warm start。`collect_expert_demonstrations` 运行 `HeuristicFormationReplayController` 或教师 actor checkpoint，记录图观测和联合动作，并保存进度、机间距离、编队位置误差、净空和终止原因等 episode 指标。数据集选择器按团队规模平衡样本，并依据这些质量字段筛选 episode。`run_actor_behavior_cloning` 使用带 mask 的动作均方误差、验证集划分和可选的 log-standard-deviation 惩罚训练共享 actor，输出可用于强化学习阶段的 actor checkpoint。

DAgger 负责补充策略访问到的风险状态。每轮中，当前 actor 先在环境中 rollout；当净空偏低、机间距离偏小、编队位置误差偏大，或出现高度合同、走廊事件和硬终止原因时，系统查询教师。教师为该状态提供动作，修正样本追加到 BC 数据集，actor 再训练若干 epoch。组合流程最后用 Graph-FlashSAC 对修正后的 actor 做微调。这样，训练数据会直接集中到当前策略最需要指导的状态区域。

Replay buffer 保存图观测、联合动作、奖励、done 标记、failure tag、失败原因和 safety cost。启用 failure replay 后，每个 batch 的一部分来自带标签 transition，例如门柱碰撞、智能体碰撞、越界、编队塌缩、近失净空、机间距离过小或编队位置误差突增。可选的集中式 safety critic 预测折扣安全代价，并把惩罚加入 actor 目标。behavior-anchor loss 可以让 actor 在指定 transition 上贴近参考策略或示范动作。动态门密度 curriculum 分阶段改变门数量、移动门速度和振幅、训练/评估 seed 以及控制器参数；速度约从 `1.15–3.50 m/s` 递增，对应加速度限制约为 `0.75–2.45 m/s²`；参考多机设置使用 8 个 agent。

## 10. Dynamic-gate runtime and supporting tools

**English**

The dynamic-gate runtime generates a reproducible gate field from a seed. The density line supports gate counts up to 60. Sparse settings use the center lane first; denser settings add two side lanes and distribute columns across the three lanes. Each gate stores a base center, yaw, motion phase, motion mode, lane index, and column index. Live centers are computed from time, amplitude, and speed. The runtime derives gate velocity from current and next-frame centers, builds a live obstacle map with velocity on every post, and caches these values per environment step. A projection pass keeps neighboring gate posts separated by the configured geometric margin.

Near a moving gate, the task-first formation module predicts the gate center at the estimated arrival time. It blends the normal formation positions toward a compressed pass arrangement, assigns agents to three lateral lanes, orders them longitudinally, and limits the pass offsets to the opening geometry. The resulting positions feed both the heuristic expert controller and the actor observation. This is a concrete design choice: the policy sees a formation target that reflects the next gate's predicted location, while the safety layer still checks the live geometry during motion.

The optional route-guidance path is a low-frequency strategic interface. `RouteGuidanceEngine` receives a compressed state and A* corridor hints, then requests one JSON object containing a relative target, heading, risk level, formation compactness, speed scale, mode code, and confidence. The local HTTP client supports asynchronous requests, caching, timeout metadata, and sanitized fallbacks. The guidance path proposes high-level context; the environment and policy retain the low-level velocity and safety responsibilities. The default gate-2D experiments use the deterministic geometric path, while the guidance engine remains available for controlled studies.

Training uses `VectorMultiGate2DEnv` to step several independent environments through one synchronous API. It batches graph observations and joint actions, resets completed rows with per-environment seeds, and returns one info record per environment. The training manager records TensorBoard scalars, checkpoints, evaluation summaries, replay summaries, and a training signature containing observation shapes and configuration identifiers. Best-checkpoint promotion is based on an evaluation gate with success, collision, hard-failure, and optional reward thresholds.

Isaac Lab appears in a separate replay and live-preview path. A saved trajectory summary is mapped to the gate and drone assets for visual inspection, while the NumPy environment remains the source of task state, metrics, and termination reasons. This separation lets me inspect geometry and policy behavior with the same manifest across CPU rollouts and rendered replays.

**中文**

动态闸门运行时根据 seed 生成可复现的闸门场景，密度线支持最多 60 个 gate。稀疏设置先使用中心车道；密集设置再加入两条侧车道，并把列分布到三条车道上。每个 gate 保存基础中心、yaw、运动相位、运动模式、车道索引和列索引。实时中心由时间、振幅和速度计算得到；运行时用当前帧与下一帧中心差分得到 gate velocity，为每个门柱构造带速度的 live obstacle map，并按环境 step 缓存这些值。投影步骤保证相邻门柱之间满足配置的几何间距。

接近移动门时，task-first formation 模块根据预计到达时间预测门中心。它把常规编队位置逐渐混合到压缩后的通过编队，将 agent 分配到三条横向车道，再按纵向顺序排列，并依据门洞几何限制通过偏移。生成的位置同时送入启发式专家控制器和 actor 观测。这是一个具体设计：策略看到的是结合下一道门预测位置的编队目标，运动过程中安全层继续检查实时几何。

可选的 route-guidance 路径提供低频战略接口。`RouteGuidanceEngine` 接收压缩状态和 A* corridor 提示，请求一个包含相对目标、航向、风险等级、编队紧凑度、速度比例、模式编码和置信度的 JSON。local HTTP client 支持异步请求、缓存、超时元数据和经过清洗的 fallback。guidance 负责高层上下文，环境和策略继续负责底层速度与安全。默认 gate-2D 实验使用确定性的几何路径，guidance engine 可用于受控对照研究。

训练通过 `VectorMultiGate2DEnv` 用一个同步接口推进多个相互独立的环境。它批量组织图观测和联合动作，为完成的环境行按独立 seed 重置，并返回每个环境的一条 info 记录。训练管理器记录 TensorBoard 标量、checkpoint、评估摘要、replay 摘要和包含观测形状及配置标识的 training signature。最佳 checkpoint 按评估 gate 选择，评估 gate 可以使用成功率、碰撞率、硬失败率和可选的回报阈值。

Isaac Lab 位于独立的 replay 和 live-preview 路径。保存的轨迹摘要会映射到 gate 和 drone 资产，用于视觉检查；NumPy 环境继续提供任务状态、指标和终止原因。这样，CPU rollout 和渲染回放可以共享同一份 manifest。

## 11. Velocity-level safety filter

**English**

The repository calls this component an action shield. Operationally, it is a heuristic velocity-level safety filter that runs before kinematic integration: the policy action is converted into desired planar velocities, configured corrections are applied, and the corrected velocities are converted back into actions. Depending on the preset, the correction order includes post-gate cruise handling, pairwise agent separation, boundary braking and inward correction, route or corridor correction, dynamic-gate channel correction, and static or dynamic obstacle repulsion and braking.

For moving gates, the filter uses the current gate positions and velocities for a short-horizon kinematic extrapolation. It can reduce forward speed near a narrow aperture and apply lateral corrections when an agent drifts toward the edge. The environment performs segment and swept-collision checks after the filter, giving both pre-action intervention and post-motion diagnostics. The dynamic swept check uses linearly interpolated relative motion over one simulation step.

The filter records diagnostics such as mean intervention norm, pair closeness, boundary closeness, obstacle closeness, guidance closeness, and gate-channel closeness. These fields quantify how often and how strongly the geometric safety layer changes the policy command, which makes safety-versus-progress trade-offs measurable.

**中文**

仓库把这个组件称为 action shield。就实际运行而言，它是一个启发式的速度级安全滤波器，在运动学积分之前工作：先把策略动作转换成期望平面速度，再按配置施加修正，最后把修正后的速度转换回动作。根据 preset，修正顺序可以包括过门后的巡航处理、智能体间距修正、边界制动和向内修正、路线或走廊修正、动态门通道修正，以及静态或动态障碍物的排斥和制动。

对于动态闸门，滤波器使用当前门的位置和速度做短时域运动学外推。在开口较窄时，它可以降低前向速度；当智能体偏向开口边缘时，它可以施加横向修正。滤波器之后，环境继续执行整段运动和 swept collision 检查，从而同时保留动作前干预和运动后诊断。动态 swept 检查基于一个仿真步内相对运动的线性插值。

滤波器会记录平均干预范数、智能体接近程度、边界接近程度、障碍物接近程度、引导接近程度和门通道接近程度等诊断。这些字段量化安全层改变策略命令的频率和幅度，使安全与进度之间的权衡可以被测量。

## 12. Reward and termination contract

**English**

The reward is deliberately multi-objective. It combines route progress, goal proximity, improvement in formation-position tracking, formation-position error, guidance tracking, boundary proximity, obstacle clearance, inter-agent separation, formation-line collapse, action magnitude, action smoothness, survival, and terminal terms. Progress is gated by formation, boundary, separation, and guidance conditions, so team progress reflects the state of the whole group.

The termination logic uses a task-level success contract. Depending on the scene, an episode can end because of a fixed-height geometry-contract violation, side bypass, corridor miss, gate-post collision, agent collision, formation-line collapse, goal reached, or out of bounds. A timeout is reported separately as truncation. Goal completion can require the team reference point to be within the goal radius, the formation-position error to be within tolerance, and the minimum obstacle clearance to remain positive.

**中文**

奖励是有意设计成多目标形式的，结合路线进度、目标接近、期望编队位置跟踪改善、编队位置误差、引导跟踪、边界接近、障碍物净空、智能体间分离、编队线塌缩、动作幅度、动作平滑性、生存奖励和终止项。进度奖励受到编队、边界、分离和引导条件调制，因此团队进度反映整个群体的状态。

终止逻辑采用任务级成功合同。根据场景，episode 可能因固定高度几何合同违规、侧向绕行、走廊遗漏、门柱碰撞、智能体碰撞、编队线塌缩、到达目标或越界而结束；timeout 会单独作为 truncation 报告。成功通常要求团队参考点进入目标半径、编队位置误差满足容差，并且最小障碍物净空为正。

## 13. Evaluation evidence in the repository

**English**

The repository contains evaluation artifacts generated on 2026-06-21. Each row is tied to a scenario, sample count, status, and method label. The `ours_mainline` label identifies the recorded mainline configuration; checkpoint and training metadata are kept in the experiment provenance. The evaluation scripts expose A*, Theta*, RRT*, Informed RRT*, and a heuristic controller as classical reference methods, so route quality and learned-policy behavior can be read against recognizable planning families.

One retained single-agent dynamic 42-gate row labelled `ours_mainline` contains 10 seeds, a 100% success rate, 0% collision rate, mean path length of `83.388 m`, mean flight time of `72.62 s`, mean minimum clearance of about `0.052 m`, and mean moving-gate swept clearance of about `0.049 m`. In the same 42-gate table, A* reports 10% success and 80% collision. These values are historical artifact measurements tied to the recorded mainline configuration and comparison rows.

The same artifact shows a clear density boundary for the mainline row: at 48, 54, and 60 dynamic gates, the status is `real_eval_failed`. At 60 gates, the row reports 0% success and 60% collision over 10 seeds. These rows identify the stress regime of the recorded configuration.

For the retained multi-agent table, each condition contains three episodes. It reports 100% success and 0% collision for static 60 gates, dynamic 36 gates, and dynamic 60 gates; dynamic 42 and 54 gates each report 66.7% success and 33.3% collision. The episode count is part of the result description.

The public `reproduce` command serves as a core-environment repeatability check. It resets and steps the same NumPy rollout twice for the same seed, usually with zero actions, and compares the diagnostics. A deterministic result confirms that the geometry, dynamics, and diagnostic path are repeatable under that runtime.

**中文**

仓库中保存了生成日期为 2026-06-21 的评测附件。每一行都应与场景、样本量、状态和方法标签一起阅读。`ours_mainline` 表示记录中的主方法配置；checkpoint 和训练元数据由实验 provenance 单独保存。评测脚本提供 A*、Theta*、RRT*、Informed RRT* 和启发式控制器作为经典参考方法，便于把路线质量和学习策略行为放到熟悉的规划方法谱系中比较。

一条保留的单机动态 42 门 `ours_mainline` 记录包含 10 个 seed，成功率 100%，碰撞率 0%，平均路径长度 `83.388 m`，平均飞行时间 `72.62 s`，平均最小净空约 `0.052 m`，平均移动门 swept clearance 约 `0.049 m`。同一张 42 门表中的 A* 成功率为 10%，碰撞率为 80%。这些是记录中主方法配置及其对照行的历史附件测量值。

同一附件显示了主方法在高密度场景下的密度边界：48、54 和 60 个动态门的状态都是 `real_eval_failed`。其中 60 门、10 个 seed 的记录成功率为 0%，碰撞率为 60%。这些行标记了记录配置进入压力区间的运行范围。

保留的多机表格每个条件包含 3 个 episode。它记录了静态 60 门、动态 36 门和动态 60 门的成功率 100%、碰撞率 0%；动态 42 门和 54 门各为成功率 66.7%、碰撞率 33.3%。同时标明 episode 数，就能把这些测量值与其评测规模对应起来。

公共 `reproduce` 命令用于核心环境重复性检查。它通常使用零动作，对同一个 seed 的 NumPy rollout 重复 reset 和 step，并比较诊断结果。确定性结果确认该运行时下几何、动力学和诊断路径具有重复性。

## 14. Research and engineering value

**English**

The central value of AeroGate Graph is the separation of a multi-agent robotics problem into inspectable contracts. Geometry defines collision and clearance. A* defines the team-level route. The virtual structure defines formation targets. The graph observation defines how variable-size relational state reaches the policy. The shield defines a pre-integration intervention point. Rewards, termination reasons, and manifests turn each run into a traceable engineering record.

For an interview, I describe it as an implemented research framework and a controlled testbed for graph-based coordination. Its current scope is fixed-height planar motion, with single-agent and 1–8-agent experiments, dynamic-gate stress cases, and explicit geometric diagnostics. A publishable study of graph-policy effects would build on this foundation with a frozen multi-seed protocol and component ablations.

**中文**

AeroGate Graph 的核心价值，是把多智能体机器人问题拆成可以独立检查的合同：几何层定义碰撞和净空；A* 定义团队级路线；虚拟结构定义编队目标；图观测定义可变规模关系状态如何进入策略；安全屏蔽定义积分前的干预点；奖励、终止原因和 manifest 把每次运行变成可追溯的工程记录。

面试中可以把它介绍为一个已经实现的、用于图协同研究的受控测试平台。当前范围是固定高度平面运动，包含单机和 1–8 机实验、动态闸门压力场景以及明确的几何诊断。后续关于图策略影响的论文实验，可以在这套基础上冻结多 seed 协议并加入模块消融。

## 15. A concise spoken explanation

**English**

“AeroGate Graph is a fixed-height 2D multi-agent gate environment I built to study coordination under route, formation, and safety constraints. I tested single-agent cases and teams from one to eight agents, with the dynamic gate-density curriculum using eight agents. A* gives the team reference point a geometric route. A virtual structure turns that route into desired formation positions for the individual agents. I encode agent state, formation positions, look-ahead waypoints, nearby obstacles, moving-gate information, and team state as a masked graph. A shared graph actor outputs planar velocity actions, and a centralized critic evaluates the joint action during training. Before integration, a velocity-level safety filter adjusts pairwise, boundary, gate-channel, and obstacle risks. The environment records progress, clearance, pair distance, formation error, planner latency, and termination reasons. My current evidence comes from fixed-height simulation artifacts, which let me analyze the design and its operating range in a controlled way.”

**中文**

“AeroGate Graph 是我构建的固定高度二维多智能体闸门环境，用来研究路线、编队和安全约束下的协同。我测试了单机，以及 1 到 8 架无人机的团队；动态门密度 curriculum 使用 8 个 agent。A* 为团队参考点生成几何路线，虚拟结构把路线转换成每个 agent 的期望编队位置。我把 agent 状态、编队位置、前视 waypoint、附近障碍物、动态门信息和团队状态表示成带 mask 的图。共享图 actor 输出平面速度动作，集中式 critic 在训练时评价联合动作。积分前，速度级安全滤波器修正机间、边界、门通道和障碍风险。环境记录进度、净空、机间距离、编队误差、规划延迟和终止原因。当前证据来自固定高度仿真附件，这些数据支持我在受控条件下分析设计和运行范围。”

## 16. Interview framing

**English**

Core sentence: “I implemented a modular fixed-height 2D research environment with graph observations, A* team-route planning, desired formation positions, configurable velocity-level safety filtering, and Graph-SAC/Graph-MASAC training paths.”

Evidence sentence: “The repository contains single-agent and 1–8-agent runs, dynamic-gate stress cases, and evaluation artifacts with explicit seeds, episode counts, clearance, collision, and termination fields.”

**中文**

核心句：“我实现了一个模块化的固定高度二维研究环境，包含图观测、A* 团队路线规划、期望编队位置、可配置速度级安全滤波，以及 Graph-SAC/Graph-MASAC 训练路径。”

证据句：“仓库包含单机和 1–8 机运行、动态闸门压力场景，以及带有 seed、episode 数、净空、碰撞和终止字段的评测附件。”

## 17. Literature grounding

**English**

The literature map behind the implementation is compact and direct. Hart, Nilsson, and Raphael introduced A* as a cost-plus-heuristic search method; in AeroGate, that idea appears as the team-reference route on an inflated 2D obstacle grid. Daniel, Nash, Koenig, and Felner introduced Theta* for any-angle grid paths, while Karaman and Frazzoli established the sampling-based RRT* line. These methods define the classical planning references used by the evaluation scripts.

Haarnoja et al. introduced Soft Actor-Critic (ICML 2018), which supplies the maximum-entropy continuous-control objective used by the actor and twin critics. Lowe et al. (NeurIPS 2017) provide the centralized-training/decentralized-execution pattern for multi-agent actor-critic methods. Jiang et al. (ICLR 2020), Tolstaya et al. (IROS 2021), and H2GNN by Zhang et al. (IEEE RA-L 2022) are useful graph-based references for relational multi-robot state and exploration. They motivate the graph encoder and the choice to expose agent-to-agent and agent-to-obstacle relations explicitly.

For formation geometry, Benzerrouk, Adouane, and Martinet (Robotics and Autonomous Systems, 2014) study constrained virtual-structure navigation; Roy et al. (IROS 2018) study virtual-structure switching and formation changes in occluded environments. For safety, Alshiekh et al. (AAAI 2018) establish shielding as an intervention layer around reinforcement learning, and Carr et al. (AAAI 2023) extend the discussion to partial observability. AeroGate uses these works as conceptual anchors, then implements a velocity-level geometric filter whose behavior is visible in intervention and clearance diagnostics.

In an interview, connect each citation to one design decision: A* for the team route, virtual structure for relative formation geometry, graph message passing for relational state, SAC for continuous velocity control, centralized critics for joint-action value estimation, and shielding for pre-integration geometric correction. Keep the description tied to the code and the recorded evaluation artifacts.

**中文**

这套实现背后的文献脉络比较直接。Hart、Nilsson 和 Raphael 提出了以代价加启发式函数进行搜索的 A*；在 AeroGate 中，它对应膨胀二维障碍网格上的团队参考点路线。Daniel、Nash、Koenig 和 Felner 提出了适用于任意角度网格路径的 Theta*；Karaman 和 Frazzoli 建立了采样式 RRT* 方法线。这些工作构成评测脚本中的经典规划参考。

Haarnoja 等人在 ICML 2018 提出的 Soft Actor-Critic 提供了连续控制中的最大熵目标，AeroGate 的 actor 和双 critic 采用了这条方法线。Lowe 等人在 NeurIPS 2017 的多智能体 actor-critic 工作体现了 centralized training / decentralized execution 思路。Jiang 等人的 ICLR 2020 Graph Convolutional Reinforcement Learning、Tolstaya 等人的 IROS 2021 Spatial Graph Neural Networks，以及 Zhang 等人的 IEEE RA-L 2022 H2GNN，为关系型多机器人状态和探索提供了图方法参考。它们解释了为什么图编码器要显式暴露 agent-agent 与 agent-obstacle 关系。

编队几何方面，Benzerrouk、Adouane 和 Martinet 在 Robotics and Autonomous Systems（2014）研究了带约束的虚拟结构导航；Roy 等人在 IROS 2018 研究了遮挡环境中的虚拟结构切换和编队变化。安全方面，Alshiekh 等人在 AAAI 2018 将 shielding 建立为强化学习外部干预层，Carr 等人在 AAAI 2023 进一步讨论了部分可观测场景。AeroGate 以这些工作为概念锚点，实现了速度级几何滤波器，并通过 intervention 和 clearance 诊断记录其行为。

面试时，把每篇引用连接到一个设计决策：A* 对应团队路线，virtual structure 对应相对编队几何，图消息传递对应关系状态，SAC 对应连续速度控制，集中式 critic 对应联合动作价值估计，shield 对应积分前的几何修正。每个表述都落到源码结构和评测附件上。

## 18. Source-backed method inventory

**English**

The executable AeroGate path is best explained as a layered method stack. The geometry layer supplies fixed-height kinematics, circular gate-post geometry, signed clearance, line-segment collision, dynamic-gate motion, and swept-clearance checks. The planning layer supplies A* waypoints for the team reference point and the classical planner scripts expose Theta*, RRT*, Informed RRT*, and a heuristic controller for comparison. The coordination layer supplies virtual-structure formation positions, dynamic-gate task-first positions, look-ahead route guidance, graph observations, and active-agent masks.

The learning layer contains SAC-style continuous control in both single-agent and multi-agent forms. The multi-agent implementation is a shared graph actor with a centralized joint-action critic, twin Q critics, target networks, entropy tuning, and masked actions. `GraphFlashSACAgent` is the repository name for this implementation path. Its training loop also includes vectorized environment stepping, reward-scale exponential moving averages, gradient clipping, weight projection, periodic evaluation, checkpoint selection, and conservative resume modes.

The imitation layer has two distinct stages. BC is supervised regression from graph observations to expert joint actions. Expert data comes from a heuristic formation replay controller or a teacher actor checkpoint, is recorded with episode-level geometry diagnostics, and is balanced across sampled team sizes. DAgger then runs the current actor, identifies risk states through clearance, pair distance, formation-position error, height, corridor, and termination signals, queries the teacher, appends the corrections, and repeats BC for the next iteration. The combined entry point can continue with Graph-FlashSAC fine-tuning.

The safety and data-quality layer has three mechanisms. The velocity-level action shield changes the command before integration. Failure-aware replay tags each transition with a failure reason and safety cost, then samples a configured fraction of tagged data. The optional safety critic predicts discounted safety cost, and the optional behavior-anchor term keeps the actor close to a reference action on selected data. These mechanisms are separate from the reward terms, so their intervention and learning effects can be logged independently.

The current AeroGate source gives direct support to the methods above. ACVRP/LKH allocation, FMM, B-spline trajectory generation, frontier semantic exploration, POMDP belief mapping, PPO, successor features, and MAP-Elites belong to other research modules in the wider portfolio. I would describe those under their own project names and keep the AeroGate explanation focused on its route, formation, graph, safety, imitation, replay, and curriculum stack.

**中文**

可执行的 AeroGate 路径适合解释成分层方法栈。几何层提供固定高度运动学、圆形门柱几何、带符号净空、线段碰撞、动态闸门运动和 swept-clearance 检查。规划层为团队参考点提供 A* waypoint；经典规划脚本还提供 Theta*、RRT*、Informed RRT* 和启发式控制器用于比较。协同层提供虚拟结构编队位置、动态闸门 task-first 编队位置、前视路线引导、图观测和 active-agent mask。

学习层包含单机和多机两种 SAC 风格的连续控制。多机实现由共享图 actor、集中式联合动作 critic、双 Q critic、target network、熵系数调节和动作 mask 组成。`GraphFlashSACAgent` 是仓库对这条实现路径的命名。训练循环还包含向量化环境推进、奖励尺度指数滑动平均、梯度裁剪、权重投影、周期性评估、checkpoint 选择和保守恢复模式。

模仿学习层有两个明确阶段。BC 是从图观测回归专家联合动作的监督学习。专家数据来自启发式编队回放控制器或教师 actor checkpoint，按 episode 记录几何诊断，并在采样团队规模之间做平衡。DAgger 运行当前 actor，通过净空、机间距离、编队位置误差、高度、走廊和终止信号识别风险状态，查询教师，追加修正样本，再进行下一轮 BC。组合入口随后可以继续用 Graph-FlashSAC 微调。

安全和数据质量层包含三个机制。速度级 action shield 在积分前修改命令。failure-aware replay 为每个 transition 保存失败原因和 safety cost，再按配置比例抽取风险数据。可选 safety critic 预测折扣安全代价；可选 behavior-anchor 项让 actor 在指定数据上接近参考动作。这些机制与奖励项分开，因此可以分别记录它们对干预和学习的影响。

当前 AeroGate 源码直接支持上面这些方法。ACVRP/LKH 分配、FMM、B-spline 轨迹生成、frontier 语义探索、POMDP belief mapping、PPO、successor features 和 MAP-Elites 属于更大作品集中的其他研究模块。面试时可以用各自项目名称介绍它们，把 AeroGate 的叙述集中在路线、编队、图观测、安全、模仿学习、replay 和 curriculum 这条方法栈上。

## 19. Input-output map for follow-up questions

**English**

| Component | Main input | Main output | Design reason |
| --- | --- | --- | --- |
| `GlobalRoutePlanner2D` | start, goal, inflated gate-post map | team-reference waypoints | solve the global geometric route once per scene or reset |
| `VirtualStructure2D` | team center, heading, agent count | desired formation positions | convert one route into per-agent geometric targets |
| `build_multi_graph_observation` | agent state, formation positions, waypoints, obstacles, masks | `node_features`, `adjacency`, `node_mask`, `action_mask` | encode relations with a fixed tensor interface |
| `GraphPolicy` | graph arrays and masks | per-agent Gaussian mean, log-std, sampled 2D actions | shared continuous-control policy |
| centralized critic | pooled graph state, joint action, action mask | scalar joint-action value | team-level credit assignment during training |
| action shield | desired velocities, pair geometry, boundaries, gates, obstacles | corrected velocities and intervention diagnostics | correct geometric risks before integration |
| BC trainer | expert graph observations and joint actions | actor checkpoint and loss history | provide a supervised warm start |
| DAgger loop | current actor states, teacher action, risk tests | correction dataset and updated actor | focus data collection on policy-visited risk states |
| failure replay | transitions, failure tags, safety costs | sampled training batch | increase exposure to rare risk transitions |
| vector environment | batch of actions, per-environment seeds | batched observations, rewards, done flags, infos | collect experience efficiently while preserving independent episodes |

**中文**

| 组件 | 主要输入 | 主要输出 | 设计理由 |
| --- | --- | --- | --- |
| `GlobalRoutePlanner2D` | 起点、终点、膨胀门柱地图 | 团队参考点 waypoint | 为场景或 reset 生成全局几何路线 |
| `VirtualStructure2D` | 团队中心、航向、agent 数量 | 期望编队位置 | 把一条路线转换成每个 agent 的几何目标 |
| `build_multi_graph_observation` | agent 状态、编队位置、waypoint、障碍物、mask | `node_features`、`adjacency`、`node_mask`、`action_mask` | 用固定张量接口编码关系 |
| `GraphPolicy` | 图数组和 mask | 每个 agent 的高斯均值、log-std、二维采样动作 | 共享连续控制策略 |
| centralized critic | pooled 图状态、联合动作、action mask | 联合动作价值标量 | 训练阶段进行团队级 credit assignment |
| action shield | 期望速度、机间几何、边界、闸门、障碍物 | 修正速度和干预诊断 | 在积分前修正几何风险 |
| BC trainer | 专家图观测和联合动作 | actor checkpoint 与损失历史 | 提供监督式 warm start |
| DAgger 循环 | 当前 actor 状态、教师动作、风险测试 | 修正数据集和更新后的 actor | 把数据集中到策略访问过的风险状态 |
| failure replay | transition、failure tag、safety cost | 训练 batch | 增加稀有风险 transition 的采样比例 |
| vector environment | 动作 batch、各环境 seed | batch 观测、奖励、done 标记、info | 提高经验采集效率，同时保持 episode 独立 |

## 20. Complete learning pipeline

**English**

The repository supports a staged learning pipeline, and each stage has a separate role. First, `collect_expert_demonstrations` rolls out either the deterministic `HeuristicFormationReplayController` or a teacher actor checkpoint. It stores graph observations, joint actions, team size, and episode-level geometry diagnostics. The collector can sample team-size buckets, score candidate episodes, and retain a balanced set of demonstrations.

Second, `run_actor_behavior_cloning` trains the shared graph actor as a supervised regressor. The network predicts a two-dimensional action for each possible agent, and `action_mask` makes the loss count active agents. The loss is masked mean-squared error between the actor's `tanh` output and the expert joint action. The implementation can add a log-standard-deviation penalty, split the data into training and validation sets, select the best validation epoch, and save an actor-weight checkpoint.

Third, `collect_dagger_corrections` runs the current actor in the environment and queries the teacher at states that need a correction. The query tests include low clearance, low pair distance, large formation-position error, height-contract failure, side bypass, corridor miss, formation-line collapse, and a terminal state before corridor completion. In non-dynamic scenes, the loop can also issue periodic scheduled queries. `append_dagger_corrections` merges the new observations and teacher actions into the existing dataset, then the next BC pass starts from the previous actor checkpoint.

Finally, `run_dagger_warmstart_then_finetune` can pass the corrected actor to the Graph-FlashSAC training loop. The RL stage uses off-policy replay, a shared actor, a centralized joint-action critic, twin Q targets, entropy tuning, and optional safety and behavior-anchor terms. This staged design gives me a practical way to move from a geometric teacher, to a policy that has seen its own difficult states, to continuous-control fine-tuning.

The default configuration records the intended protocol in code: 12 expert episodes, 10 BC epochs, a 10% validation split, two DAgger iterations, two rollout episodes per iteration, a query interval of five steps in scheduled-query scenes, and up to 512 corrections per iteration. These values describe the configured pipeline; an interview claim about performance still needs the corresponding run manifest and evaluation artifact.

**中文**

仓库支持分阶段学习流程，每一阶段有明确职责。第一步，`collect_expert_demonstrations` 运行确定性的 `HeuristicFormationReplayController`，或者加载教师 actor checkpoint。它保存图观测、联合动作、团队规模和 episode 级几何诊断。采集器可以按团队规模 bucket 采样，对候选 episode 打分，并保留平衡后的示范集。

第二步，`run_actor_behavior_cloning` 把共享图 actor 作为监督回归器训练。网络为每个可能的 agent 预测二维动作，`action_mask` 让损失按 active agent 计入。损失是 actor 的 `tanh` 输出与专家联合动作之间的带 mask 均方误差。实现还可以加入 log-standard-deviation 惩罚，划分训练集和验证集，按最佳验证 epoch 选择模型，并保存 actor 权重 checkpoint。

第三步，`collect_dagger_corrections` 让当前 actor 在环境中运行，并在需要修正的状态查询教师。查询条件包括净空偏低、机间距离偏小、编队位置误差偏大、高度合同失败、侧向绕行、走廊遗漏、编队线塌缩，以及走廊完成前已经终止的状态。在非动态场景中，循环还可以按固定间隔发起 scheduled query。`append_dagger_corrections` 把新的观测和教师动作合并到现有数据集，下一轮 BC 从上一轮 actor checkpoint 继续训练。

最后，`run_dagger_warmstart_then_finetune` 可以把修正后的 actor 交给 Graph-FlashSAC 训练循环。RL 阶段使用 off-policy replay、共享 actor、集中式联合动作 critic、双 Q target、熵系数调节，以及可选的 safety 和 behavior-anchor 项。这个分阶段设计把几何教师、策略自己访问过的困难状态和连续控制微调连接起来。

默认配置在代码中记录了预期流程：12 个专家 episode、10 个 BC epoch、10% 验证集、2 轮 DAgger、每轮 2 个 rollout episode、scheduled-query 场景下每 5 步查询一次教师，以及每轮最多 512 个修正样本。这些数字描述的是配置协议；面试中涉及性能的陈述仍然要对应具体 run manifest 和评测附件。

## 21. Method and source map

**English**

| Method or mechanism | Source entry | What I can explain in an interview |
| --- | --- | --- |
| Fixed-height kinematic core | `shared/core/fixed_height.py`, `kinematics_2d.py` | planar state update, speed and acceleration limits, height invariant |
| Gate geometry and collision | `shared/core/collision_2d.py`, `team_geometry.py` | circular posts, point and segment checks, signed clearance, pair distance |
| Global route planning | `multi_gate/planners/global_route_planner.py` | inflated-grid A*, eight-neighbor search, Euclidean heuristic, waypoint simplification |
| Virtual structure | `multi_gate/formation/virtual_structure.py` | rotate relative formation positions with the route heading and adapt the layout to team size |
| Dynamic-gate task-first formation | `multi_gate/dynamic_gate_task_slots.py` | predict gate arrival, blend toward a compact pass arrangement, respect the opening geometry |
| Graph observation | `multi_gate/env/observation_multi.py`, `observation_runtime.py` | agent, formation-position, waypoint, obstacle, guidance, and team nodes with masks |
| Shared graph actor | `multi_gate/graph_rl/graph_policy.py` | MLP node embedding, masked message passing, mean pooling, Gaussian action head |
| Fixed-height kinematic-3D adapter | `multi_gate/env/multi_gate_kinematic_3d_env.py` | reuse the task contract in a fixed-height kinematic shell for controlled scene variants |
| Graph-SAC / Graph-MASAC | `single_gate/graph_rl`, `multi_gate/graph_rl/graph_masac.py`, `graph_flashsac.py` | single-agent Graph-SAC plus shared-actor multi-agent Graph-MASAC with a centralized joint-action critic |
| Modular actor heads | `multi_gate/graph_rl/graph_policy.py` | optional route, formation, and avoidance proposals mixed by learned soft gates |
| Geometric action shield | `multi_gate/env/safety_shields.py` | pairwise, boundary, guidance, gate-channel, obstacle, braking, and repulsion corrections before integration |
| Expert replay controller | `multi_gate/replay.py` | deterministic formation tracking, feed-forward motion, repulsion, and dynamic-gate handling |
| BC warm start | `multi_gate/imitation.py`, `actor_behavior_clone_loss` | masked action regression, validation selection, actor checkpoint export |
| DAgger correction | `multi_gate/dagger.py` | policy rollout, risk-triggered teacher query, dataset append, repeated BC |
| Failure-aware replay | `multi_gate/graph_rl/replay_buffer.py` | failure tags, reasons, safety costs, and a configured failure fraction in each batch |
| Safety critic | `graph_masac.py` | optional centralized discounted safety-cost prediction and actor penalty |
| Behavior anchor | `graph_masac.py` | optional distance to a frozen reference actor or selected replay action |
| Dynamic curriculum | `density_multi/scripts/train_dynamic_gate_density_8d_curriculum.py` | staged gate density, motion amplitude/speed, controller parameters, and eight-agent evaluation |
| Route guidance interface | `multi_gate/guidance`, `multi_gate/env/guidance_runtime.py` | optional low-frequency target, heading, risk, speed, mode, and confidence context |
| Vectorized training | `multi_gate/env/vector_multi_gate_env.py`, `training.py` | batched independent environments, resets, checkpoints, evaluation gates, and manifests |
| Planner evaluation harness | `scripts/run_classic_planner_baselines.py`, `internal/planners` | evaluate A*, Theta*, RRT*, Informed RRT*, heuristic, and available stronger planner rows |
| Replay and visualization | `multi_gate/isaac_replay.py`, `live_preview.py` | map a saved trajectory to Isaac Lab assets for visual inspection while metrics stay in the NumPy core |

**中文**

| 方法或机制 | 源码入口 | 面试时可以解释的内容 |
| --- | --- | --- |
| 固定高度运动学核心 | `shared/core/fixed_height.py`、`kinematics_2d.py` | 平面状态更新、速度与加速度限制、高度不变量 |
| 闸门几何与碰撞 | `shared/core/collision_2d.py`、`team_geometry.py` | 圆形门柱、点和线段检查、带符号净空、机间距离 |
| 全局路线规划 | `multi_gate/planners/global_route_planner.py` | 障碍膨胀网格 A*、八邻域搜索、欧氏启发式、waypoint 简化 |
| 虚拟结构 | `multi_gate/formation/virtual_structure.py` | 按路线航向旋转相对编队位置，并随团队规模调整布局 |
| 动态门 task-first 编队 | `multi_gate/dynamic_gate_task_slots.py` | 预测到门时间，向紧凑通过编队混合，并满足门洞几何 |
| 图观测 | `multi_gate/env/observation_multi.py`、`observation_runtime.py` | 带 mask 的 agent、编队位置、waypoint、障碍物、guidance 和团队节点 |
| 固定高度运动学 3D 适配器 | `multi_gate/env/multi_gate_kinematic_3d_env.py` | 在固定高度运动学外壳中复用任务合同，用于受控场景变体 |
| 共享图 actor | `multi_gate/graph_rl/graph_policy.py` | MLP 节点嵌入、带 mask 的消息传递、平均池化和高斯动作头 |
| Graph-SAC / Graph-MASAC | `single_gate/graph_rl`、`multi_gate/graph_rl/graph_masac.py`、`graph_flashsac.py` | 单机 Graph-SAC，以及带集中式联合动作 critic 的共享 actor 多机 Graph-MASAC |
| 模块化 actor head | `multi_gate/graph_rl/graph_policy.py` | 用学习到的 soft gate 混合路线、编队和避障动作提议 |
| 几何 action shield | `multi_gate/env/safety_shields.py` | 积分前完成机间、边界、guidance、门通道、障碍、制动和排斥修正 |
| 专家回放控制器 | `multi_gate/replay.py` | 确定性编队跟踪、前馈运动、排斥和动态门处理 |
| BC warm start | `multi_gate/imitation.py`、`actor_behavior_clone_loss` | 带 mask 的动作回归、验证集选择和 actor checkpoint 导出 |
| DAgger 修正 | `multi_gate/dagger.py` | 策略 rollout、风险触发教师查询、数据集合并和重复 BC |
| 失败感知 replay | `multi_gate/graph_rl/replay_buffer.py` | failure tag、失败原因、安全代价和每个 batch 的配置比例 |
| Safety critic | `graph_masac.py` | 可选的集中式折扣安全代价预测和 actor 惩罚 |
| Behavior anchor | `graph_masac.py` | 可选地让 actor 接近冻结参考 actor 或选定 replay 动作 |
| 动态 curriculum | `density_multi/scripts/train_dynamic_gate_density_8d_curriculum.py` | 分阶段改变门密度、运动振幅/速度、控制器参数，并进行 8 机评估 |
| Route guidance 接口 | `multi_gate/guidance`、`multi_gate/env/guidance_runtime.py` | 可选的低频目标、航向、风险、速度、模式和置信度上下文 |
| 向量化训练 | `multi_gate/env/vector_multi_gate_env.py`、`training.py` | 独立环境 batch、重置、checkpoint、评估 gate 和 manifest |
| 规划器评测工具 | `scripts/run_classic_planner_baselines.py`、`internal/planners` | 评估 A*、Theta*、RRT*、Informed RRT*、heuristic 及可用的更强规划器行 |
| 回放与可视化 | `multi_gate/isaac_replay.py`、`live_preview.py` | 把保存的轨迹映射到 Isaac Lab 资产进行视觉检查，指标仍由 NumPy 核心提供 |

## 22. A natural full-pipeline explanation

**English**

“I built the project as a layered pipeline. A* gives the team reference point a route, and the virtual-structure layer turns that route into desired formation positions for the agents. The graph observation then combines local state, teammate relations, formation targets, route look-ahead, and nearby gate geometry in one masked tensor interface. The shared actor proposes one planar velocity for each active agent. During training, a centralized critic sees the pooled graph state and the joint action, so the value estimate includes team interactions. Before the kinematic update, the action shield checks pair distance, boundaries, obstacles, and the gate channel and can correct the velocity command.

For learning, I can start with behavior cloning from a heuristic formation controller or a teacher actor. BC gives the actor a supervised warm start. DAgger then rolls out that actor, asks the teacher for actions at risky states, appends those corrections, and repeats BC. The corrected actor can be fine-tuned with Graph-FlashSAC. The replay buffer also keeps failure labels and safety costs, so rare collision or near-miss transitions can be sampled deliberately. In the eight-agent dynamic-gate curriculum, I change gate density and motion difficulty stage by stage and keep the episode diagnostics attached to each run.”

**中文**

“我把这个项目实现成分层流程。A* 为团队参考点生成路线，虚拟结构层再把这条路线转换成各个 agent 的期望编队位置。图观测把局部状态、队友关系、编队目标、路线前视信息和附近闸门几何放进一个带 mask 的张量接口。共享 actor 为每个 active agent 提出一个平面速度。训练时，集中式 critic 同时看到 pooled 图状态和联合动作，因此价值估计包含团队交互。运动学更新之前，action shield 检查机间距离、边界、障碍物和门通道，并可以修正速度命令。

在学习部分，我可以先用启发式编队控制器或教师 actor 收集示范，再进行 behavior cloning。BC 为 actor 提供监督式 warm start。DAgger 让这个 actor 在环境中运行，在风险状态查询教师动作，把修正样本追加回来，再重复 BC。修正后的 actor 可以继续用 Graph-FlashSAC 微调。Replay buffer 还保存失败标签和安全代价，因此可以有意增加碰撞或近失状态的采样比例。在 8 机动态门 curriculum 中，我分阶段改变门密度和运动难度，并把 episode 诊断和每次运行一起保存。”
