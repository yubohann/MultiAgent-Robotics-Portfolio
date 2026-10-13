# AeroGate Graph | Research Defense Display

> **Use**: slide-ready content for the FAIR Lab interview deck. Each block is designed to become one Reveal.js section. The English text is presentation content; the Chinese text is speaker support. The content is grounded in the AeroGate Graph source tree and retained evaluation artifacts.

## 01 | Research question

**English**

### AeroGate Graph

**Coordinating a variable-size team through a changing gate course**

AeroGate Graph is a modular research environment for studying multi-agent coordination under three coupled requirements: route progress, formation geometry, and geometric safety. I built the environment around a fixed-height, two-dimensional kinematic task so that the scene, action space, success definition, and diagnostics stay stable while the coordination design changes.

The central question is: **How should a team with a changing number of agents move through a constrained gate course while keeping a usable formation and reacting to static or moving obstacles?**

The design separates four decisions:

1. the route of the team reference point;
2. the desired relative formation position of each agent;
3. the relational state passed to the policy;
4. the correction applied when a velocity command approaches a geometric constraint.

**中文**

### AeroGate Graph

**让规模可变的多机团队通过动态闸门场景**

AeroGate Graph 是一个研究多智能体协同的模块化环境，围绕三个相互耦合的要求展开：路线进度、编队几何和几何安全。我采用固定高度二维运动学任务，使场景、动作空间、成功定义和诊断字段保持稳定，从而可以逐项研究不同的协同设计。

核心问题是：**当团队规模变化、障碍物可能移动时，多机团队如何在保持可用编队的同时通过受限闸门场景？**

设计上把问题拆为四个决策：团队参考点的路线、每架无人机的期望相对编队位置、输入策略的关系状态，以及速度命令接近几何约束时的修正方式。

---

## 02 | Task contract

**English**

### A controlled planar task

The shared default configuration uses a `0.1 s` simulation step, a planar command-speed limit of approximately `3.5 m/s`, and an acceleration limit of approximately `2.45 m/s^2`. The environment state contains planar position, planar velocity, yaw, and time. The policy action is a normalized two-dimensional command that is converted to a desired planar velocity.

The kinematic update has a fixed order:

```text
policy action
    -> desired planar velocity
    -> velocity clipping
    -> acceleration limit
    -> position integration
    -> segment collision and clearance checks
    -> reward and termination update
```

Static gates are represented by circular gate-post obstacles. Dynamic gates update their centers and velocities from the episode time. The environment evaluates both point clearance and the full movement segment, and dynamic scenes additionally use a relative swept-clearance check.

This contract gives every method the same geometric meaning for progress, collision, clearance, formation error, and termination.

**中文**

### 受控的平面运动任务

共享默认配置使用 `0.1 s` 时间步、约 `3.5 m/s` 的平面速度上限和约 `2.45 m/s²` 的加速度上限。环境状态包括平面位置、平面速度、航向角和时间。策略输出归一化二维动作，环境再把它转换为期望平面速度。

运动学更新按固定顺序执行：策略动作转换为速度、速度裁剪、施加加速度限制、位置积分、整段运动碰撞与净空检查，最后更新奖励和终止状态。

静态闸门由圆形门柱障碍物表示。动态闸门根据 episode 时间更新中心和速度；动态场景还会在一个时间步内计算相对运动的 swept clearance。这样，不同方法使用的是同一套几何含义和任务合同。

---

## 03 | End-to-end architecture

**English**

### One state contract, several decision layers

```text
scenario configuration
        -> gate layout and obstacle geometry
        -> A* route for the team reference point
        -> desired formation positions from a virtual structure
        -> masked graph observation
        -> shared actor action for each active agent
        -> velocity-level geometric action shield
        -> speed and acceleration limits
        -> state integration
        -> collision, clearance, formation, reward, and termination records
```

The source tree keeps the contracts inspectable:

| Layer | Source area | Output used by the next layer |
| --- | --- | --- |
| Geometry and kinematics | `shared/core` | positions, velocities, obstacles, clearance |
| Route planning | `multi_gate/planners` | team-reference waypoints |
| Formation | `multi_gate/formation` | desired formation positions |
| Observation | `multi_gate/env/observation_multi.py` | graph tensors and masks |
| Policy and critic | `multi_gate/graph_rl` | per-agent velocity actions and joint value |
| Safety | `multi_gate/env/safety_shields.py` | corrected velocity commands |
| Evaluation | `multi_gate/env`, `evaluation_artifacts` | metrics, failure labels, termination reason |

**中文**

### 一套状态合同，多个决策层

完整执行链是：场景配置生成闸门和障碍物几何；A* 为团队参考点生成路线；虚拟结构生成期望编队位置；环境构造带 mask 的图观测；共享 actor 为 active agent 输出动作；速度级几何 action shield 进行修正；运动学模块施加速度和加速度限制并积分；最后记录碰撞、净空、编队、奖励和终止状态。

源码按层次组织：`shared/core` 负责几何与运动学，`multi_gate/planners` 负责路线，`formation` 负责编队，`observation_multi.py` 负责图观测，`graph_rl` 负责 actor/critic，`safety_shields.py` 负责动作修正，评测模块负责指标和失败原因。

---

## 04 | Route and formation

**English**

### A* chooses the route; the virtual structure chooses the team geometry

The global route planner runs A* on an obstacle-inflated two-dimensional grid with eight-neighbour connectivity and a Euclidean heuristic. After reconstructing the grid path, it converts the path to world coordinates and simplifies a longer segment when a collision check confirms that the shortcut is clear.

The planner output is a waypoint sequence for the **team reference point**. It is not an independent path for every agent. The virtual-structure layer then places each agent at a desired relative position around that reference point and rotates the offsets into the world frame using the current route heading.

The default formation geometry uses approximately `2.4 m` lateral spacing and `3.0 m` longitudinal spacing. The layout changes with team size: smaller teams use fewer columns, while larger teams use more columns. In the source, these desired positions are stored in an array called `slots`; in the presentation I call them **desired formation positions**.

This decomposition gives a clean interpretation to every error signal:

| Quantity | Meaning |
| --- | --- |
| route progress | how far the team reference point has advanced |
| formation-position error | how far an agent is from its geometric team target |
| minimum clearance | distance to gate posts or obstacles |
| minimum pair distance | separation between agents |

**中文**

### A* 选择路线，虚拟结构决定团队几何

全局规划器在障碍膨胀的二维网格上运行 A*，使用八邻域连接和欧氏距离启发函数。路径重建后转换为世界坐标；当碰撞检查确认较长线段安全时，规划器会进行 waypoint 简化。

规划器输出的是**团队参考点**的 waypoint 序列，而不是每个 agent 的独立轨迹。虚拟结构层在参考点周围生成每个 agent 的期望相对位置，再根据当前路线航向把这些偏移旋转到世界坐标。

默认横向间距约 `2.4 m`，纵向间距约 `3.0 m`。团队规模变化时，编队列数也会变化。源码把这些位置存为 `slots`，面试表达中使用更自然的说法 **desired formation positions**。

---

## 05 | Graph observation

**English**

### Relational state with a stable tensor interface

The multi-agent observation is represented by four arrays:

```text
node_features | adjacency | node_mask | action_mask
```

The graph is padded to a fixed capacity. `node_mask` identifies valid entities, and `action_mask` identifies active agents that should receive actions. The portfolio contains single-agent runs and multi-agent runs with teams from one to eight agents; the dynamic gate-density curriculum uses eight agents.

The graph carries six types of information:

| Node type | Information carried | Why it is present |
| --- | --- | --- |
| Agent | relative position, velocity, formation-position error, clearance, progress | local motion and team state |
| Formation position | desired offset, heading context, index, progress | geometric team target |
| Waypoint | look-ahead position, route heading, order, progress | near-term route context |
| Obstacle / gate | relative position, radius, velocity, short prediction, clearance | static and moving geometric risk |
| Guidance | target, heading, risk or speed, confidence | optional high-level context |
| Team | virtual center, heading, overall clearance, progress | shared global context |

The encoder maps node features through an MLP, performs masked message passing, and mean-pools valid nodes. The actor combines each agent embedding with the pooled team embedding and produces one two-dimensional action for each possible agent.

**中文**

### 用稳定张量接口保留关系状态

多智能体观测由四个数组组成：`node_features`、`adjacency`、`node_mask` 和 `action_mask`。图被填充到固定容量，`node_mask` 标识有效实体，`action_mask` 标识当前 active agent。作品集包含单机和 1 到 8 机团队实验，动态门密度 curriculum 使用 8 机。

图中包含六类信息：agent 节点描述局部运动和编队状态；formation-position 节点描述几何目标；waypoint 节点提供路线前视；obstacle/gate 节点描述静态或动态风险；guidance 节点提供可选的高层信息；team 节点提供团队全局上下文。

编码器先用 MLP 映射节点特征，再执行带 mask 的消息传递，并对有效节点做平均池化。actor 把每个 agent 的嵌入与团队 pooled embedding 结合，为每个可能的 agent 输出二维动作。

---

## 06 | Why this graph design

**English**

### The policy receives the relations that affect the action

The main design choice is to make the relations explicit. A local agent state alone does not describe whether a teammate is closing in, whether the formation target is inside the next gate, or whether the team is approaching a boundary. Those quantities appear as agent-agent, agent-formation, agent-waypoint, and agent-obstacle relations in the graph.

Padding and masks provide a stable interface across team sizes. Parameter sharing gives active agents the same actor parameters, while their node embeddings and the pooled team embedding carry their different local contexts. The ordered agent block also makes action-to-agent mapping easy to inspect in a run record.

The graph encoder is therefore an **interface choice** that exposes relational state to a continuous-control policy. A study of whether it improves performance over a non-graph policy would use paired seeds, fixed geometry, and component ablations.

**中文**

### 策略接收真正影响动作的关系

这里最核心的设计是显式保留关系信息。单个 agent 的局部状态无法说明队友是否正在接近、编队目标是否落在下一道门内、团队是否正在接近边界。这些信息分别通过 agent-agent、agent-formation、agent-waypoint 和 agent-obstacle 关系进入图观测。

padding 和 mask 让不同团队规模共享稳定接口。参数共享保证 active agent 使用同一套 actor 参数，而各自的节点嵌入和团队 pooled embedding 携带局部差异。有序 agent block 也让运行记录中的动作和 agent 对应关系容易检查。

因此，图编码器首先是一个把关系状态送入连续控制策略的接口设计。若要研究它相对于非图策略的性能影响，应使用配对 seed、固定几何和组件消融。

---

## 07 | Graph-SAC and centralized training

**English**

### Continuous control with a team-level value estimate

The actor is shared across active agents. Its graph encoder produces node and team embeddings, and the policy head outputs a Gaussian mean and log standard deviation for each agent. Reparameterized sampling and `tanh` produce bounded planar actions; `action_mask` removes inactive-agent controls.

The multi-agent path follows centralized training with decentralized execution:

| Training side | Execution side |
| --- | --- |
| centralized joint-action critic | shared actor produces one action per active agent |
| pooled graph state plus masked joint action | each action uses the agent and team embeddings |
| twin Q critics and target critics | local action output remains compact and continuous |
| SAC entropy term and temperature tuning | active-agent mask controls which actions are applied |

The repository exposes this path as `GraphMASACAgent` and `GraphFlashSACAgent`. The training loop also provides reward-scale EMA, gradient clipping, weight projection, soft target updates, vectorized environments, checkpoint selection, and conservative resume handling.

**中文**

### 连续控制与团队级价值估计

所有 active agent 共享 actor。图编码器生成节点嵌入和团队嵌入，策略头为每个 agent 输出高斯均值和对数标准差；重参数化采样和 `tanh` 生成有界平面动作，`action_mask` 屏蔽 inactive agent。

多机路径采用 centralized training / decentralized execution：训练侧使用集中式联合动作 critic、pooled 图状态、mask 后的联合动作、双 Q critic、target critic、SAC 熵项和温度调节；执行侧由共享 actor 为每个 active agent 输出动作。

仓库对外提供 `GraphMASACAgent` 和 `GraphFlashSACAgent`。训练循环还包括 reward-scale EMA、梯度裁剪、权重投影、软更新、向量化环境、checkpoint 选择和保守恢复。

---

## 08 | Imitation and reinforcement learning pipeline

**English**

### From a geometric teacher to policy fine-tuning

The learning path is staged so that every dataset and checkpoint has a clear role.

```text
heuristic formation controller or teacher actor
        -> expert demonstrations
        -> behavior cloning (BC)
        -> policy rollout on its own visited states
        -> DAgger teacher corrections
        -> repeated BC
        -> Graph-FlashSAC fine-tuning
```

**Behavior cloning.** `collect_expert_demonstrations` records graph observations, joint actions, team size, and episode-level geometry diagnostics. The source can be the deterministic `HeuristicFormationReplayController` or a teacher actor checkpoint. `run_actor_behavior_cloning` trains the shared actor with masked action MSE, an optional log-standard-deviation penalty, a train/validation split, and best-epoch checkpoint selection.

**DAgger.** `collect_dagger_corrections` runs the current actor and queries the teacher at policy-visited states with low clearance, low pair distance, large formation-position error, height-contract events, side bypass, corridor miss, formation-line collapse, or terminal status before corridor completion. The correction samples are appended to the BC dataset, and the next BC pass starts from the previous actor checkpoint.

**Fine-tuning.** `run_dagger_warmstart_then_finetune` sends the corrected actor to the off-policy Graph-FlashSAC loop. The replay buffer stores graph observations, joint actions, rewards, done flags, failure tags, failure reasons, and safety costs.

**中文**

### 从几何教师到策略微调

学习流程是分阶段的，每个数据集和 checkpoint 都有明确作用：启发式编队控制器或教师 actor 产生专家示范；BC 进行监督 warm start；当前策略在自己访问过的状态上运行；DAgger 查询教师修正；追加数据后重复 BC；最后用 Graph-FlashSAC 微调。

BC 的 `collect_expert_demonstrations` 保存图观测、联合动作、团队规模和 episode 几何诊断，数据来源可以是确定性的 `HeuristicFormationReplayController` 或教师 actor checkpoint。`run_actor_behavior_cloning` 使用带 mask 的动作 MSE，可选 log-standard-deviation 惩罚、训练/验证划分和最佳 epoch checkpoint。

DAgger 的 `collect_dagger_corrections` 让当前 actor 运行，在低净空、过小机间距离、较大编队位置误差、高度合同事件、侧向绕行、走廊遗漏、编队线塌缩，或走廊完成前终止的状态查询教师。修正样本追加到 BC 数据集，下一轮 BC 从上一轮 actor checkpoint 开始。

最终，`run_dagger_warmstart_then_finetune` 把修正后的 actor 交给 off-policy Graph-FlashSAC。Replay buffer 保存图观测、联合动作、奖励、done、失败标签、失败原因和安全代价。

---

## 09 | Failure-aware learning

**English**

### Use execution records to shape the training data

The replay and safety mechanisms are separate from the reward definition, so their effects can be inspected independently.

| Mechanism | Stored or computed information | Training role |
| --- | --- | --- |
| Failure-aware replay | failure tag, failure reason, safety cost | sample a configured fraction of rare risk transitions |
| Centralized safety critic | discounted safety-cost estimate | add a safety penalty to the actor objective when enabled |
| Behavior anchor | distance to a frozen reference actor or selected action | keep a warm-start actor near a trusted behavior on selected data |
| Modular actor heads | route, formation, and avoidance proposals | mix proposals through learned soft gates when enabled |
| Risk-weighted BC sampling | geometry fields from demonstration records | give difficult dynamic-gate states more training exposure |

The default failure-replay configuration enables a `0.3` failure fraction. The stored failure categories include gate-post collision, inter-agent collision, out-of-bounds, formation-line collapse, near-miss clearance, small pair distance, and a sharp formation-position error. These labels make a failure a queryable training object rather than an unstructured log message.

**中文**

### 用执行记录组织训练数据

Replay 和安全机制与奖励定义分开，因此可以单独检查它们对训练的影响。

failure-aware replay 保存 failure tag、失败原因和 safety cost，并按配置比例抽取稀有风险 transition。可选 centralized safety critic 预测折扣安全代价，再把安全惩罚加入 actor 目标。behavior anchor 让 warm-start actor 在指定数据上接近冻结参考 actor 或参考动作。模块化 actor head 可以分别生成路线、编队和避障动作提议，再通过学习到的 soft gate 混合。BC 数据选择器还可以利用几何字段增加困难动态门状态的训练覆盖。

默认 failure-replay 配置启用 `0.3` 的风险样本比例。失败类别包括门柱碰撞、机间碰撞、越界、编队线塌缩、近失净空、机间距离过小和编队位置误差突增。这样，失败状态可以作为可查询的训练对象参与采样。

---

## 10 | Dynamic-gate runtime and curriculum

**English**

### The next gate changes the formation target

The dynamic-gate runtime generates a seeded gate field and updates each gate center from time, amplitude, and speed. The density line supports gate counts up to 60. The runtime stores current and next-frame centers, derives gate velocities, builds a live obstacle map, and evaluates swept clearance over the simulation step.

When the team approaches a moving gate, the task-first formation module estimates the arrival time and predicts the gate center. It blends the normal formation positions toward a compact pass arrangement, assigns agents to lateral lanes, orders them longitudinally, and limits the offsets to the opening geometry. The resulting desired positions feed both the heuristic controller and the graph observation.

The separate eight-agent curriculum changes difficulty stage by stage. Its schedule can vary gate count, moving-gate amplitude and speed, controller parameters, and train/evaluation seeds. The recorded speed range is approximately `1.15–3.50 m/s`, paired with acceleration limits of approximately `0.75–2.45 m/s^2`.

This runtime turns a moving obstacle from a static penalty into a time-indexed geometric constraint that affects both formation planning and action correction.

**中文**

### 下一道门会改变编队目标

动态门运行时根据 seed 生成闸门场景，并根据时间、振幅和速度更新每道门的中心。密度线支持最多 60 道门。运行时保存当前帧和下一帧中心，计算门的速度，构造 live obstacle map，并在一个仿真步内检查 swept clearance。

当团队接近移动门时，task-first formation 模块估计到达时间并预测门中心，把常规编队位置逐渐混合到紧凑通过编队，将 agent 分配到横向车道，按纵向排序，并依据门洞几何限制偏移。生成的期望位置同时提供给启发式控制器和图观测。

独立的 8 机 curriculum 按阶段改变门数量、移动门振幅和速度、控制器参数以及训练/评估 seed。记录中的速度范围约为 `1.15–3.50 m/s`，加速度限制约为 `0.75–2.45 m/s²`。

---

## 11 | Safety and termination

**English**

### Intervene before integration, verify after integration

The action shield operates in velocity space before kinematic integration:

```text
policy action
    -> desired velocity
    -> post-gate cruise handling
    -> pair-separation correction
    -> boundary braking and inward correction
    -> route / corridor correction
    -> dynamic gate-channel correction
    -> obstacle repulsion and braking
    -> corrected velocity
```

The environment then applies speed and acceleration limits and checks the resulting movement segment. The record keeps the intervention norm and closeness diagnostics together with minimum clearance, pair distance, formation-position error, reward terms, planner calls, latency, and terminal reason.

The task-level termination contract includes:

| State | Evaluation meaning |
| --- | --- |
| goal completion | team reference point reaches the goal with formation and clearance conditions satisfied |
| gate or obstacle collision | geometric contact check fails |
| inter-agent collision | pair separation check fails |
| corridor miss | the team leaves the required gate corridor |
| formation-line collapse | the configured formation condition is violated |
| boundary or timeout | the state leaves the allowed region or reaches the time limit |

**中文**

### 积分前干预，积分后验证

action shield 在运动学积分前的速度空间工作。策略动作先转为期望速度，再依次进行过门巡航处理、机间分离修正、边界制动和向内修正、路线/走廊修正、动态门通道修正，以及障碍物排斥和制动，最后得到修正后的速度。

环境随后施加速度和加速度限制，并检查整段运动。运行记录同时保存干预范数、接近程度诊断、最小净空、机间距离、编队位置误差、奖励项、规划调用、延迟和终止原因。

终止合同包括目标完成、门柱或障碍物碰撞、机间碰撞、走廊遗漏、编队线塌缩、越界和超时，每一种都有明确的评测含义。

---

## 12 | Evaluation design

**English**

### Every result carries its operating conditions

The evaluation scripts record scenario family, gate count, team size, seed or episode count, method label, and task metrics in machine-readable CSV/JSON artifacts. The main metrics are:

| Category | Metrics |
| --- | --- |
| Task outcome | success rate, progress distance, termination reason, timeout rate |
| Safety | collision rate, minimum clearance, minimum pair distance |
| Coordination | mean and maximum formation-position error, line-collapse state |
| Efficiency | path length, flight time, mean speed, planner calls, latency |
| Dynamic behavior | moving-gate speed, amplitude, corridor completion, swept clearance |

The public `reproduce` path runs the NumPy core twice for the same seed and compares the diagnostic output. It verifies repeatability of the geometry, dynamics, and record path. It is an environment check; learned-policy performance comes from the separate training and evaluation artifacts.

**中文**

### 每个结果都带有运行条件

评测脚本把场景类型、门数量、团队规模、seed 或 episode 数、方法标签和任务指标保存为 CSV/JSON。指标包括任务结果、安全、协同、效率和动态行为：成功率、进度、终止原因、碰撞率、最小净空、最小机间距离、编队位置误差、路径长度、飞行时间、规划调用、延迟、动态门速度/振幅以及 swept clearance。

公共 `reproduce` 路径用同一个 seed 重复运行 NumPy 核心，并比较诊断输出。它验证几何、动力学和记录路径的重复性；学习策略性能则来自独立的训练和评测附件。

---

## 13 | Recorded evidence

**English**

### Historical artifacts retained in the repository

The following values come from the retained artifacts generated on 21 June 2026. The sample size and method label belong to each row.

#### Single-agent dynamic gate comparison

| Method / gate count | Seeds | Success | Collision | Evidence fields |
| --- | ---: | ---: | ---: | --- |
| `ours_mainline` / 42 gates | 10 | 100% | 0% | path `83.388 m`, flight time `72.62 s`, min clearance `0.052 m`, moving-gate swept clearance `0.049 m` |
| `astar` / 42 gates | 10 | 10% | 80% | planner-only comparison row in the same table |

#### Multi-agent retained table

| Scenario | Gates | Episodes | Success | Collision | Path length | Flight time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| static | 60 | 3 | 100% | 0% | `66.525 m` | `35.267 s` |
| dynamic | 36 | 3 | 100% | 0% | `67.087 m` | `47.000 s` |
| dynamic | 60 | 3 | 100% | 0% | `66.669 m` | `38.033 s` |

The recorded dynamic 42- and 54-gate multi-agent rows each contain three episodes with 66.7% success and 33.3% collision. The single-agent mainline rows at 48, 54, and 60 dynamic gates are marked `real_eval_failed`; the 60-gate row contains 10 seeds, 0% success, and 60% collision. Those rows define the stress range of the recorded configuration.

**中文**

### 仓库中保留的历史附件

以下数字来自 2026 年 6 月 21 日生成的评测附件，每一行都和自己的样本量、场景及方法标签绑定。

单机动态 42 门对比中，`ours_mainline` 使用 10 个 seed，成功率 100%、碰撞率 0%，路径长度 `83.388 m`、飞行时间 `72.62 s`、最小净空 `0.052 m`、移动门 swept clearance `0.049 m`。同一张表中的 A* 行使用 10 个 seed，成功率 10%、碰撞率 80%，属于 planner-only 对照。

多机保留表中，静态 60 门、动态 36 门和动态 60 门各有 3 个 episode，成功率均为 100%、碰撞率均为 0%；路径长度分别为 `66.525 m`、`67.087 m` 和 `66.669 m`。动态 42 门和 54 门各有 3 个 episode，成功率为 66.7%、碰撞率为 33.3%。单机动态 48、54、60 门主方法行标记为 `real_eval_failed`；其中 60 门使用 10 个 seed，成功率 0%、碰撞率 60%，用于界定压力区间。

---

## 14 | Scientific argument

**English**

### What the experiment is designed to test

The project is organized around three testable mechanism questions:

| Question | Controlled comparison |
| --- | --- |
| Does relational state reach the policy in a stable way as team size changes? | vary active team size while keeping graph capacity and masks fixed |
| Does route-plus-formation decomposition keep team geometry interpretable? | inspect route progress, formation-position error, pair distance, and gate clearance in one record |
| Does velocity-level intervention change risky commands before they become geometric failures? | compare intervention norm, clearance, collision state, and progress under the same task contract |

The evidence is read with the operating conditions attached. Smoke tests and deterministic reproduction support the core environment contract. The retained tables support the recorded scenario outcomes. A component-level claim about Graph-SAC, Graph-MASAC, or the shield is framed through paired runs and ablations, with the component held constant everywhere else.

**中文**

### 实验要检验的机制问题

项目围绕三个可以验证的机制问题组织：团队规模变化时，关系状态能否通过固定容量图和 mask 稳定进入策略；路线与编队的分层是否让团队几何更容易解释；速度级干预是否在风险变成几何失败前改变动作。对应的比较分别控制团队规模和图容量，联合查看路线进度、编队位置误差、机间距离与净空，以及同时记录干预范数、碰撞状态和进度。

结果解释始终带着运行条件。smoke 和 deterministic reproduce 支持核心环境合同，保留的表格支持已记录场景的结果。关于 Graph-SAC、Graph-MASAC 或 shield 的组件级结论，采用配对运行和消融，把其他条件固定下来。

---

## 15 | Research contribution and engineering contribution

**English**

### What I can defend precisely

**Research contribution**

- a fixed task contract that couples route progress, desired formation positions, geometric clearance, and termination semantics;
- a variable-team graph interface that exposes agent, formation, route, obstacle, guidance, and team relations;
- a dynamic-gate runtime in which predicted gate arrival changes the desired formation geometry;
- an evaluation record that links decision state, applied action, execution state, and terminal reason.

**Engineering contribution**

- a runnable NumPy core with deterministic layouts and repeatability checks;
- modular PyTorch paths for Graph-SAC, centralized-critic Graph-MASAC, BC, DAgger, failure-aware replay, safety critic, and behavior anchor;
- vectorized training, checkpoint manifests, and optional Isaac Lab replay adapters;
- source-level diagnostics for planner calls, latency, intervention magnitude, clearance, pair distance, and formation error.

The strongest claim is the implemented and inspectable research framework. The algorithmic effect of each learning component is a separate experimental question, so the next paper-oriented step is a frozen multi-seed ablation matrix.

**中文**

### 我可以准确答辩的内容

研究贡献包括：把路线进度、期望编队位置、几何净空和终止语义放进同一任务合同；设计支持团队规模变化的图接口；让预测到达时间改变动态门通过编队；以及把决策状态、实际动作、执行状态和终止原因连接到同一运行记录。

工程贡献包括：可运行的 NumPy 核心、确定性场景和重复性检查；Graph-SAC、集中式 critic Graph-MASAC、BC、DAgger、failure-aware replay、safety critic 和 behavior anchor 的模块化 PyTorch 路径；向量化训练、checkpoint manifest 和可选 Isaac Lab 回放；以及规划调用、延迟、干预幅度、净空、机间距离和编队误差等源码级诊断。

最稳妥的表述是：这是一个已经实现、可以检查和扩展的研究框架。每个学习组件的算法效果对应独立实验问题，下一步论文实验是冻结多 seed 的组件消融矩阵。

---

## 16 | Literature anchors

**English**

### One design decision for each reference line

| Reference line | Role in AeroGate |
| --- | --- |
| Hart, Nilsson, Raphael, A* (1968) | global route for the team reference point on an inflated grid |
| Daniel, Nash, Koenig, Felner, Theta* (JAIR, 2010) | any-angle grid planner used in the comparison family |
| Karaman and Frazzoli, RRT* (RSS 2010; IJRR 2011) | sampling-based planning reference |
| Haarnoja et al., Soft Actor-Critic (ICML 2018) | maximum-entropy continuous velocity control |
| Lowe et al., multi-agent actor-critic (NeurIPS 2017) | centralized training and decentralized execution pattern |
| Jiang et al., Graph Convolutional Reinforcement Learning (ICLR 2020) | graph message passing for relational policy state |
| Tolstaya et al., spatial graph networks for multi-robot exploration (IROS 2021) | graph representation for multi-robot spatial relations |
| Zhang et al., H2GNN (IEEE RA-L 2022) | graph-based multi-robot interaction reference |
| Benzerrouk, Adouane, Martinet (RAS, 2014) | constrained virtual-structure navigation |
| Alshiekh et al., Safe RL via Shielding (AAAI 2018) | intervention layer around a learned policy |

I use these papers as design anchors. The interview explanation connects each citation to a concrete source module and an observable quantity in the run record.

**中文**

### 每条文献对应一个设计决策

A* 对应团队参考点的全局路线；Theta*、RRT* 对应经典规划对照；SAC 对应连续速度控制；Lowe 等人的多智能体 actor-critic 对应 centralized training / decentralized execution；Graph Convolutional RL、Spatial Graph Networks 和 H2GNN 对应图消息传递与多机器人关系表示；virtual-structure 文献对应编队几何；shielding 文献对应策略外部的动作干预层。

面试中不把文献当作装饰，而是把每篇文献连接到具体源码模块和运行记录中的可观测量。

---

## 17 | Defense-ready oral summary

**English**

“AeroGate Graph is a fixed-height 2D environment I built to study multi-agent coordination through static and moving gates. The team reference point follows an A* route, and a virtual-structure layer converts that route into desired formation positions for the individual agents. I represent agent state, formation targets, route look-ahead, nearby gates, and team state as a masked graph, so teams with different active sizes can use one policy interface. The shared actor outputs a planar velocity for each active agent, while a centralized critic evaluates the joint action during training.

Before integration, the action shield checks pair distance, boundaries, obstacles, and the gate channel and can correct the velocity command. The environment then records progress, clearance, pair distance, formation-position error, planner latency, intervention magnitude, and the terminal reason. For learning, I can warm-start the actor with behavior cloning from a geometric teacher, use DAgger to query that teacher at policy-visited risk states, and continue with Graph-FlashSAC fine-tuning. The repository contains single-agent and one-to-eight-agent runs, an eight-agent dynamic-gate curriculum, and evaluation artifacts with explicit seeds or episode counts. The main contribution I can defend is the controlled, inspectable research framework that connects route, formation, graph state, safety intervention, and evidence in one task contract.”

**中文**

“AeroGate Graph 是我构建的固定高度二维环境，用来研究多机团队通过静态和动态闸门时的协同。团队参考点沿 A* 路线前进，虚拟结构层把这条路线转换成每架无人机的期望编队位置。我把 agent 状态、编队目标、路线前视信息、附近闸门和团队状态表示成带 mask 的图，使不同 active team size 可以共享一个策略接口。共享 actor 为每个 active agent 输出平面速度，集中式 critic 在训练时评价联合动作。

积分前，action shield 检查机间距离、边界、障碍物和门通道，并可以修正速度命令。环境随后记录进度、净空、机间距离、编队位置误差、规划延迟、干预幅度和终止原因。学习部分可以先从几何教师进行 BC warm start，再用 DAgger 在策略访问到的风险状态查询教师，最后使用 Graph-FlashSAC 微调。仓库包含单机和 1 到 8 机运行、8 机动态门 curriculum，以及带有明确 seed 或 episode 数的评测附件。我最有把握答辩的是这套受控、可检查的研究框架：它把路线、编队、图状态、安全干预和证据放在同一个任务合同中。”

---

## 18 | Source and evidence index

**English**

### Where each statement comes from

| Statement | Primary source |
| --- | --- |
| task contract and defaults | `multi_gate/configs/experiment_config.py`, `shared/core` |
| route planning | `multi_gate/planners/global_route_planner.py` |
| formation geometry | `multi_gate/formation/virtual_structure.py` |
| graph tensors and node groups | `multi_gate/env/observation_multi.py`, `observation_runtime.py` |
| graph actor and critic | `multi_gate/graph_rl/graph_policy.py`, `graph_masac.py` |
| BC data and training | `multi_gate/imitation.py` |
| DAgger correction loop | `multi_gate/dagger.py` |
| expert controller and dynamic-gate response | `multi_gate/replay.py`, `dynamic_gate_task_slots.py` |
| action shield | `multi_gate/env/safety_shields.py` |
| failure replay | `multi_gate/graph_rl/replay_buffer.py` |
| dynamic runtime | `multi_gate/env/dynamic_gate_runtime.py` |
| eight-agent curriculum | `density_multi/scripts/train_dynamic_gate_density_8d_curriculum.py` |
| retained metrics | `evaluation_artifacts/results/csv_json/*.csv`, `*.json` |
| evaluation interpretation | `evaluation_artifacts/report/gate_graph_2d_evaluation_report.md` |

**中文**

### 每类表述的来源

任务合同和默认参数来自 `multi_gate/configs/experiment_config.py` 与 `shared/core`；路线来自 `global_route_planner.py`；编队来自 `virtual_structure.py`；图张量来自 `observation_multi.py` 和 `observation_runtime.py`；actor/critic 来自 `graph_policy.py` 与 `graph_masac.py`；BC、DAgger、专家控制器、动态运行时、安全 shield、failure replay 和 8 机 curriculum 分别对应表格中的源码入口；历史指标来自 `evaluation_artifacts/results/csv_json` 下的 CSV/JSON，解释依据评测报告。

---

## 19 | Closing slide

**English**

### From a geometric contract to a research program

I designed AeroGate Graph so that a new mechanism can be added without changing the meaning of the task record. The next experiment is straightforward to state: freeze the geometry, team-size buckets, horizon, and seeds; compare graph and non-graph policies; report the effect of the action shield and formation layer through paired ablations; and retain the complete decision-to-termination trace for every episode.

That is the research direction I would like to continue at FAIR: connect structured state estimation, active perception, multi-agent decision making, and real execution constraints in a way that remains measurable at every interface.

**中文**

### 从几何合同走向研究计划

我把 AeroGate Graph 设计成一种可以扩展新机制、同时保持任务记录含义不变的研究框架。下一步实验可以明确写成：冻结几何、团队规模 bucket、horizon 和 seed；比较图策略与非图策略；通过配对消融测量 action shield 和编队层的作用；并为每个 episode 保留从决策到终止的完整轨迹。

这也是我希望在 FAIR 继续推进的方向：把结构化状态估计、主动感知、多智能体决策和真实执行约束连接起来，并让每个接口的影响都可以被测量。
