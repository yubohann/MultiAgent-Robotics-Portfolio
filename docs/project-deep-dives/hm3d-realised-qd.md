# HM3D Realised-QD：项目技术说明

## 300 字项目概括

HM3D Realised-QD 研究 HM3D-derived 三维环境中的无目标多 UAV 协作探索，主指标是共享 CF2X、通信、安全和物理时间合同下的 `Explored-Free-Flight-Volume-AUC_time`。系统不使用手工目标点、目标数量、确认召回或 RGB-D 作为正式任务合同，而是让每架无人机从 `sparse_range_3d` 观测建立局部 belief，形成受限候选池，通过通信交换必要摘要，执行带安全检查的团队计划，并且只依据真实执行 receipt 更新行为多样性。代码将 contracts、observation、realised-QD、RL、runtime、safety、evaluation 和 Isaac adapters 分开；P0-P10 文档、训练/基线公平预算、控制器冻结和结果 schema 作为实验治理层。realised-QD 的关键不是离线生成漂亮的 archive，而是把候选质量和行为描述绑定到已完成的物理轨迹、覆盖体积、时间和失败原因，避免“尚未执行的计划”污染 QD。Isaac/PhysX 运行必须通过项目包装器并绑定本地 CF2X USD；HM3D 资产、私有评测、checkpoint、raw log 和结果留在本地。项目适合展示多智能体探索、QD、强化学习和安全执行的结合，同时必须把 unit/property/replay 证据与正式论文主表严格分离。

## 项目定位

- **任务**：无目标、在线、三维协作探索；HM3D-derived 场景提供几何环境。
- **观测**：正式 profile 为 `sparse_range_3d`；`physics_only` 仅用于 H15 吞吐对照。
- **指标**：以执行后的自由飞行探索体积随物理时间的 AUC 为主，不混入 ObjectNav completion/SR/SPL。
- **执行对象**：四架 CF2X；真实 Isaac/PhysX 合同与通信/安全 profile 共同决定资格。

## 技术架构

```text
HM3D-derived scene + sparse_range_3d
          -> local belief / frontier candidates
          -> bounded communication and team candidate pool
          -> realised-QD archive proposal
          -> guarded CF2X execution
          -> receipt / coverage / failure reason
          -> archive update and evaluation
```

realised-QD 更新必须等待执行回执；安全层检查边界、碰撞、通信新鲜度、物理时间和取消条件。RL 训练和外部 baseline 都需使用同一控制器、动作预算、观测 profile、场景划分和统计脚本。

## 核心代码结构

| 路径 | 作用 |
| --- | --- |
| `src/realised_qd/contracts/` | episode、observation、action、通信、版本和证据合同 |
| `src/realised_qd/runtime/hm3d_realised_qd.py`、`src/realised_qd/archives/qd.py` | 行为描述、archive、候选、执行回执驱动的更新 |
| `src/realised_qd/rl/` | 策略、训练、低标签/消融和 baseline 适配 |
| `src/realised_qd/runtime/` | CF2X 运行、时间、通信和任务循环 |
| `src/realised_qd/safety/` | 边界、碰撞、过期消息、取消和失败降级 |
| `src/realised_qd/evaluation/` | 覆盖体积、AUC、统计和结果 schema |
| `scripts/` | 组装、审计、训练、回放、Isaac 启动和报告生成 |
| `configs/`、`manifests/`、`tests/` | HM3D profile、实验绑定、泄漏/property/performance/integration 测试 |

## 开发过程

1. 先清理旧 ObjectNav/目标搜索合同，冻结无目标探索任务和 `sparse_range_3d` profile。
2. 建立 CF2X、通信、物理时间、取消和安全 contracts，再实现局部 belief 与候选池。
3. 将 QD archive 更新改为 receipt-gated，分别记录 coverage、time、failure 和 diversity。
4. 固定控制器、训练预算、baseline 和场景 split，形成 P0-P10 门槛与公平评测。
5. 通过 IsaacLab wrapper、回放和统计脚本输出论文级表格；私有 HM3D 资产与 evaluator 数据不进入发布包。

## 涉及知识点

- HM3D 场景、三维占据/coverage 与 sparse range sensing
- 多 UAV frontier exploration、局部 belief 和有限通信
- Quality-Diversity、archive、行为描述和执行后更新
- 多智能体 RL、候选策略、baseline 公平预算
- CF2X/PhysX、碰撞/边界/时间和安全降级
- 统计 schema、manifest、P0-P10 gate 和 leakage tests

## 测试与验证

```powershell
uv sync --extra dev --extra rl --extra hm3d
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m ruff check src tests scripts
```

Isaac 运行使用 `scripts/run_isaac_python.ps1` 和经验证的 IsaacLab interpreter，并设置 `AEROCITY_CF2X_USD`。单元测试、静态检查或回放通过只证明对应合同；正式主表还需相同场景、预算、控制器、观测 profile、版本和统计脚本。

## 学术申请中的项目价值

该项目适合展示多智能体探索、QD 和真实执行约束的结合。最有分量的技术点是把“候选多样性”与物理轨迹 receipt 绑定，并把通信新鲜度、失败原因和安全边界纳入评估；这比只展示 archive 图更能说明研究严谨性。

## 运行入口

- 测试：`tests/`
- 训练/评测脚本：`scripts/`
- Isaac 入口：`scripts/run_isaac_python.ps1`
- 活动合同索引：`docs/README.md`

## 证据边界

HM3D-derived 场景、私有评测、CF2X USD、checkpoint 和 raw logs 不是公开源码输入；本地 wrapper 通过不等于真实四机长任务完成。文档中的 P0-P10 是门槛和执行计划，只有绑定到具体 manifest、receipt 和结果文件的项目才可进入正式主表。

## 源码审计记录

| 层次 | 代表目录/文件 | 复核内容 |
| --- | --- | --- |
| 合同 | `src/realised_qd/contracts/` | observation/action、scene、episode、通信、时间、版本、失败和证据字段。 |
| 观测 | sparse-range adapters、belief modules | 稀疏 range 的单位、视场、时间新鲜度、局部地图和 padding/mask。 |
| 候选 | `realised_qd/` | frontier/候选生成、行为描述、archive、候选池和多样性计算。 |
| 执行 | `runtime/`、Isaac wrappers | 四机同步 step、物理时间、通信、CF2X action、receipt 和取消。 |
| 安全 | `safety/` | 边界、碰撞、队间距离、过期消息、任务超时和失败降级。 |
| 学习 | `rl/`、training scripts | policy、baseline、训练预算、checkpoint、控制器冻结和 ablation。 |
| 评测 | `evaluation/`、统计脚本 | free-flight volume、AUC_time、失败类别、置信区间和 schema。 |
| 证据 | `configs/`、`manifests/`、`docs/` | P0-P10 gate、场景 split、版本绑定、公平预算和主表边界。 |
| 测试 | `tests/` | unit/property/leakage/performance/integration、replay 和 contract 回归。 |

## 一次 realised-QD episode 的调用链

1. 运行 manifest 固定 HM3D scene revision、CF2X 资产、观测 profile、控制器、通信和 physical-time budget。
2. sparse range adapter 将传感器消息转为局部 belief，belief 只包含策略允许看到的几何和新鲜度信息。
3. frontier/candidate generator 产生若干任务候选，通信层只交换 bounded summary，不广播私有真值。
4. QD 计算候选的行为描述和质量估计，但候选尚未执行时不能写入正式 archive。
5. safety/runtime 选择可执行候选，推进四机 PhysX，记录轨迹、覆盖、时间、通信和失败原因。
6. receipt validator 通过后才更新 realised archive；evaluation 计算 coverage-volume curve、AUC_time 和失败分层。

## P0-P10 的阅读方法

P0 应先回答任务/观测/控制器/时间和数据边界是否可运行；中间 gate 检查 QD 有效性、候选丰富性、通信和安全是否有可证伪证据；P10 只汇总满足前置条件的主表。历史计划、retired contract 和当前活动合同分开存放，不能用旧 ObjectNav 文档中的 target recall、目标数量或 completion 重新定义当前任务。

## 公平评测矩阵

主方法、无 QD、neutral/shuffled、外部 baseline 和 RL candidate 必须共享场景划分、控制器、观测字段、动作预算、通信限制、物理时间和统计脚本。任何方法若使用不同传感器 profile、不同 episode cap 或不同成功定义，应单列为 transfer/diagnostic，不能和主表直接平均。

## 失败与恢复复核

需要区分 collision、out-of-bounds、stale communication、no valid candidate、controller timeout、asset load、episode deadline 和 evaluator rejection。每种失败都应有停止动作、重试上限、恢复入口和 receipt。尤其不能把“候选生成成功”当作“探索完成”，也不能在执行前用 predicted coverage 更新 archive。

## 交付检查表

- [ ] HM3D/CF2X 私有输入未进入公开包和公共观测。
- [ ] archive 更新只由真实执行 receipt 触发。
- [ ] 主方法、baseline、ablation 使用相同 budget/controller/statistics。
- [ ] P0-P10、历史合同和最终主表互相可追溯。
- [ ] AUC_time、coverage、通信和失败分层都保留原始记录。

## 原理与理论基础

### 1. 无目标探索的 POMDP

任务状态包含三维占据、无人机位姿、通信状态和已探索区域，策略只能看到 sparse range 和局部 belief，因此仍是 POMDP。候选 frontier 或 route 不是目标真值，而是根据当前 belief 的动作提议。主指标用探索自由飞行体积随时间的曲线 A(t)，再计算 AUC_time=integral A(t)dt；它同时奖励覆盖和尽早覆盖，避免只看最终体积。

### 2. Quality-Diversity 的 archive

QD 不只寻找一个最高 reward 策略，而是在行为描述空间中保存多个有差异的解。候选可由描述向量 b(x) 表示，例如覆盖方向、飞行高度、通信占用或风险模式；质量 q(x) 可以是覆盖体积/时间。realised-QD 的关键约束是 q 和 b 必须来自真实执行 receipt，不能用未执行的预测 coverage 直接占据 archive，否则 archive 会充满“计划质量”而非“执行质量”。

### 3. 通信和安全约束

多机通信可以抽象为有延迟、有丢包的消息通道 m(t,age,version)。候选池只接受版本、任务、地图和位姿一致且 age 在阈值内的消息；过期的 receiver/route/pose 不能推进任务。安全控制将动作投影到边界、碰撞、队间距离、速度和物理时间约束集合；集合为空时进入停止或恢复，而不是隐式换用另一个任务。

### 4. 公平实验与统计

主方法、neutral/shuffled QD、无 QD、RL 和外部 baseline 必须共享 scene split、controller、observation profile、通信预算、episode cap 和统计脚本。AUC、coverage、成功闭环和失败原因应按 seed 报告，均值之外保留方差/置信区间和失败样本；不能把 completion、SR、SPL 或其他任务定义混入当前表格。

### 5. 理论假设和失效模式

系统假设 sparse range 足以支持局部决策、CF2X 动力学与运行时一致、receipt 可靠且通信版本可验证。场景空旷、传感器稀疏、通信过期、控制器饥饿、资产载入失败和错误的 archive 更新都会使结果失真，所以 P0-P10、leakage/property 测试和严格 replay 是方法理论的一部分。

---

## 六、阅读范围、证据等级与文档边界

这份说明不是把 README 改写成宣传页，而是依据当前工作区的源代码、配置、实验计划和测试文件整理出的技术导读。阅读范围包括 `src/realised_qd/` 的合同、运行时、评测、学习和适配层，`configs/`、`manifests/`、`scripts/`、`tests/` 以及 `docs/` 中的正式计划和历史结果。`archive/` 保留了早期方向、旧合同和失败实验，作为研究过程的可追溯记录，但不应被误认为当前在线入口。

文中使用三种证据标签：

1. **源码事实**：可以直接在当前 Python 文件、JSON 合同或测试断言中找到的字段、常量和控制流程。例如 26 条稀疏射线、四架 CF2X、0.25 m belief 体素和 AUC 的梯形积分都属于这一类。
2. **历史附件证据**：已生成的日志、旧实验表或 P0-P10 记录。它们必须同时绑定场景、种子、预算、版本和日期。历史结果可以说明某个组件曾经怎样工作，不能自动升级为完整主方法结论。
3. **计划或建议**：尚未完成的 P07、P08、P09、P10、完整 `RB-SF-SAC` 在线接入和 RFG 正式收益检验。此类内容使用“应当”“计划”“待验证”，不能写成“已经实现并胜出”。

特别需要分清三件事：

- HM3D-derived 是几何场景来源，不等于任务使用了完整 HM3D 网格；策略只接收合同允许的公共稀疏测距结果。
- Isaac/PhysX 的碰撞几何可以被评测器用于安全审计，但策略和 QD 描述符不能读取评测器的 ESDF、自由空间掩码或剩余体积真值。
- 项目内出现了 `UAV`、`CF2X` 和多机通信等术语，论文或申请材料中应解释它们是仿真平台和多智能体几何探索的载体，而不是把项目描述成真实飞行系统验证。

## 七、完整目录地图：每一层解决什么问题

### 7.1 合同与数据模型

`src/realised_qd/contracts/` 是整个项目的边界层。`exploration.py` 定义探索任务中的观测、候选、单机计划、执行回执和 transition；`models.py` 保存场景、片段、时间段和 manifest 的不可变数据结构；`hm3d_public_schema.py` 将公共观测、通信和自由空间字段限制在可公开重放的范围内；`privacy.py` 拒绝目标坐标、评估器掩码和私有几何进入策略输入；`snapshot.py` 为 episode 状态提供可复现快照；`io.py` 负责规范化 JSON 和 SHA-256 内容摘要。

合同层的价值在于，算法函数不再以一个无类型的 `dict` 互相传递信息。一个候选必须有 `candidate_id`、`manifest_hash`、`belief_version_sha256`、可执行性和拒绝原因；一个执行结果必须有 `outcome_id`、时间范围、实际轨迹、观测 ID、消息投递 ID、失败类别、真实描述符和最终摘要。缺少这些字段时，结果在组装正式表格之前就会被拒绝。

### 7.2 运行时层

`runtime/` 是从公开观测到物理回执的主链：

- `hm3d_belief.py`：幂等地接收稀疏 range，维护 occupied/free/unknown 体素并生成内容哈希。
- `hm3d_frontiers.py`：在 unknown 与 free 的边界上聚类三维 frontier，并生成候选观察位姿。
- `hm3d_candidates.py`：把 frontier 转换为观察、区域访问、路线推进、保持和中继等候选，形成团队级候选池。
- `hm3d_team_collaboration.py`：检查团队轨迹的平移不变性、时间一致性、队间距离和公共地图贡献。
- `communication.py`：按照范围、视线、时延、丢包和 TTL 建立可审计的中继消息队列。
- `hm3d_trajectory.py`：生成 rest-to-rest 或允许中间点通过的时间参数化轨迹，并检查速度、加速度和角点条件。
- `hm3d_cf2x_execution.py`：单集群的路径守卫、控制器、推力分配、传感器节拍、碰撞检测和执行回执。
- `hm3d_cf2x_vectorized_execution.py`：在一个共享 PhysX step loop 中运行多个隔离集群，以降低批量实验的墙钟成本，同时为每个集群保留独立 manifest、token 和结果。
- `hm3d_start_resets.py`、`hm3d_calibration_geometry.py`：处理起点集合、重置审计和几何标定；它们不负责修改策略的目标定义。
- `resource_guard.py`、`process_boundary.py`、`tokens.py`、`physx_query_cache.py`：限制资源、进程和预计算结果的使用范围，避免一份旧结果被错接到新的 manifest。

### 7.3 评测与实验治理层

`evaluation/` 不是“运行完再打印一个分数”的后处理脚本，而是把实验是否有资格进入主表作为程序约束：

- `hm3d_exploration_metrics.py` 计算物理时间上的探索曲线、AUC、最终覆盖、能耗、通信和失败率。
- `hm3d_exploration_contract.py` 校验指标的单位、单调性、归一化和分母冻结。
- `hm3d_safety.py` 对静态碰撞、轨迹管、机间分离和同步轨迹做几何复核。
- `hm3d_preflight.py` 顺序审计 P01–P10，拒绝场景、种子、观测、预算、原始结果或方法矩阵漂移。
- `hm3d_p07_matrix.py` 负责任务有效性和跨层执行证据；`hm3d_p08_qd_matrix.py` 负责 `no_qd`、`planned_qd`、`realised_qd` 的配对 QD 证据。
- `hm3d_evidence_classification.py` 和 `ledger.py` 为每个结果标注证据类别、来源和摘要，避免把静态 wrapper 测试冒充 Isaac 实测。
- `hm3d_outcome_dataset.py` 将经过身份和哈希校验的 outcome 转成训练或统计数据集。

### 7.4 学习、片段和外部基线

`learning/` 包含 `replay.py`、`selector.py`、`rb_sf_sac.py`、`masked_ppo.py` 和 `vanilla_sac.py`。其中 `replay.py` 强制候选集合、合法动作掩码、持续时间、终止状态和维度一致；`selector.py` 把策略概率映射为实际候选索引；`rb_sf_sac.py` 是带候选掩码、持续时间折扣、成本 critic、successor feature 和自适应熵系数的离散 SAC 内核。`masked_ppo.py` 与 `vanilla_sac.py` 作为对照或兼容实现存在，不应在没有同一预算和控制器的情况下直接与主方法比较。

`fragments/` 记录可重放片段的来源、前缀、场景和 outcome 哈希。RFG（Outcome-Gated Fragment Reuse）在研究设计中被定义为回执门控的片段复用：只有真实完成并通过安全审计的轨迹片段才获得可复用信用；负迁移或新回执否定旧信用时，信用必须撤销。当前代码已经具备 provenance 和 `reusable_fragment_count` 等数据基础，但正式 RFG 收益、负迁移率和全链路在线接入仍属于待完成的实验项。

`adapters/` 将统一合同接到不同执行或论文基线：`hm3d_execution.py` 是片段执行样本适配，`hm3d_baselines.py` 提供公共价值和保守时序模型，`hm3d_external_baselines.py`、`hm3d_marl_ipp.py`、`hm3d_marvel.py`、`hm3d_marvel_author_sac.py` 和 `hm3d_single_rl.py` 负责外部方法或单机 RL 的接口层。`vendor/marvel_icra2025/` 保存带许可证和来源说明的局部参考实现；它的存在不代表整个外部论文已经被无条件复现。

## 八、从 manifest 到 episode 的实现步骤

### 步骤一：冻结输入与证据身份

运行开始前由 manifest 固定 scene revision、几何哈希、四机资产、起点布局、传感器 profile、通信合同、控制器 ID、物理时间预算、随机种子和统计版本。每个衍生文件都记录父 manifest 的 SHA-256。这样做的原因是，三维探索的结果对起点、传感器视场、通信半径和控制器速度极其敏感；只改变其中一项就不能继续沿用旧的主表行。

### 步骤二：初始化公共 belief

每架 CF2X 绑定一个 `SparseVoxelBelief(scene_id, agent_id, resolution_m=0.25)`。reset 阶段可以产生 bootstrap 观测，但后续回放必须按 `source_observation_id` 幂等处理：同一帧重复送入不会重复增加体素计数，也不会制造额外覆盖。观测时间戳、传感器 profile 和 source ID 被保存在 belief 的内容哈希中，供候选和 outcome 引用。

### 步骤三：整合成功通信

当前本地 belief 与收到的队友增量合并时，只有成功投递、未过期、版本正确的消息可以进入公共状态。occupied 优先于 free，防止一架机器早先标记的空闲体素覆盖另一架机器后来发现的障碍。失败、丢弃和过期消息仍写入通信审计，但不能影响下一轮候选生成。这一步把“网络发生了一个消息事件”和“策略真的获得了共享信息”区分开。

### 步骤四：生成三维 frontier 与候选

frontier 由未知体素邻接已知 free 区域得到。聚类后保留中心、外法向、未知体素数量、高度带、候选观察点和预期公共增益。候选层随后创建单机 `AgentExplorationPlan` 和团队 `TeamExplorationCandidate`，每个候选保存计划意图描述符、上下文哈希、候选成本、路线和 admission reasons。候选生成只负责提出可能的动作，不等于路线已经通过物理执行检查。

### 步骤五：执行前的多重守卫

进入 selector 前，候选至少要通过以下检查：起点和终点在三维飞行空间内；折线路径的每个采样点满足静态 clearance；轨迹管满足 `FLIGHT_CLEARANCE_M=0.30 m` 加 tracking margin；所有机对的参考路径满足计划队间距；通信或中继约束没有使用已过期的位姿；时间段和前序交通预约不冲突。合法候选数至少为 1 才允许推进；如果为空，记录 `no_valid_candidate`，而不是偷偷切换到一个未审计的动作。

### 步骤六：物理 step loop

单机执行器使用时间参数化的梯形参考和 SO(3) 反馈跟踪器。推力分配矩阵使用 CF2X 的四旋翼力矩臂和反扭矩符号；每一步记录位置、速度、加速度、姿态、转子推力、接触力、碰撞、越界、传感器时间戳和通信状态。航点不是简单地“距离小于阈值就跳过”：最终航点需要位置误差、速度近零和 settle 条件；中间航点可以在受控速度下通过，但最终段仍要用 rest-to-rest 条件，以免角点处的残余速度破坏已认证的 clearance。

向量化执行器将多个独立集群放入共享 PhysX 循环；它共享计算节拍而不共享 outcome。`PrecomputedClusterExecutionBackend` 只允许匹配完全相同的 `manifest_hash` 和 token digest 的结果被消费一次，从接口层阻止“把一个集群的成功回执接到另一个实验”。

### 步骤七：组装 execution receipt

执行结束后，以每架机器的实际轨迹、观测帧、投递消息、自由体素增量、碰撞与守卫干预构造 `ExplorationExecutionOutcome`。receipt 中同时放入 candidate hash、decision ID、起止物理时间、是否中止、不可恢复失败类型和 outcome digest。重要的是，receipt 是对“发生了什么”的记录，不是对“计划应该发生什么”的复述。

### 步骤八：指标与 archive 更新

评测器使用 outcome 和冻结的 evaluator denominator 计算自由飞行体积曲线。只有 outcome 通过身份、时间、观测、失败和安全字段校验后，才允许计算 realised descriptor；只有执行可行且 descriptor/quality/cost 都完整的结果才可以进入 QD archive。训练 transition 的 `state_hash`、`candidate_set_hash`、`selected_candidate_hash`、`outcome_hash` 和 `next_state_hash` 必须互相链接，确保后续 replay 能追溯到原始回执。

## 九、公共观测合同与信息隔离

### 9.1 为什么使用稀疏 range

正式 profile `sparse-range-3d-vfov90` 每个更新包含 26 条稀疏射线、最大量程 20 m、10 Hz 更新频率。它刻意保留“几何可探索性”而不提供完整场景网格；算法必须从少量测距结果逐步构造自己的公共 belief。`physics_only` 可以用于传感器吞吐或执行 pilot，但不能把它当作正式策略观测。

观测窗口由 `reset_bootstrap`、`transit`、`observe`、`dwell` 区分。每个窗口都要绑定 source observation ID 和阶段，防止把 reset 帧、过渡帧和停留帧混成相同频率的独立样本。范围方向由 `range_sensing.py` 的固定模式解析，不能运行中按结果动态增加射线。

### 9.2 belief 的状态转移

可将一帧公共观测写成：

\[
  B_{t+1}=\operatorname{merge}(B_t, O_t, D_t^{\mathrm{delivered}}),
\]

其中 `B_t` 是稀疏体素状态，`O_t` 是本机成功观测，`D_t` 是成功投递且未过期的队友增量。合并不是平均概率，而是按 occupied 优先的确定性规则。unknown 不等于 free；只有命中射线内的体素可以被写为 free，射线末端或碰撞命中体素写为 occupied。

belief 的 `content_sha256` 使候选上下文可重放。若同一 source ID 在不同 payload 下重复出现，或 scene/agent/resolution 不一致，回放应直接失败，而不是静默覆盖旧数据。

### 9.3 禁止的捷径

以下信息可以供 evaluator 检查结果，但不能进入策略、QD 描述符或训练 transition：完整 HM3D mesh、私有 ESDF、真值自由体积、目标位置/数量、评估器 free-flight mask、未执行动作的反事实覆盖。这样设计使实验回答的是“在稀疏公共感知下是否能有效探索”，而不是“知道地图后如何走最优路径”。

## 十、三维 frontier、候选与团队动作空间

### 10.1 frontier 的几何意义

二维 frontier 通常是占据栅格中未知和已知 free 的边界；本项目将其扩展到三维体素。每个 cluster 记录 centroid、outward normal、unknown voxel count、relative height band 和 viewpoint candidates。centroid 不是目标点，只是一个用于生成观察姿态的几何代表；normal 提供离开已知区域的方向提示。

### 10.2 候选类型

候选生成器允许将局部 frontier 组织为多种团队动作：

- `observation`：移动到 viewpoint 并在规定 dwell 时间采样；
- `route_progress`：沿一条已经通过守卫的路线推进；
- `region_access`：进入未充分观测的局部区域；
- `hold`：在通信或交通冲突时保持安全位置；
- `relay`：为公共地图或消息传递提供中继位置。

每个候选不是单一坐标，而是四个 agent plan 的集合。候选的 planned descriptor 只描述意图，例如预期高度变化、终点分散和方向互补；它可以用于排序、预测和诊断，但不能冒充真实行为 descriptor。

### 10.3 团队守卫和公平候选池

候选池先进行可行性筛选，再交给 no-QD、planned-QD、realised-QD 或 RL selector。这样做很重要：如果某个方法看到的候选池不同，最后的差异可能来自动作可行性而不是选择策略。P08 的公共 value backbone 由所有 QD 对照共享；QD 只能在公共价值近似相当的合法候选之间引入多样性偏好。

## 十一、通信模型：可复现的网络合同，而非无线电结论

`communication.py` 用范围加视线建立无向 relay graph。默认 P07 合同给出最大通信距离 10 m，消息基础时延 0.05 s，每跳追加 0.02 s，TTL 0.5 s，telemetry 频率 10 Hz。`RelayGraphSnapshot` 保存邻接矩阵、连通分量、直接链路数和最大跳数；图断开时明确记录 partition，而不是假定全队仍然共享地图。

消息队列的投递流程如下：

1. 发布者创建带 source timestamp、payload digest 和 TTL 的 `RelayMessage`。
2. 队列按照当前图寻找最短中继路径；没有路径、时延尚未到期、确定性丢包或 TTL 已过期时，形成对应的 `DROPPED` 或 `EXPIRED` outcome。
3. 只有满足时延且有路径的消息产生 `RelayDelivery`，其中包含 hop path、delivery timestamp 和 payload digest。
4. episode 结束时，仍未解决的接收者被显式结算为 horizon expiry。

丢包使用消息、接收者和 payload digest 的确定性哈希抽样，确保同一 manifest replay 得到相同结果。该模型可以研究新鲜度、断连和中继带来的算法影响，但不能据此声称已经建立真实室内 RF 传播模型；文档中应称为“可重放的范围/视线消息合同”。

## 十二、CF2X/PhysX 执行与安全理论

### 12.1 路径净空与轨迹管

规划中心线不能直接作为实际飞行轨迹。代码同时使用物理 clearance、tracking envelope 和采样 Lipschitz reserve：静态几何要求至少 0.30 m，规划阶段还保留 0.20 m 跟踪余量；队间物理根间距为 0.50 m，联合计划会为两台机器各自的跟踪误差再增加余量。路径按 0.10 m 采样，并对采样点和端点分别查询几何净空。

这是一个典型的“规划安全”和“执行安全”分离：规划器只保证参考路径有足够裕量，物理 step loop 仍然在每一步检查真实 root trace、接触力和越界。一旦参考路径安全而实际跟踪偏离，receipt 必须记录具体 collision/clearance violation，不能用规划器的通过结果覆盖物理结果。

### 12.2 时间参数化与控制器

`hm3d_trajectory.py` 计算 rest-to-rest 的最小持续时间，执行器使用最大参考速度 1.0 m/s、最大参考加速度 0.8 m/s² 的时间合同。Bitcraze Lee 和 Mellinger 控制器以来源 URL、commit、控制频率和控制器 ID 固定身份；项目的 SO(3) 反馈层还记录姿态、推力、速度和位置误差。控制器的 feedback authority 与 planner 的 reference limit 分开，避免把“规划最大速度”误当成“控制器能修正的最大加速度”。

航向采用限速参考，默认独立 heading reference 不超过 10 deg/s；直接把 180 度航向跳变送给 SO(3) tracker 会消耗姿态余量，因而被作为控制层错误处理。最终航点需要 `WAYPOINT_SETTLE_SPEED_MPS=0.05` 和位置误差约束；这些值属于当前执行 ABI，修改后必须重新做 timing calibration，不能和旧 outcome 混用。

### 12.3 碰撞、越界和取消

接触力达到硬失败阈值、静态查询命中障碍、飞行器越过边界、通信状态违反候选合同或 episode deadline 到期时，执行器会停止当前片段并把原因写入样本。取消是有界状态转移，不是简单抛异常：必须保留已经执行的轨迹和已产生的 observation/message outcome，以便分析失败发生在 transit、observe 还是 dwell 阶段。

## 十三、执行回执与可追溯性

一次 outcome 最少连接四类身份：

1. **任务身份**：episode、decision、candidate 和 manifest hash；
2. **物理身份**：agent、控制器、执行 backend、轨迹和时间；
3. **感知身份**：source observation IDs、公共 range outcome IDs、belief version；
4. **网络身份**：message IDs、delivery IDs、payload digest 和 TTL 结算。

`ExplorationTransition` 再把 outcome 接到下一状态。这样可以回答“这个 archive cell 是由哪一段实际轨迹产生的”“这个训练样本是否使用了过期消息”“某次失败是否在相同 geometry 上可重放”等问题。

可靠性规则包括：

- outcome digest 对规范化字段计算，字段顺序和浮点格式不能随意变化；
- manifest、controller、sensor 和 evaluator denominator 的哈希必须一致；
- 预计算后端只能消费一次且必须匹配 token；
- 失败 outcome 不从账本删除，必要时只在训练数据过滤阶段按规则排除；
- 一个 receipt 不能同时声称 `executed=False` 和拥有完整物理轨迹，也不能同时 `terminated=True` 与 `truncated=True`。

这些规则是研究可复现性的组成部分，而不是额外的工程装饰。

## 十四、realised-QD 的算法细节

### 14.1 描述符与质量的分离

当前正式 archive 是 4×4×4 的三维离散空间，共 64 个名义 cell。三个轴为：

1. `vertical_motion_ratio`：真实轨迹中垂直运动的比例；
2. `team_spatial_dispersion`：团队实际空间分散程度；
3. `public_observation_complementarity`：各机公共 free-voxel footprint 的互补程度。

描述符回答“这个团队实际表现出了哪一种行为模式”，质量回答“这次执行为任务带来了多少有效公共探索、花费多少时间和能量”。两者必须分开。一个安全但覆盖增益小的轨迹可以占据某个稀有行为 cell，但不应因此被误报为高质量；一个覆盖很高但行为模式重复的结果也不应单独证明 archive 丰富。

`RealisedQDDescriptor` 强制三维数值均在 `[0,1]`。`OutcomeQDFeatureVector` 额外保存若干预注册的诊断轴，例如公共垂直观测跨度和 unique contribution balance，但部署仍冻结三轴 archive，避免看到验证结果后不断扩充维度。

### 14.2 archive 接纳规则

`archives/qd.py` 中的 `QDArchive` 以 descriptor cell 为索引，每个 cell 保存 `Elite`：candidate ID、manifest hash、behavior hash、realised descriptor、quality、cost、feasible、source 和重复评估统计。空 cell 可接纳第一个合法 elite；同 cell 只在质量更高或成本更低时替换；相同 behavior hash 的重复结果被拒绝或作为 reevaluation 记录。archive revision 和 checkpoint hash 用于回放和跨进程校验。

在线选择前的准入门槛是至少 12 个真实 outcome、至少 6 个 occupied cells、Shannon effective cells 至少 4.0，轴绝对相关性不超过约 0.90，并且行为 footprint 在 cell 内外具有可分性。门槛的含义是“档案足够有信息支持选择”，不是“方法已经比 baseline 好”。

### 14.3 richness 审计为什么必要

只统计 occupied cells 很容易制造假丰富：三个高度相关的标量也能把少量样本撒到多个 cell。项目因此加入轴相关性、协方差行列式、Shannon effective cells、跨场景支持、footprint separation、重放一致性和 calibration mode contrast。若所有结果都来自一个场景、同一条局部路线或某个单调速度变量，richness 审计应失败。

### 14.4 计划意图与执行行为

`PlannedQDSelector` 的作用是控制诊断：它根据计划 descriptor 提前写 archive，正好展示为什么“按计划写入”会造成泄漏和虚假多样性。它只能作为 `planned_qd` 对照，不能作为 realised-QD 正式方法。

`OutcomeGroundedQDSelector` 则遵循如下规则：计划 intent 只进入历史预测器；执行结束后，只有 `execution_feasible=True` 且 receipt 完整的 outcome 才调用 `observe(...)`；碰撞、越界、未完成片段和无效时间段不得进入 predictor 或 archive；训练记录还要保留 outcome digest。由此形成的核心不变量是：

\[
  A_{t+1}=A_t\cup\{(b(\eta_t),q(\eta_t),c(\eta_t))\}
  \quad\text{且仅当 }\eta_t\text{ 已真实执行并通过回执审计。}
\]

## 十五、意图到结果的预测、公共需求与 abstention

QD 并不是在每个时刻都应该介入。`PublicExplorationNeed` 从当前公共 belief 和各 agent free voxel footprint 计算三维需求：垂直探索缺口、空间分散缺口和重复观测缺口。垂直项使用高度带熵，空间项使用 free voxel 的 xy radial spread，重复项使用 agent footprint 的 Jaccard overlap。它是 selector context，不是 archive descriptor，也不是 evaluator score。

intent-to-realised predictor 使用 archive 中与计划 intent 相邻的历史真实 descriptor，估计 \(P(\text{realised descriptor}\mid\text{intent})\) 及不确定性。当前公共需求强度低于 0.15，或者预测不确定性高于冻结阈值时，selector 应 abstain，回退到公共价值 backbone；即使 QD 介入，也只能在公共 gain/cost 近似相当的候选中改选。这样可以避免为了填充 archive 而牺牲当前任务收益。

这套设计对应两个研究假设：第一，真实执行反馈可以让行为档案比计划意图更可信；第二，QD 的收益取决于当前地图缺口，而不是在所有状态持续施加多样性偏好。P08 应分别报告 abstention rate、改选率、need alignment、预测误差和 paired AUC，而不是只报告最终一个均值。

## 十六、强化学习内核与训练数据

### 16.1 候选动作而不是固定坐标动作

策略的动作空间是当前合法候选的索引。每个 transition 保存 context、候选特征矩阵、合法 mask、选择的 action、reward、cost、行为特征、next context、next candidates、duration、done 和截断状态。`pad_candidate_batch` 将不同 episode 的候选数补齐，但 padding 永远是非法动作；如果 action 指向 padding 或非法候选，transition 构造会失败。

这种动作设计比固定的离散方向更贴合三维探索：候选数量随 frontier 和通信状态变化，策略可以在 observation、relay、hold 等不同类型中选择；同时，通过合法 mask 保证学习器不能绕过几何和安全守卫。

### 16.2 RB-SF-SAC 的更新

`RBSFSAC` 使用 actor、双 Q critic 和 target network。对每条 transition，折扣因子按持续时间计算：

\[
  \gamma_{\Delta t}=\gamma^{\Delta t},\qquad
  y=r+(1-d)\gamma_{\Delta t}V(s').
\]

双 critic 取最小值以减轻过估计；actor 在合法动作的归一化概率上计算熵正则；`target_entropy_ratio` 根据合法动作数调整目标熵。若启用 cost critics，策略同时学习执行成本，成本超过 `cost_limit` 时由对偶乘子增大约束权重。若 `sf_dim>0`，successor feature critics 预测行为特征的持续折扣和，用于把当前偏好映射到长期行为结果。

训练后保存的 checkpoint 包含 schema version、完整配置、网络和 optimizer 状态、随机数状态、更新步数以及内容 hash。加载时必须逐项验证配置、启用的 cost/SF 模块和 hash；这保证不同候选维度、折扣或安全约束的 checkpoint 不会被静默复用。

### 16.3 数据过滤与 replay

训练数据应从 outcome dataset 生成，而不是从规划日志直接拼接。失败回执可以保留用于 cost、failure attribution 和安全学习，但不能在没有标签说明的情况下当作成功覆盖样本。重放时必须重新检查 candidate set hash、合法 mask、source observation identity 和 outcome digest；任何字段漂移都应使 replay 失败。

## 十七、片段复用 RFG 的研究思路

RFG 的目标不是把一条旧轨迹简单复制到新场景，而是判断一段已经执行过的局部行为能否在当前公共状态下安全复用。一个候选片段至少要匹配：起点区域和几何上下文、路线前缀、观测 profile、控制器版本、通信状态和安全合同。复用成功后，新的 receipt 仍然是唯一有效的信用来源。

建议的 RFG 记录包括：片段来源 outcome hash、可复用前缀、适用条件、实际复用次数、通过/拒绝原因、positive transfer gain、negative transfer count 和撤销时间。RFG credit 可以作为 candidate feature，也可以作为 realised quality 的附加项，但只有片段真的执行、真的完成且没有违反安全合同时才计入。P08 需要把“有片段 provenance 记录”和“RFG 已证明提升任务指标”分开报告。

## 十八、P0–P10 实验设计与阶段出口

| 阶段 | 要回答的问题 | 主要证据 | 当前解读 |
| --- | --- | --- | --- |
| P01 | 许可证、资产、场景和边界是否锁定 | 来源、scene/geometry hash、合同 | 已有基础证据 |
| P02 | Isaac/HM3D reset 是否可重复 | A-B-A reset、起点和恢复日志 | 已有基础证据 |
| P03 | 三维飞行空间是否真实可用 | 双面碰撞、ESDF/clearance、连通高度带 | 已有基础证据 |
| P04 | 公共稀疏测距接收器是否正确 | 26-ray、时间窗、幂等 replay | 已有基础证据 |
| P05 | train/validation/test split 是否冻结 | 场景清单、split hash | 已有基础证据 |
| P06 | 正式 sensor profile 是否有真实吞吐 | H15 pilot、丢帧/延迟/可见率 | 待重跑或补齐 |
| P07 | 四机跨层链路是否完成 | public observation、通信、候选、执行、receipt | 未完成 |
| P08 | realised-QD 是否比 controls 更有证据 | paired no/planned/realised、richness、alignment、AUC | 组件证据已有，正式闭环未完成 |
| P09 | 方法和统计合同是否冻结 | 独立 seed、alpha/power、protocol hash | 未完成 |
| P10 | holdout 主表是否可发布 | 每个测试场景的完整矩阵、raw result hash | 未完成 |

每个阶段都应有“通过、失败、阻塞、允许进入下一阶段”的明确状态。尤其不能因为 P01–P05 通过就跳过 P06–P09，直接运行 P10；也不能用历史 archive 文件替代当前冻结的 train descriptor admission。

### 18.1 P08 的配对比较

P08 至少比较 `no_qd`、`planned_qd` 和 `realised_qd`。三者共享 candidate value backbone、场景、起点、seed、CF2X、传感器、通信、控制器、时间预算和统计脚本；同一 unit 使用相同的初始状态，唯一改变 QD 机制。报告内容应包括：AUC_time、最终覆盖、archive occupied/effective cells、公共 need strength、abstention、改选、计划-结果误差、通信和安全失败。

P08 的 `hm3d_p08_qd_matrix.py` 会检查训练 descriptor admission、跨场景支持、archive spec digest、来源 record hash、验证阶段不得调参以及方法矩阵完整性。它明确把“QD-only diagnostic”与完整 `realised-QD + RFG + RB-SF-SAC` 主方法分开，防止一部分组件的结果被包装成全链路结果。

### 18.2 P09/P10 的冻结逻辑

P09 至少需要三个独立 seed、冻结的 primary metric、alpha/power 设置和不访问测试场景的证据。P10 只接受与 P09 相同的 scene IDs、seed、fleet size、方法列表、预算、sensor entitlement、public observation contract 和 evaluator denominator。每一行结果还必须对应独立 raw result artifact，不能一个日志被复制到多个 cell。

## 十九、历史结果的正确读法

`docs/P10主表结果_2026-08-08.md` 中的结果应被标为历史/组件级证据。例如 00626 train 场景中 realised-QD 组件的 AUC 均值约 0.0984，no-QD 约 0.0909，planned-QD 约 0.0926；这只能说明在当时协议和场景下，realised-QD 组件存在对照差异，不能写成完整主方法已经取得稳定增益。00459 train 场景的排序不同，也说明 QD 贡献可能依赖场景难度和可产生的行为模式。

旧文档中出现的 40 秒、240 秒、安全零违规或 transit 完成等数字，必须和原始日期、scene、seed、controller、sensor profile、预算及“历史附件”标签一同引用。若其中任一字段不能绑定，最稳妥的写法是“历史运行记录，尚不足以进入当前正式主表”。

## 二十、如何从零复现一个合格实验

下面是一条面向研究者的实际执行顺序，而不是一个保证成功的命令清单：

1. 阅读 `docs/README.md` 和当前活动合同，确认没有把 archive 或 retired ObjectNav 计划当成入口。
2. 使用 `configs/external/hm3d_formal_preflight_protocol.json` 和对应 evidence 运行 preflight；先解决资产、几何、sensor、split 和时间合同问题。
3. 用 `scripts/assemble_hm3d_p03_p05_admission.py`、`scripts/admit_hm3d_aba_reset_evidence.py` 等脚本生成或审计 P01–P05 证据。
4. 使用 H15 脚本检查真实 sensor throughput；记录执行帧数、丢帧、source ID 连续性和窗口分布，不要只观察进程是否退出。
5. 运行 P07 exploration episode 或 outcome batch，保存 manifest、telemetry、raw receipt、communication outcomes、belief hash 和资源信息。
6. 对所有 outcome 做 schema、safety、replay 和 provenance 审计；失败项留在 ledger，按预先声明的规则决定是否进入训练。
7. 在 train split 上收集足够的真实 outcome，运行 descriptor richness、cross-scene support、footprint separation 和 reproducibility 审计，生成冻结的 train descriptor admission。
8. 只在 train admission 通过后运行 P08 配对对照；任何 selector/descriptor 变化都生成新 schema 或新 protocol hash。
9. P09 冻结统计和方法矩阵，清除测试场景在训练、调参和 descriptor 选择中的访问痕迹。
10. P10 逐场景运行 holdout，生成每行独立 raw artifact 和最终 summary；论文只引用经过 preflight 的行。

若本机没有 HM3D 资产、CF2X USD、IsaacLab、私有 evaluator 或正式 receipt，不能用单元测试的绿色状态代替上述步骤；可以运行合同和回放测试，但证据等级应标为静态/模拟。

## 二十一、测试体系与故障定位

测试目录按功能覆盖合同、archive、belief、通信、执行、轨迹、传感器、preflight、RL 和 IO。建议按以下层次解读：

- **单元测试**：验证一个函数的数值和异常行为，例如 archive cell 映射、Jaccard overlap、TTL、合法 mask 和指标积分。
- **性质测试/泄漏测试**：验证平移不变性、occupied 优先、幂等 replay、禁止真值字段和 schema hash 稳定性。
- **执行回放测试**：将已记录的 outcome 重新送入 ledger 和 archive，检查 revision、摘要和失败分类是否一致。
- **集成测试**：把候选、通信、执行和指标接起来，但若没有真实 Isaac 资产，仍不等价于物理实验。
- **性能测试**：检查向量化 step loop 的每物理秒成本、消息队列开销和缓存命中，不能改变策略输入和证据定义。

常见故障应按产生位置定位：

| 现象 | 首先检查 | 不应采取的修复 |
| --- | --- | --- |
| 候选池为空 | frontier、静态净空、队间距、通信 freshness | 放宽 clearance 或偷偷使用私有地图 |
| archive cell 很多但 footprint 重复 | richness、轴相关性、footprint separation | 增加维度或手动改 descriptor |
| 通信率高但 belief 不变 | payload digest、TTL、source ID、occupied 优先 | 把所有发送消息直接合并 |
| replay 结果不同 | manifest hash、随机状态、时间窗、浮点规范化 | 用最终均值覆盖差异 |
| RL 选择非法候选 | padding mask、transition action、候选数 | 在执行器里静默换动作 |
| 物理碰撞但路线审计通过 | root trace、tracking margin、settle、动态接触 | 只提高 arrival tolerance |
| P10 被拒绝 | P09 freeze、denominator、raw artifact identity | 删除拒绝行或重新命名结果 |

## 二十二、论文或学术申请中的准确表述

### 22.1 推荐的技术叙述

可以这样概括项目：

> 我实现了一个面向 HM3D-derived 三维室内环境的多智能体无目标探索研究框架。系统以受限稀疏测距构建公共体素 belief，通过三维 frontier 生成团队候选，在范围/视线通信和 CF2X 物理执行约束下运行；我重点实现了 outcome-grounded Quality-Diversity 机制，使行为 archive 只有在真实执行回执通过安全和身份审计后才更新，并用物理时间上的自由飞行探索体积 AUC 评估覆盖速度与最终覆盖。

这段话突出的是可验证的研究设计：观测受限、候选可审计、执行有回执、archive 不泄漏、评价考虑时间。它没有声称真实无人机部署，也没有把未完成 P08–P10 写成已发表结果。

### 22.2 不应使用的说法

- “在 HM3D 上完成了目标搜索并达到某个成功率”：当前任务没有目标真值，也不是 ObjectNav。
- “提出了完整可部署的多机飞控系统”：当前是 Isaac/PhysX 仿真和合同化执行器。
- “realised-QD 已经证明优于所有基线”：当前主要是组件级历史证据，完整主方法和 holdout 主表仍需收尾。
- “实现了真实无线通信”：代码实现的是可重放的范围/视线消息模型，不是 RF 传播实测。

### 22.3 项目名称的可读性

`hm3d-realised-qd` 适合作为仓库技术目录名，但对不熟悉项目的导师不够友好。正文第一次出现时应写成“HM3D-derived 三维多智能体探索与执行回执驱动的质量多样性研究”，随后再括号说明代码目录名。这样既保留 `realised-QD` 的方法辨识度，也让读者先理解任务、观测和贡献。

## 二十三、术语表与源码索引

| 术语 | 本项目中的含义 | 代码/文档入口 |
| --- | --- | --- |
| public observation | 策略被允许读取的 sparse range 结果 | `contracts/hm3d_public_schema.py`、`runtime/hm3d_belief.py` |
| belief | 由成功公共观测重建的稀疏 occupied/free/unknown 体素状态 | `runtime/hm3d_belief.py` |
| frontier | unknown 与已知 free 的三维边界簇 | `runtime/hm3d_frontiers.py` |
| candidate | 带上下文、路线、可行性和计划意图的团队动作提议 | `runtime/hm3d_candidates.py`、`contracts/exploration.py` |
| receipt/outcome | 真实执行后的轨迹、观测、通信、时间、安全和失败记录 | `contracts/exploration.py`、`contracts/models.py` |
| realised descriptor | 从执行后公共证据计算的行为描述符 | `runtime/hm3d_realised_qd.py` |
| planned descriptor | 执行前的意图向量，只能用于预测或 planned-QD 对照 | `runtime/hm3d_candidates.py`、`runtime/hm3d_realised_qd.py` |
| archive | 按行为 cell 保存 elite 的 QD 数据结构 | `archives/qd.py` |
| public exploration need | 从公共 belief 计算的当前探索缺口，不是 archive 轴 | `runtime/hm3d_realised_qd.py` |
| relay graph | 按范围和视线建立的公共通信图 | `runtime/communication.py` |
| AUC_time | 自由飞行探索体积随物理时间曲线的归一化面积 | `evaluation/hm3d_exploration_metrics.py` |
| P07 | 四机任务有效性和跨层执行链证据 | `evaluation/hm3d_p07_matrix.py`、相关脚本 |
| P08 | QD 机制配对比较和 richness 审计 | `evaluation/hm3d_p08_qd_matrix.py` |
| P09/P10 | 统计冻结和 holdout 正式主表 | `evaluation/hm3d_preflight.py` |
| RFG | 真实回执门控的片段复用设计，仍需正式收益证据 | `fragments/provenance.py`、主方法设计文档 |
| RB-SF-SAC | 带候选 mask、成本 critic、持续时间和 successor feature 的 SAC 内核 | `learning/rb_sf_sac.py` |

## 二十四、结语：这个项目真正训练的能力

这个项目的核心不是把若干流行名词叠加在一起，而是把一个容易被“知道地图、提前写 archive、忽略执行失败”污染的探索问题拆成可检查的接口。稀疏观测限制了策略能知道什么；belief 和 frontier 决定候选从哪里来；通信合同决定哪些信息真的共享；路径和控制器决定计划是否能落到物理轨迹；receipt 决定行为是否实际发生；QD archive 和 RL 只消费经过审计的数据；AUC_time、失败类别和公平矩阵决定论文结论能否被复核。

因此，当前最稳妥的研究结论是：代码已经形成了较完整的“公共观测—候选—安全执行—回执—QD/学习—评测”实验骨架，并且对信息泄漏、归档污染和协议漂移做了明确防护；但完整主方法的 P07–P10 证据、RFG 正式收益和 RB-SF-SAC 在线接入仍应在实验收尾后单独报告。申请材料中可以展示这种系统性研究能力，论文中则必须等待冻结协议和 holdout 结果后再写最终性能结论。
