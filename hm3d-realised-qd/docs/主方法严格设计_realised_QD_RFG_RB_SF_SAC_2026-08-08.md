# 主方法严格设计：realised-QD + RFG-enhanced RB-SF-SAC

日期：2026-08-08（v1，冻结前征求意见稿）
状态：设计规格（Design Spec）。实现差距见 §10。所有符号与现有 P07 协议
（hm3d-public-candidate-pool-v7、pass_through v5 transit、isaac-so3-feedback-v6、P03–P06 冻结）兼容。

---

## 1. 问题形式化

### 1.1 任务
四架 CF2X 四旋翼（N=4）在未知室内三维空间（HM3D 真实扫描）内，在固定物理预算
`T` 秒内最大化**累积确认自由飞行体积**。无目标真值、无地图先验、无深度相机；
每架无人机只有公共稀疏三维测距（P06 sparse_range_3d）。

### 1.2 决策级 SMDP
任务被建模为**决策级半马尔可夫决策过程（SMDP）**（每个高层决策持续一个物理片段）：

- 状态 `s_t`：公共信念图 `B_t`（FREE/OCCUPIED/UNKNOWN 体素）、四机位姿与能量、
  通信中继图、任务保留（task reservation）、决策时钟。
- 观测 `o_t`：公共信念 `B_t` 的**公共特征**（前沿簇、可飞连通分量、区域访问路线、
  稀疏测距回执）——所有方法共享同一观测权限（T0–T4 观测层级中最高的公共层）。
- 动作 `a_t`：从共享候选池 `C(B_t)` 中选择一个**团队候选**（每机一个 transit+observation
  片段，含 hold 选项）。动作空间大小 = |C|（有限）。
- 转移：执行器在 PhysX 中真实执行所选候选，产出**执行回执**
  `η_t = (realised trajectories, sparse-ray outcomes, safety ledger, energy, timing)`。
- 奖励 `r_t`：决策片段的 AUC 增量
  `r_t = AUC(B_{t+1}) - AUC(B_t)`，其中 AUC 为
  Explored-Free-Flight-Volume-AUC_time（对冻结分母 `V_reach` 归一）。
- 目标：`max E[Σ_{t=1}^{H} γ^{t-1} r_t]`，H 为预算内决策数上限。

### 1.3 核心困难（本文主张所针对的）
规划动作 `a_t^plan` 与真实执行 `a_t^real` 不一致：避障重写、通信等待、控制误差、
净空保护、多机分离会使候选**缩短、变形或失败**。若按计划把 `a_t^plan` 记入 QD 档案
或直接复用片段，学习系统会把"没有真正发生的收益"当成经验（虚假多样性与负迁移）。
本文主张：**以真实回执 `η_t` 驱动档案维护（QD）与片段复用（RFG），并由 RL 在
回执修正后的价值信号上学习选择**。

---

## 2. 系统架构（三层）

```
┌────────────────────────────────────────────────────────────┐
│ 层1 感知与候选生成（共享动作权威，所有方法同一）                  │
│   B_t → 前沿簇 → observation/route_progress/region_access     │
│   → 意图多样性增强的候选池 C(B_t)（上限 K，全部经 route guard  │
│     与 joint guard 准入）                                      │
├────────────────────────────────────────────────────────────┤
│ 层2 选择（方法差异只在这里）                                    │
│   QD 组件：archive A  + 需求对齐选择                            │
│   RL 组件：RB-SF-SAC 策略 π_θ(a|s)                             │
│   RFG 组件：片段复用评分 v(η) 融合进选择与训练信号               │
│   → 选中候选 a_t                                               │
├────────────────────────────────────────────────────────────┤
│ 层3 执行与回执（共享执行器）                                    │
│   PhysX/CF2X 执行 → η_t → belief 更新 → archive 更新 → 训练样本 │
└────────────────────────────────────────────────────────────┘
```

原则：
- 层1 与层3 对所有方法**完全共享**（公平协议）。
- 层2 中，基线只含一个组件；主方法按消融链逐级叠加（§9）。
- 意图多样性增强与 QD 意图审计在层1/层2 边界执行（候选池可用性检查，
  审计失败时按 §4.5 优雅回退，绝不修改安全合同）。

---

## 3. 共享动作权威（候选池 C(B_t)）

- 生成：公开稀疏测距 → 公共体素信念 → FUEL 式前沿簇 → 三类候选视角
  （observation / route_progress / region_access），每类经 `_public_free_space_path_result`
  在已知 FREE 连通分量内做 26-连通受限 BFS。
- 准入：静态净空守卫（0.90 m 规划包络 / 0.50 m 终端余量合同）、
  联合守卫（路线管间距 ≥ 0.50 m、端点间距 ≥ 0.95 m、中继连通）、
  时序窗口（pass_through v5 transit model）。
- 意图多样性增强（已实现，`build_public_candidate_pool`）：当可行候选的
  计划意图 cell 覆盖不足（任一轴 < 2 bins 或联合 < 6 cells），继续评估其余可行
  分配并做覆盖优先修剪；不改变安全与质量准入。
- 池上限 K（正式实验固定 K=16）。

---

## 4. QD 组件（realised-QD）

### 4.1 档案
档案 `A`：3 维描述符空间（4×4×4 = 64 cells）：
- `d1 = vertical_motion_ratio`：执行轨迹的垂直位移 / 总位移
- `d2 = team_spatial_dispersion`：四机端点归一化分散度
- `d3 = public_observation_complementarity`：公共观测互补性（重叠惩罚）

每个 cell 存 `Elite = (descriptor d, quality q, cost c, outcome_hash h_η, revision)`。
**关键约束：`d` 与 `q` 必须来自真实回执 `η_t`（realised descriptor/quality），
而非计划意图。** 计划的 intent `d^plan` 只用于预测对齐（§4.3），不直接入档。

### 4.2 入档规则
候选 `a_t` 执行后：
1. 计算 realised descriptor `d_t`、realised quality `q_t`（该片段的公共新自由体积增益）
   与 cost `c_t`（能量 + 时间）。
2. 若执行满足：无碰撞/越界/分离违规、片段完成、且 `d_t` 对应 cell 的
   现有 elite 被 `q_t` 支配（或 cell 为空）→ `add_or_update`。
3. 排除项：碰撞恢复、翻译复制轨迹、无新自由体素、stationarity 未通过
   （均记 `archive_exclusion_reasons`，不抹除执行证据）。

### 4.3 意图→描述符预测器
从 `A` 中学习 `P(d | d^plan)`：以计划意图 `d^plan` 为输入，用邻域加权估计
对应 realised 描述符的期望 `d̂` 与不确定性 `σ`（kernel：archive 中与 `d^plan`
最近的 cell 的 realised 分布）。**预测器不引入真值、不读评估器几何。**

### 4.4 选择规则（给定当前池与公共探索需求 `n(B_t)`）
```
base_utility(c) = quality_hint(c) / cost_hint(c)
utility_floor = max_u - slack * (max_u - min_u)          # slack = 0.1（冻结）
eligible = {c : base_utility(c) ≥ floor}
对 c ∈ eligible:
    d̂, σ = predictor(plan_intent(c))
    if σ > σ_max: abstain
    alignment(c) = n(B_t).alignment(d̂)                    # 与当前探索缺口对齐
score(c) = base_utility(c) + w_qd * alignment(c)         # w_qd 冻结
选择 argmax score；若 n(B_t) 不活跃（need < 0.15）→ abstain（回退公共价值）。
```
- 改选只允许在"预测的 realised 模式对当前真实缺口有实质优势"时发生（价值保护）。
- 记录每个决策的 selected / abstention / QD_INTENT_FALLBACK 证据。

### 4.5 意图审计与优雅回退（已实现）
每个决策检查 `C(B_t)` 的意图丰富度（≥6 可行候选、每轴 ≥2 bins、联合 ≥6 cells、
Shannon ≥4.0）。不满足时：
- 层1 意图多样性增强先补救（§3）；
- 仍不满足 → 本决策回退到公共价值选择器，记录 `QD_INTENT_FALLBACK_TO_PUBLIC_VALUE`，
  **不终止 episode、不改安全合同**。回退率是 QD 机制的诊断指标（论文如实报告）。

### 4.6 冷启动
archive 初始化：`--qd-history` 加载 ≥12 个、跨 ≥2 个 train 场景的真实回执记录
（同 P07 合同校验）。`MINIMUM_OUTCOME_ARCHIVE_ENTRIES_FOR_SELECTION=6` 前
选择器 abstain（等价 no_qd），达到后开始干预。

---

## 5. RL 组件（RB-SF-SAC）

### 5.1 为什么 RL 在此任务中需要"稀疏共享前沿 + 循环状态"
高层选择是**带时间相关性的组合决策**：一次选择影响后续可达性；观测（公共信念特征）
随历史演化。为此：**Recurrent Belief-State Shared-Frontier SAC（RB-SF-SAC）**：

- **Recurrent**：策略/价值以 LSTM 隐状态 `h_t` 编码历史（跨决策保持，片段内
  transit+observation 作为一个决策单元）。
- **Belief-state**：输入使用公共信念的特征（而非原始体素）——信念体素网格本身
  就是部分可观测下的 belief 表示。
- **Shared-Frontier**：所有无人机共享同一前沿特征编码（公共动作权威的自然延伸），
  动作是**团队候选索引**（每机子动作由候选 manifest 决定）。

### 5.2 MDP 化（与 §1.2 对齐）
- 状态特征 `φ(s_t)`：公共上下文特征（决策窗口、平均能量、通信度、机数）
  + 候选特征（planned descriptor 3 维、quality、cost，每候选）
  + LSTM 隐状态 `h_t`（历史压缩）。
- 动作：`a_t ∈ {1..K}`（候选索引）。掩码：不可行候选掩掉。
- 奖励：`r_t`（§1.2 的 AUC 增量，来自真实执行）。
- 成本（可选约束）：能量 `c_t`；cost-limit 使能时用拉格朗日乘子。

### 5.3 网络与训练目标
- 结构：`π_θ(a|φ,h)`（策略）、`Q_ω`（双 critic，SF 共享编码）、LSTM 编码器。
- 目标：SAC 最大熵（自动温度 α，target_entropy_ratio=0.7 冻结）；
  可选 cost critics（enable_cost_critics）。
- 训练数据：**离线优先**——先用已采集的真实回执 transition（每决策一条，
  见 §7.2）做 replay 更新；随后在线收集新回执补充（每决策一个样本，
  多次梯度复用同一真实交互，禁止制造未执行的 counterfactual）。
- 隐状态：episode 内跨决策传递；episode 结束或机器人失效时清零。

### 5.4 与 QD 的关系（关键接口）
- **QD 先于 RL 干预**：RL 策略的训练奖励直接是回执 AUC 增量；
  QD 提供的是**行为多样性约束/偏好**，不是额外奖励注入。
- 接口形式（冻结前二选一，须在实验报告中声明）：
  (A) QD 偏好作为策略输入特征（preference_dim>0，预测器输出 d̂ 向量作为上下文）；
  (B) QD 作为动作遮罩（只允许 archive 邻近 cell 的候选被选，价值保护不变）。
  默认选 (A)：侵入最小、消融干净（preference_dim=0 即退化为普通 RB-SAC）。

### 5.5 冷启动与训练-评测分离
- 训练 checkpoint 必须来自 train 场景回执（P05 划分冻结）；validation/test 场景
  只能用于评测，绝不进 replay。
- 训练预算按"环境交互数"而非梯度数对基线对齐（§7.3）。

---

## 6. RFG 组件（真实回执门控片段复用，Outcome-Gated Fragment Reuse）

### 6.1 可复用片段定义
片段 `f`（transit 或 observation）**可复用**当且仅当（全部满足）：
1. 该片段在回执 `η_t` 中**完整执行**（transit_completed / observation_completed，
   无碰撞、无越界、无分离违规、无超时）；
2. provenance 允许（`outcome_to_replay` 通过：时间合同、执行身份哈希、清单匹配）；
3. 其执行轨迹已被公共 belief 吸收（片段终点的观测已进入 `B_{t+1}`）。

### 6.2 门控复用收益
对可复用片段 `f`，记录 `(plan(f), η(f), gain(f))`：
- `gain(f)` = 该片段的公共新自由体积贡献（按无人机归属拆分）。
- 复用发生时：把 `f` 的实测 gain 作为该区域**后续候选的收益下界/先验**
  （RFG credit），仅在候选与 `f` 共享已验证轨迹前缀时授予。
- **负迁移防护**：片段一旦在后续执行中被守卫重写、超时或失败，
  立即撤销其 RFG credit（outcome-gated：信用以新回执为准）。

### 6.3 与 QD/RL 的整合
- QD：RFG credit 计入 realised quality `q_t` 的一部分（仅当该片段真的被复用且真的完成）。
- RL：RFG credit 作为候选特征之一（候选特征维 +1），使策略学会"优先选择
  已被回执验证的片段"；消融为 credit 特征置零（= 无 RFG 的 realised-QD+RB-SAC）。

### 6.4 统计口径
论文报告：fragment_acceptance_rate、fragment_negative_transfer_rate、
gain_per_accepted_fragment、reuse 对 AUC 的增量（配对）。

---

## 7. 数据、训练与评测协议

### 7.1 场景与划分（P05 冻结，不可改）
- train：00626、00459（已资格化）+ 后续按需扩充（00770 等，需完整 P03–P06 资格链）
- validation：00803（已资格化）
- test：P05 冻结的 test 集（00810+，评估前不得触碰）
- 正式主表场景数：≥3 train（至少 00626/00459 + 1 新增）；若无法扩充，
  主表只报告 2 场景并声明局限。

### 7.2 训练数据规模（最低门槛）
- QD archive 初始化：≥12 回执 × ≥2 场景（已达 104 文件）。
- RB-SF-SAC 训练：**≥1000 决策级 transition**（当前 301+60，不达标）。
  采集 300 决策/场景 × 4 场景 或 500×2 场景。每 transition 为
  (φ(s_t), pool, a_t, r_t, φ(s_{t+1}), pool', h_t)。
- 正式门槛：训练后必须在开发集上 AUC 高于 random + 2σ 才进入主表（否则如实报告失败）。

### 7.3 公平训练预算
- 所有学习型方法（single_rl、marl_ipp_port、主方法）共享**同一真实交互预算**
  （同一批回执，可不同梯度数——梯度复用不产生新交互）。
- 主方法不得比基线多拿观测权限、种子或调参轮数。

### 7.4 统计检验（主表强制）
- 每格 ≥5 seeds（消融 ≥3）；配对（同 seed 同起点同池）t 检验 / Wilcoxon；
- 主指标 AUC_time 全场景聚合报告 mean±std 与效应量；
- 若 5 seeds 后主方法 vs 最强基线差异 p>0.05，如实报告"方向一致但未达显著"，
  不宣称胜出。

### 7.5 执行预算
- 正式回合 40 s（与 P0 冻结一致）；240 s 仅作机制诊断，不进主表均值。

---

## 8. 基线（同一共享池、同一合同）

| id | 机制 | 角色 |
|---|---|---|
| random | 随机选池内候选 | 下界，验证任务非饱和 |
| frontier_3d | 信息增益最大化 | 几何控制 |
| auction | 增益/距离/风险分配 | 经典协调 |
| gvp_mrep_port | 动态拓扑图 Voronoi 分区（受控迁移） | 强规划基线（声明非原 ROS 复现） |
| single_rl | 朴素 SAC（无 QD、无 RFG） | 无 QD 学习控制 |
| marl_ipp_port | 作者 PPO+LSTM 受控迁移 | 强 RL 原型（资格后入主图） |

---

## 9. 消融链（隔离变量，配对于同 seed）

```
base = single_rl（朴素 SAC）                     → 对照
+ realised-QD（QD 组件，§4）                     → 验证"真实回执档案"
+ RB-SF-SAC 结构（recurrent + shared-frontier）   → 验证"结构"
+ RFG（§6）                                     → 验证"片段复用"
= 主方法 realised-QD + RFG-enhanced RB-SF-SAC
```
另配对：planned_qd vs realised_qd（计划意图 vs 真实回执）；
no_qd vs realised_qd（有无档案）。每对同 seed 同起点，报告改选率、abstention 率、
回退率与 AUC 增量。

---

## 10. 实现状态与差距清单（诚实）

| 组件 | 现状 | 差距 |
|---|---|---|
| 层1 候选池 + 意图多样性增强 | ✅ 已实现并验证 | 无 |
| QD 档案（真实回执） | ✅ 已实现并验证 | 无 |
| QD 选择器 + 意图审计 + 回退 | ✅ 已实现并验证 | 无 |
| 意图→描述符预测器 | ✅ 存在（`_prediction`） | 需校准报告（不确定性校准） |
| RB-SF-SAC 网络/训练器 | ⚠️ `rb_sf_sac.py` 存在，`sf_dim=0` 未启用 SF；single_rl 用的朴素 SAC | 启用 SF/LSTM 结构；写训练入口 |
| RB-SF-SAC 接入在线选择 | ❌ 未实现（`_select` 无该策略） | 新增 strategy + checkpoint 加载 |
| QD×RL 接口（preference 特征） | ❌ 未实现 | 按 §5.4(A) 实现 |
| RFG 片段复用 | ⚠️ provenance/`reusable_fragment_count` 存在 | 复用收益计算、credit 入 QD/RL、负迁移撤销 |
| 训练数据 ≥1000 transitions | ❌ 当前 301（00626）+60（00459） | 补采 |
| 主表 ≥5 seeds + 统计检验 | ❌ 当前 3/2 seeds | 补跑 + 检验 |
| 第 3 个 train 场景资格 | ❌ 未开始（00770 候选） | 补 P03–P06 + 审计 |

---

## 11. 可验证命题（Pre-registered Hypotheses）

H1（回执价值）：realised-QD ≥ planned-QD ≥ no_qd（同池同 seed，00626 主指标）。
H2（RL 增益）：主方法全链 > realised-QD 组件（AUC 增量，配对检验）。
H3（复用安全）：RFG 的 fragment_negative_transfer_rate ≈ 0 且增益为正。
H4（难度相关）：QD/RL 增益在大场景（00626 类）大于小场景（00459 类）。
H5（公平）：主方法在相同交互预算下不依赖额外观测/种子。

任一假设被否定时：如实报告，不替换指标、不放松合同、不删除失败数据。
