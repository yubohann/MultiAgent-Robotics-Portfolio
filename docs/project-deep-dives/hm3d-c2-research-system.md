# HM3D 语义搜索与 C2 几何探索研究系统：实现、复现与实验设计

## 0. 文档定位与阅读方法

本文介绍一套正在收敛中的多机器人研究系统。它不是把几个仓库拼成一个“已经完成”的产品，而是把三个相互关联、实验身份不同的研究层放在同一份技术说明中：

1. **HM3D-Habitat 主线**：双机器人 RGB-D ObjectNav，研究自然语言目标驱动的语义搜索。
2. **C2 几何探索主线**：多机器人在未知空间中的 target-free exploration，研究连通性建模、任务分配、路径规划和执行反馈。
3. **HM3D-C2 transfer**：在两条主线的环境、观测边界和统计协议冻结后，验证区域级候选、可达性、分配和执行接口能否迁移。

本文依据以下本地资料整理：

- `E:\HM3D_2026_papers_and_repos` 资料库及其 `reproduction_tracking` 追踪记录；
- `E:\HM3D_2026_papers_and_repos\official_repositories\Co-NavGPT2_native_one_ep5_20260823` 运行副本；
- `C:\Users\Administrator\ECRTA_C2` 项目根目录及 `docs/` 中的复现、协议和审计文档；
- WSL 工作区 `/home/c2dev/c2_explorer_reproduction/workspace/` 中的 `falcon_official` 与 `racer_official`；
- 现有作品集深度文档中关于实验合同、证据分层和项目写法的约定。

源码、配置、脚本和 Markdown 记录按文件检查；数据集压缩包、模型权重、图片、视频、JSON 压缩结果和长日志不适合逐字符阅读，因此以文件名、大小、哈希、schema、manifest、首尾状态和聚合字段进行审计。二进制“存在”不被当作“实验已通过”，历史附件也不自动等同于当前实现。

文中使用四种证据标签：

| 标签 | 含义 | 可以支持的结论 |
| --- | --- | --- |
| `S` | 源码、配置、schema 或静态合同 | 接口或算法分支确实存在 |
| `T` | 测试、编译、schema 或绑定检查 | 某个工程约束在给定环境下通过 |
| `D` | development / smoke / bounded 运行 | 说明运行链可以执行，不能代表论文总体 |
| `F` | 论文规模、身份冻结、统计和失败账本均满足 | 才允许写入论文主表 |

当前 HM3D 和 C2 都仍有 `D` 级证据，不能把开发运行写成 `F` 级结论。

## 1. 三百字项目概括

本项目构建了一条从三维室内语义导航到多机器人几何探索的研究复现链。HM3D 主线在 Habitat 中运行两个机器人，以 RGB-D、GPS、compass 和文本任务为输入，通过 YOLO-World 与 Mobile-SAM 获取目标观测，维护共享地图，提取 frontier，再由视觉语言模型或可复现的弱基线分配探索区域，最后由 FMM 和离散动作执行 ObjectNav。C2 主线则不读取目标真值，围绕未知空间的体素地图、自由/未知/门洞连通图、frontier、ACVRP/LKH 分配、A* 或 kinodynamic A*、B-spline 轨迹和碰撞重规划组织多机器人探索。两条线共享“候选—可达性—分配—局部执行—回执”的接口思想，但任务定义和指标保持分离：HM3D 使用 SR、SPL、DTG，C2 使用 completion、探索时间、路径长度、coverage 和速度。研究的难点不是让某个脚本返回 JSON，而是固定数据、传感器、模型身份、场景、seed、统计端点和失败账本，证明每次比较只改变预定变量。未来的 HM3D-C2 transfer 只验证区域级几何接口是否可迁移，不把 C2 的完成率冒充为 HM3D 的语义成功率，也不把未收尾的实验包装成论文结论。

## 2. 研究问题与系统边界

### 2.1 HM3D 主问题：语义目标搜索

给定一个室内场景、两个机器人、一个文本目标类别和有限步数，机器人只能使用允许的观测和自身历史，寻找目标并在 evaluator 合同下调用 STOP。策略不能读取：

- `objectgoal` 或目标类别的内部枚举值；
- 目标坐标、goal annotation、oracle map 或 evaluator 状态；
- 由 Habitat 评测器计算出的距离和成功标志。

这使问题更接近部分可观测决策过程。目标语义来自文本任务，空间位置要由视觉、深度、地图和行动逐步推断。策略如果只是“看到了一个类别标签”却不能到达 evaluator 所定义的目标位置，仍不算成功。

### 2.2 C2 主问题：几何空间探索

C2 不知道目标是什么，也不需要 ObjectNav 目标。机器人从稀疏深度或激光观测构建局部/全局几何 belief，发现 frontier，判断未知区域之间的连通关系，分配任务并执行轨迹。完成通常由探索覆盖、未知体素减少、任务单元耗尽或时间边界定义，不能与 HM3D 的语义成功混为一谈。

### 2.3 Transfer 的最小问题

迁移层不重新定义一个“大一统指标”，而是检查以下接口是否可以从一个环境复用到另一个环境：

```text
局部观测 -> 候选区域 -> 可达性检查 -> 多机器人分配
        -> 局部规划 -> 动作执行 -> 成功/失败回执
```

迁移实验的输入只能是允许的区域级几何和执行状态；不能把 C2 的隐藏地图、HM3D 的目标 annotation 或任何 evaluator-private 字段带入另一条主线。

### 2.4 明确的非目标

- 不是对论文结果的无条件复刻；模型替换、数据子集和协议重实现必须显式标注。
- 不是把 FALCON、RACER、C2、Co-NavGPT2 的所有代码合并成一个可执行包。
- 不是用一段视频、一个 smoke 或一个成功 episode 代替统计实验。
- 不是把所有历史脚本都视为当前推荐入口。
- 不是在实验尚未收尾时提前宣布“完整系统已经验证”。

## 3. 资料库与工作区地图

### 3.1 HM3D 资料库根目录

`E:\HM3D_2026_papers_and_repos` 是数据、论文、官方仓库、运行时依赖和复现记录的总目录。它包含 `datasets/`、`official_repositories/`、`papers_pdf/`、`runtime_dependencies/`、`model_weights/`、`reproduction_tracking/`、`logs/` 和多组按日期命名的实验输出。

这些目录职责不同：

| 目录 | 用途 | 不能据此推出的结论 |
| --- | --- | --- |
| `datasets/` | HM3D v0.2 tar、staging、runtime overlay、episode 提取 | 数据可加载不等于策略得分通过 |
| `official_repositories/` | Co-NavGPT2、L3MVN、Habitat 及其他来源 | 有源码不等于已使用该源码运行 |
| `reproduction_tracking/` | 计划、身份矩阵、失败审计、manifest、聚合结果 | 计划文件不等于实验完成 |
| `logs/` | stdout、stderr、逐步 telemetry 和旧运行日志 | 旧日志必须绑定 source/config/hash 后才能比较 |
| `model_weights/` | YOLO-World、Mobile-SAM、RedNet 等权重 | 权重可读取不等于模型语义正确 |
| `papers_pdf/` | 论文原文和文本抽取 | 摘要不能替代方法和实验细节 |
| `hm3d_*` 实验目录 | 不同 runner、策略、步数和审计阶段的输出 | 目录名中的 `formal` 或 `final` 不是资格证明 |

关键追踪文档包括：

- `HM3D_current_status_and_execution_plan_20260823.md`：当前身份、资产、运行进度和阻塞项；
- `HM3D_弱基线正式执行计划_20260823.md`：Greedy、Cost-Utility、Random、Multi-SemExp 的共同合同；
- `HM3D_弱基线源代码审计_20260823.md`：弱基线与论文文字、代码入口的对应关系；
- `HM3D_C2_cross_task_fusion_and_reviewer_defense_20260822.md`：两条主线的边界和审稿风险；
- `C2_HM3D_two_dialogue_coordination_20260822.md`：C2 对话与 HM3D 对话之间的事实同步板；
- `HM3D_official3d_target_chain_audit_20260822.md`：目标观测、深度、地图和停止链的审计；
- `HM3D_protocol_mismatch_audit_20260822.md`：官方作者入口与共同协议 runner 的差异；
- `official_baseline_inventory_20260821.md`：官方基线来源和可复用性清单。

### 3.2 Co-NavGPT2 运行副本

当前副本位于：

`E:\HM3D_2026_papers_and_repos\official_repositories\Co-NavGPT2_native_one_ep5_20260823`

顶层文件的职责如下：

| 路径 | 职责 |
| --- | --- |
| `main.py` | 官方单环境入口，保留作者的 Habitat reset、目标任务和 evaluator 路径 |
| `main_vec.py` | 多进程或批量入口 |
| `arguments.py`、`constants.py` | 命令行参数、动作和实验常量 |
| `agents/vlm_agents.py` | 单机器人视觉语言导航 agent |
| `agents/vlm_multi_agents.py` | 双机器人共享地图、候选和分配逻辑 |
| `agents/ros2_agents.py`、`ros2_single_agent.py` | 现实 ROS 机器人接口和单机/多机执行适配 |
| `utils/mapping.py` | RGB-D 点云投影、障碍物和已探索地图更新 |
| `utils/explored_map_utils.py` | 已探索区域和 frontier 相关处理 |
| `utils/fmm_planner.py` | Fast Marching Method 距离场和局部路径规划 |
| `utils/shortest_path_follower.py` | 将路径点翻译为 Habitat 离散动作 |
| `utils/detection_segmentation.py` | YOLO-World 检测和 Mobile-SAM 分割 |
| `utils/chat_utils.py` | VLM 请求、响应解析、token/latency 审计和 fallback 记录 |
| `system_prompt.py` | 给分配模型的候选 frontier 与机器人状态提示模板 |
| `configs/*.yaml` | HM3D ObjectNav、双机器人、传感器和预算配置 |
| `multi-robot-setting/habitat/` | 对 Habitat 的多机器人任务、simulator、动作和 evaluator 改造 |
| `data/`、`weights/`、`img/` | 场景数据、模型权重和可视化资源；真实私有权重不得公开 |

### 3.3 C2 根目录和 WSL 工作区

Windows 研究根目录：`C:\Users\Administrator\ECRTA_C2`

其中 `docs/` 保存论文—代码对照、B0/B1/B2 计划、实验规则、问题清单和交接提示词；`source_latest/`、`scripts/`、`tests/` 和 `debug/` 分别承担代码、运行脚本、回归测试和诊断输出。

Linux/ROS 实验工作区：`/home/c2dev/c2_explorer_reproduction/workspace/`

| 子目录 | 来源/作用 | 研究角色 |
| --- | --- | --- |
| `falcon_official/` | FALCON 官方仓库，HEAD `312eb4d32c6c7af1a482f94a0a204aa2bb150cca` | 覆盖路径引导的快速探索和评测环境参考 |
| `racer_official/` | RACER 官方仓库，HEAD `049c332e3634ef72d8beb155b4c13dc91ca52916` | 去中心化多机器人探索、通信和负载均衡参考 |
| B0/B1/B2 工作副本 | C2 官方发布代码和最小修复候选 | 论文级复现和消融实验 |
| `logs/`、`receipts/`、`manifests/` | 运行状态和统计输入 | 证据链，不是算法源码 |

## 4. HM3D 数据、场景和实验合同

### 4.1 数据规模与绑定

正式 HM3D v0.2 val 版本包含 36 个 scene、1000 个 ObjectNav episode 和 6 类目标。当前归档中类别数量为：chair 195、sofa 187、toilet 166、bed 165、plant 152、tv_monitor 135。`val_mini` 只有 30 个 episode、2 个 scene，只能用于 smoke 或开发诊断。

正式归档和运行材料包括：

```text
formal tar:
E:/HM3D_2026_papers_and_repos/datasets/hm3d_val_formal_download_20260823/
staging:
E:/HM3D_2026_papers_and_repos/datasets/hm3d_val_formal_staging_20260823
runtime overlay:
E:/HM3D_2026_papers_and_repos/datasets/hm3d_val_formal_runtime_overlay_20260823
episode 5 smoke:
scene 00877-4ok3usBNeis, 5 steps, smoke_pass
```

每个正式运行应至少绑定：`source_file`、`source_episode_index`、`episode_id`、`scene_id`、category、geodesic 元数据、manifest hash、代码 revision、运行时和输出 stem。仅凭 `episode_id` 不足以保证正式 val 的唯一性，因此当前 runner 增加了 `scene_id` 约束。

### 4.2 传感器和步数

共同协议使用两个机器人、RGB、depth、GPS、compass 和文本任务。作者配置的 paper budget 为 500 steps；成功距离的协议约束为原始 STOP/evaluator 合同，不能因为开发集表现不佳而放宽阈值。策略只能消费允许的观测投影，runner 还要记录被传入模型的字段，防止隐藏目标通过包装器泄漏。

### 4.3 正式表身份

HM3D 结果至少分为：

1. `HM3D-Habitat-main`：共同协议下的双机器人语义搜索；
2. `Habitat-native-reference`：保留作者入口、作者 reset 和 evaluator 的参考运行；
3. `HM3D-C2-transfer`：独立的区域级几何迁移诊断。

`completion`、探索时间、coverage 和路径长度属于 C2-native；`SR`、`SPL`、`DTG` 属于 HM3D-main。它们可以在接口审计中并列说明，但不能放进一张表相加或排序。

## 5. Co-NavGPT2 的实际实现链

### 5.1 环境初始化和双机器人状态

`main.py`/runner 首先创建 Habitat 环境，读取 scene、episode 和 action space。双机器人改造位于 `multi-robot-setting/habitat/`，负责将两个 agent 的传感器、位置和任务状态组织到同一 episode 中。历史 harness 曾误把 agent-0 的 reset state 复用于两个机器人，现已改为每个 agent 读取自己的 runtime state；旧结果保留为混淆证据，不再进入共同协议统计。

### 5.2 RGB-D 到三维点云

对像素 $(u,v)$、深度 $d$ 和相机内参 $(f_x,f_y,c_x,c_y)$，点云投影为：

$$
x=(u-c_x)d/f_x,\quad y=(v-c_y)d/f_y,\quad z=d.
$$

点云再根据 GPS/compass 或姿态变换到局部/共享地图坐标系。代码必须明确深度单位、相机坐标轴、裁剪范围和无效深度处理；否则同一个目标会在地图中产生系统性偏移。`detection_segmentation.py` 的分割掩码和深度点数共同决定目标点云是否可靠。

### 5.3 YOLO-World 与 Mobile-SAM

YOLO-World 提供开放词汇检测框，Mobile-SAM 将候选框细化为像素掩码。点云由掩码内有效深度点反投影得到，随后写入 persistent object point cloud。官方实现的关键语义是：

- 超过检测阈值并且存在足够有效深度点时，目标观测才可能累积；
- `found_goal` 与 object cloud 是否为空相关，而不等同于 evaluator 的 success；
- object cloud 被选为局部/最终目标后才进入更深的处理；
- 某些逐帧处理在官方代码中被注释，不能擅自解释为“每一帧都完成完整融合”。

因此，plant、toilet 等失败局要同时检查检测框、掩码、有效深度点、持久化点云、FMM 目标和 STOP 时刻，不能只看最后的 SR。

### 5.4 地图与 frontier

`utils/mapping.py` 将 RGB-D 投影到 top-view，维护障碍物、自由空间和已探索区域。`Frontier_Det` 使用形态学操作与连通区域分析，把“已探索自由空间”和“未知空间”的边界提取为 frontier centroid。

对二维栅格地图 $M$，可以用四邻域或八邻域连通区域标记 frontier 集合。每个候选至少需要记录坐标、来源机器人、观察时间、是否已分配、是否仍可达和与已走区域的关系。重复候选、地图坐标漂移和不同机器人局部地图没有对齐，会直接改变分配结果。

### 5.5 VLM 分配与弱基线

VLM 接收经过清洗的候选 frontier 图像、机器人状态和历史摘要，输出每个机器人对应的候选或 `None`。`chat_utils.py` 负责请求、JSON schema、token、latency、fallback 和审计。当前 DeepSeek 是模型替换，不能写成 GPT-4o 原生复现。

为了建立可解释的共同基线，runner 支持：

- **Greedy**：选择最近或当前代价最小的未分配 frontier；
- **Cost-Utility**：在候选收益与到达代价之间做分配；
- **Random Sampling**：从 observed-free map 中随机抽取长期目标；
- **Multi-SemExp**：官方多机语义探索方向，但当前身份仍为组件级或依赖阻塞。

三臂共同冻结 detector、地图、FMM、离散动作、STOP 和 evaluator，只替换长期 assignment policy。否则比较出来的差异可能来自传感器或局部控制器，而不是分配策略。

### 5.6 FMM 与动作执行

FMM 在可通行栅格上求解从当前位姿到目标点的距离场。最短路径 follower 将路径方向与机器人 compass 比较，输出：

```text
STOP / MOVE_FORWARD / TURN_LEFT / TURN_RIGHT / LOOK_UP / LOOK_DOWN
```

局部执行不是“规划完一次就结束”。每个时间步需要重新读取传感器、更新地图、检查候选是否仍有效、判断是否应该 replan，并把动作与 before/after 状态写入 trace。距离最短但目标点处于障碍、视线被遮挡或超过 STOP 合同，仍然会失败。

### 5.7 官方入口与共同协议 runner 的差异

同一 episode 5、同一正式场景已经有两类参考：

- 作者入口：53 steps、success=1、SPL=0.760、DTG=0.031 m；
- objectgoal-stripped 共同协议 runner：46 steps、success=1、SPL=1、DTG=0.051 m、path=9.430 m。

它们使用相同场景和 500-step budget，但身份不同：作者入口保留 ObjectGoalSensor 和作者 reset，是 `Habitat-native-reference`；共同 runner 把策略输入限制为 RGB/depth/GPS/compass，是 `HM3D-Habitat-main`。两条结果用于协议差异审计，不应直接作为 superiority 结论。

## 6. C2 技术链：从 FALCON/RACER 到 B0/B1/B2

### 6.1 FALCON 的可复用思想

FALCON 是覆盖路径引导的快速空中探索系统。源码包含 exploration manager、voxel mapping、trajectory、pathfinding、simulator、ROS launch 和可视化。它把在线生成的 coverage path 作为探索规划器的先验，再交给局部轨迹和碰撞约束执行。

重要思想不是直接复制 ROS Noetic 工程，而是：

1. 先用体素或点云表示未知/自由/占据空间；
2. 从 coverage path 产生有顺序的候选视点；
3. 用可执行的轨迹优化替代只在栅格上“连两点”；
4. 以物理执行时间和实际覆盖评估规划器。

### 6.2 RACER 的可复用思想

RACER 强调去中心化、多机器人、异步和受限通信。其源码包含 `active_perception`、`hgrid`、连接图、CCL、LKH、B-spline、A*、kinodynamic A*、collision 和 replan 等模块。

RACER 对 C2 的价值在于把“多机协作”拆成可验证的机制：

- 每架机器人维护局部状态并交换必要摘要；
- 通过连通图避免把不可达未知区域当成可分配任务；
- 使用 LKH 或等价求解器分摊候选路径；
- 用 B-spline 或动力学约束平滑路径；
- 运行时检测碰撞、通信和路径失效并触发重规划。

### 6.3 C2 官方发布代码

C2 官方 release 的核心组件包括：

- CCL 和 unknown/free/portal connectivity graph；
- unknown disconnected filtering 与 adjacency penalty；
- ACVRP/LKH 和任务分配；
- Proposal/Commit/Finalize/Cancel 任务状态机；
- ATSP/CP 路径求解；
- B-spline 轨迹生成；
- collision replan 和执行回执。

论文文字与发布代码仍有需要记录的差异，例如论文描述的 graph-based A* 与代码中的 BFS/版本分支、CVRP demand 解释、参数和 coverage/path 统计口径。复现时必须把代码真实行为写入身份记录，不能用论文段落替代 source read-back。

### 6.4 B0、B1、B2 的实验分层

| 臂 | 内容 | 允许的改变 | 不能声称 |
| --- | --- | --- | --- |
| B0-original | 官方发布代码、固定地图、固定渲染和控制参数 | 只读运行与统计审计 | 不能把一局成功写成论文复现通过 |
| B1-allfix | 证据支持的最小执行/FBL 修复 | 只能修改已定位的故障接口 | 不能堆叠未经验证的机制 |
| B2-core | observed spatial prior 的 disabled/neutral/shuffled/live 四臂 | 只增加空间先验变量 | 不能在 B1 未冻结前启动 |

C2 论文锚点使用 Cubicle、Open-plan、Octa 等场景及 4-UAV 运行条件。当前 B0 的 Cubicle/Open-plan 多局完成，Octa 存在未完成局；这说明运行链能跑，但不等于已经达到论文时间/路径锚点。每个结果必须绑定源码、地图、参数、渲染、seed、统计端点和 raw log。

## 7. 理论基础

### 7.1 POMDP 视角

可以把 HM3D ObjectNav 写成 $(S,A,O,T,R)$ 的部分可观测决策过程。场景状态 $s_t$ 包含真实几何、目标位置和机器人位姿；策略只能得到观测 $o_t$，通过历史 $\tau_t=(o_0,a_0,\ldots,o_t)$ 形成 belief $b_t(s)$。C2 则主要维护几何 belief 和 frontier，而不是语义目标 belief。

两条线共同面对的工程事实是：策略不能直接看到真实状态，候选、地图和执行回执都是不完全信息下的中间变量。

### 7.2 ObjectNav 指标

- **SR**：成功 episode 数除以 episode 总数。
- **SPL**：成功与路径效率的结合，通常将最短可行路径与实际路径比较。
- **DTG**：终止时到目标的距离。

在多机器人设置中，要明确 episode 级 success 是“任一机器人成功”还是“所有机器人完成”，以及 SPL 的路径是否合并、取最小或按团队定义。不能只沿用单机器人公式而不检查作者代码。

### 7.3 Frontier 与探索

frontier 是已知自由空间与未知空间的边界。它不是目标本身，而是“继续获得信息”的候选位置。一个好的 frontier 分配要同时考虑：

$$
U(g)=\text{expected information gain}(g)-\lambda\,\text{travel cost}(g)-\mu\,\text{risk}(g).
$$

当前弱基线只允许使用与论文定义一致的简化形式；EIG、风险和均衡机制不能在没有独立实验时私自加入。

### 7.4 连通图、CCL 与任务分配

栅格或体素中先用 CCL 标出自由/未知区域，再用 portal、门洞或相邻边构造连通图。候选区域若与当前已知自由区不连通，应该进入不可达或延迟队列，而不是直接送入 LKH。任务分配可以形式化为多车辆路径问题：

$$
\min \sum_i c_i(\pi_i),\quad
\text{s.t.}\;\text{each candidate is assigned at most once,}\;
\text{capacity, connectivity and return constraints hold}.
$$

LKH/ATSP 能够快速给出顺序，但不能代替碰撞、动力学和通信检查。

### 7.5 B-spline 与执行反馈

B-spline 用控制点 $P_i$ 和基函数 $N_{i,k}(t)$ 表示连续轨迹：

$$
p(t)=\sum_i N_{i,k}(t)P_i.
$$

轨迹生成阶段要满足速度、加速度、碰撞和地图边界约束。执行阶段若局部地图变化、通信过期或碰撞预测失败，必须取消旧计划并生成新的 candidate/trajectory。C2 的 Proposal/Commit/Finalize/Cancel 就是把“计划存在”和“执行成功”区分开。

### 7.6 质量多样性与真实回执

如果后续将 `hm3d-realised-qd` 的思想用于 transfer，archive 的更新必须等待真实执行回执。候选计划本身不能当作行为质量；只有完成的轨迹、覆盖、时间、失败原因和安全状态才可以更新质量档案。这一点与 C2 的执行反馈天然相关，但不能因此把 QD 指标加入 HM3D ObjectNav 主表。

## 8. 复现与实现步骤

### 8.1 第一步：锁定身份和来源

运行前先记录：

```text
paper / benchmark identity
source repository and commit
runner and config hash
dataset tar/staging/runtime overlay hash
model identity and credential location
scene, episode, category and source index
step budget and evaluator contract
```

任何字段缺失都先标为 `identity_incomplete`，不解释性能。

### 8.2 第二步：验证数据和场景

先对 tar、staging、scene GLB、navmesh、episode JSON 和 manifest 做绑定检查，再执行单局 5-step scene-load smoke。smoke 需要确认：场景加载、navmesh、agent reset、传感器字段、动作空间和结果 schema。它只证明环境能启动。

### 8.3 第三步：运行共同协议单局

建议使用运行副本中已有的脚本：

```powershell
Set-Location E:\HM3D_2026_papers_and_repos\official_repositories\Co-NavGPT2_native_one_ep5_20260823
python ..\..\reproduction_tracking\run_conavgpt2_policy_smoke.py `
  --episode-id <episode> --scene-id <scene> `
  --assignment-policy greedy --max-steps 5
```

实际参数应以脚本 `--help` 和对应 manifest 为准；示例不是对缺失路径或模型凭据的替代。成功后再扩展到同一局 30/100/500 steps，保留 stdout、stderr、raw JSON、逐步 trace、LLM audit 和 failure ledger。

### 8.4 第四步：匹配弱基线

固定场景、episode、传感器、地图、FMM、动作、STOP 和 evaluator，只改变 `assignment_policy`。每个方法至少经历：

```text
5-step contract smoke -> 3-episode bounded -> 12-episode stratified development
-> protocol audit -> 论文规模正式运行
```

不确定配置不能直接长跑。失败行、缺失行和超时行必须保留，聚合脚本只能读取当前 manifest 绑定的结果。

### 8.5 第五步：C2 单独运行

C2 使用 WSL、Xvfb 和固定 OpenGL/GLSL 运行时。启动前核验源码、renderer、地图、参数和 launch；B0 只读，B1 先逐事件定位故障，B2 在 B1 冻结后才进入四臂空间先验消融。每个 run 保存 raw log、telemetry、manifest、source/binary hash、seed 和失败局。

### 8.6 第六步：建立 transfer adapter

迁移 adapter 只暴露：

```text
candidate_region_id
candidate_pose_or_frontier
reachable / unreachable / unknown
assigned_robot
local_plan
execution_receipt
failure_reason
```

它不暴露 HM3D goal annotation、C2 完整占据图、evaluator 真值或另一条线的隐藏统计。先做 offline trace replay，再做同一场景的 bounded online 运行，最后才讨论 transfer 指标。

## 9. 实验设计与公平比较

### 9.1 主实验矩阵

| 研究层 | 场景/数据 | 控制变量 | 唯一变量 | 主要指标 |
| --- | --- | --- | --- | --- |
| HM3D 方法比较 | formal val，36 scenes/1000 episodes | 两机器人、RGB-D、地图、FMM、动作、500 steps | assignment policy 或模型身份 | SR、SPL、DTG |
| HM3D 机制审计 | 分层 12 episodes | 同一 episode、同一模型、同一 budget | detector/map/STOP 的一个接口 | 逐帧错误链、DTG、失败原因 |
| C2 B0/B1 | Cubicle/Open-plan/Octa | renderer、地图、速度、通信、统计端点 | 官方代码或单一执行修复 | completion、time、path、coverage |
| C2 B2 | 同一场景和 seed | B1 冻结后的全部设置 | spatial prior disabled/neutral/shuffled/live | 同上及先验敏感性 |
| Transfer | 配对区域/回放 trace | public boundary、局部执行器、预算 | 是否使用可迁移区域接口 | 迁移成功、可达率、回退原因 |

### 9.2 统计纪律

正式聚合应报告样本数、缺失数、成功定义、均值和离散程度，并给出 scene/category 分层结果。不要只汇报均值 makespan，也不要删掉失败 episode。不同方法的输出必须由同一个 schema validator 和同一个统计脚本读取。

### 9.3 当前 HM3D 开发证据

三局 DeepSeek 开发运行：

| episode | success | SPL | DTG/m | path/m | steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| 7 | 1 | 1.000 | 0.155 | 10.197 | 51 |
| 10 | 1 | 0.344 | 0.054 | 20.045 | 81 |
| 29 | 1 | 0.397 | 0.157 | 41.020 | 150 |

它们只有 3 个 `val_mini` episode，且使用 DeepSeek，不是 GPT-4o 原生结果。它们可用于检查调用、schema、fallback 和地图链，不能写成超过论文的证据。

固定 12 局、500-step、raw STOP 的 `greedy_papertext_v2` 开发结果为 SR 3/12、SPL 0.158274、mean DTG 1.643444 m；11/12 局观察到目标，但只有 3/11 成功，另有早期错误 STOP 和 timeout。该结果说明检测、持久化目标点云、FMM 目标和 STOP 链仍需审计，不能靠放宽成功距离修饰。

### 9.4 当前 C2 证据

B0 已在 Cubicle、Open-plan 多局完成，Octa 存在未完成局；官方论文 Octa 4-UAV 锚点为约 52.8±7.1 s 和 324.2±32.9 m。当前数据与论文锚点之间的差异必须先检查地图、渲染、统计端点、终止判据和代码版本，不能立即归因于算法优劣。B1 的 physical-region failure memory 仍处在逐事件审计阶段，B2 尚未满足启动条件。

## 10. 失败账本与诊断方法

### 10.1 HM3D 目标链失败

对每个失败 episode，按以下顺序检查：

1. 输入是否真的只有 RGB/depth/GPS/compass/文本；
2. 检测框是否出现，置信度是多少；
3. SAM 掩码是否落在目标表面；
4. 掩码内有效深度点是否达到阈值；
5. persistent object point cloud 是否写入且坐标稳定；
6. map goal 是否被选为 FMM 目标；
7. follower 是否到达合法终止区域；
8. STOP 是否满足 evaluator 的距离和调用条件。

这条链可以解释“11 局看到了目标却只有 3 局成功”这类现象。它也能避免把一个 `found_goal` 标志误当成任务 success。

### 10.2 C2 运行失败

C2 每局需要解剖：

- 每架机器人状态时间线：IDLE、PLAN、EXEC、REPLAN、FINISH；
- candidate、分配、通信和取消事件；
- collision、no path、handoff 和 replan 原因；
- 轨迹、速度、覆盖和完成方式；
- 是否发生 active method mode、FBL/VPOF 等运行期事件。

目录名、manifest 标签和“完成”字段优先级低于 telemetry。若 manifest 写着某个模式而运行期没有对应 `mode_initialized` 或 `suppression_active`，该 run 只能标为身份无效或 telemetry incomplete。

### 10.3 常见工程坑

- `episode_id` 在正式 val 中不唯一，必须配合 `scene_id` 和 source index；
- 并发 gzip 输出若共用 stem 会覆盖，必须使用 episode 独立 stem；
- 压缩 JSON 的结果行不能只看文件存在，要检查 schema 和 failure ledger；
- OpenCV headless、CPU detector fallback 等兼容修改必须记录，否则无法解释运行身份；
- L3MVN 的 full-map random sampler 不能直接拼接 Co-NavGPT2 FMM，接口不一致的尝试要保留为 `interface_mismatch`；
- `val_mini` 的高分不能外推到 36 scene/1000 episode；
- 计划、脚本、旧日志、视频和截图都不是论文级结果的替代品。

## 11. 当前状态与下一步

### 已建立的基础

- HM3D v0.2 正式 val tar 已归档、哈希、staging、runtime overlay 和 scene smoke；
- episode 5 / `00877-4ok3usBNeis` 已完成官方入口与共同协议 paired check；
- Co-NavGPT2 的 runner 已加入 scene binding、每机器人 reset、输出隔离和 LLM audit；
- Greedy、Cost-Utility、Random 三臂已经有共同协议开发运行，Multi-SemExp 仍需身份审计；
- C2 官方 release、FALCON、RACER 的源码、参数和运行结构已建立对照；
- HM3D 与 C2 的指标、身份和 transfer 边界已写入协作板和计划文档。

### 尚未完成的工作

1. 扩展带完整 LLM audit 的 HM3D 正式批次，并完成 1000 episode 的缺失/失败/scene/category 统计。
2. 逐局审计 Greedy 的检测、深度投影、持久化点云、map goal 和 STOP 链。
3. 补齐 Cost-Utility 的 lambda、归一化和重复分配规则来源，完成单因素配对。
4. 确定 Multi-SemExp 是否能在共同协议下运行；不能运行时保持 proxy/dependency-blocked 身份。
5. 完成 C2 B1 的 failure trace 审计，再决定是否启动 B2；不得在 B1 冻结前运行 B2 或 LLM/Ours 组合。
6. 两条主线身份冻结后，才开始 HM3D-C2 transfer 的 offline replay 和 bounded online 试验。
7. 所有正式结果还需要统一的 manifest、hash、raw log、失败账本和统计脚本归档。

## 12. 论文写作建议与可声称范围

### 12.1 论文主线

论文可以把 HM3D 语义搜索和 C2 几何执行写成一个“语义目标搜索—几何执行地基”的研究框架，但章节和表格要明确分工：

- HM3D 章节回答语义目标是否被发现、接近和成功停止；
- C2 章节回答未知空间如何连通、分配、规划和执行；
- transfer 章节只回答接口是否能把区域级候选和执行反馈迁移过去。

### 12.2 可以写的表述

- “在指定 HM3D v0.2、双机器人、RGB-D 和 500-step 合同下实现了可审计的协议重实现。”
- “在指定 scene/episode 上完成了 source-bound 的开发运行和失败链审计。”
- “C2 release 包含连通图、分配、轨迹和执行反馈组件，B0/B1/B2 采用分层复现设计。”
- “DeepSeek 作为外部模型替换，用于开发验证，不等同于 GPT-4o 原生复现。”

### 12.3 不能写的表述

- “已完整复现论文并超过作者结果”，除非论文规模、身份和统计均冻结且通过审计；
- “一次成功的 episode 证明方法有效”；
- “C2 completion 与 HM3D SR/SPL 可直接比较”；
- “存在官方代码所以所有模块都已经运行”；
- “formal” 目录、视频、截图或 smoke 结果自动等于正式得分。

## 13. 文件索引与阅读入口

### HM3D 主线入口

```text
E:\HM3D_2026_papers_and_repos\reproduction_tracking\HM3D_current_status_and_execution_plan_20260823.md
E:\HM3D_2026_papers_and_repos\reproduction_tracking\HM3D_弱基线正式执行计划_20260823.md
E:\HM3D_2026_papers_and_repos\reproduction_tracking\HM3D_弱基线源代码审计_20260823.md
E:\HM3D_2026_papers_and_repos\reproduction_tracking\HM3D_C2_cross_task_fusion_and_reviewer_defense_20260822.md
E:\HM3D_2026_papers_and_repos\reproduction_tracking\C2_HM3D_two_dialogue_coordination_20260822.md
```

### Co-NavGPT2 源码入口

```text
E:\HM3D_2026_papers_and_repos\official_repositories\Co-NavGPT2_native_one_ep5_20260823\main.py
E:\HM3D_2026_papers_and_repos\official_repositories\Co-NavGPT2_native_one_ep5_20260823\agents\vlm_agents.py
E:\HM3D_2026_papers_and_repos\official_repositories\Co-NavGPT2_native_one_ep5_20260823\agents\vlm_multi_agents.py
E:\HM3D_2026_papers_and_repos\official_repositories\Co-NavGPT2_native_one_ep5_20260823\utils\mapping.py
E:\HM3D_2026_papers_and_repos\official_repositories\Co-NavGPT2_native_one_ep5_20260823\utils\explored_map_utils.py
E:\HM3D_2026_papers_and_repos\official_repositories\Co-NavGPT2_native_one_ep5_20260823\utils\fmm_planner.py
E:\HM3D_2026_papers_and_repos\official_repositories\Co-NavGPT2_native_one_ep5_20260823\utils\detection_segmentation.py
E:\HM3D_2026_papers_and_repos\official_repositories\Co-NavGPT2_native_one_ep5_20260823\utils\chat_utils.py
E:\HM3D_2026_papers_and_repos\official_repositories\Co-NavGPT2_native_one_ep5_20260823\multi-robot-setting\habitat\
```

### C2 入口

```text
C:\Users\Administrator\ECRTA_C2\docs\C2_B0到B1到B2再到HM3D融合总执行计划_20260822.md
C:\Users\Administrator\ECRTA_C2\docs\C2_官方论文代码对照审计_20260822.md
C:\Users\Administrator\ECRTA_C2\docs\C2_连续执行_新对话总接管提示词_20260822.md
/home/c2dev/c2_explorer_reproduction/workspace/falcon_official/
/home/c2dev/c2_explorer_reproduction/workspace/racer_official/
```

## 14. 术语表

| 术语 | 含义 |
| --- | --- |
| HM3D | Habitat-Matterport 3D 室内场景与 ObjectNav 数据 |
| ObjectNav | 根据目标类别搜索并到达目标的导航任务 |
| frontier | 已探索自由空间和未知空间之间的边界 |
| FMM | 在栅格地图上求解距离场的 Fast Marching Method |
| CCL | Connected Component Labeling，连通区域标记 |
| ACVRP/ATSP | 带容量或非对称代价的车辆/旅行商分配问题 |
| LKH | Lin-Kernighan-Helsgaun 路径求解器 |
| B-spline | 用控制点和基函数表示的平滑轨迹 |
| SR/SPL/DTG | HM3D 的成功率、路径效率和终止距离 |
| completion/time/path/coverage | C2 的完成、时间、路径和覆盖指标 |
| native-reference | 保留作者入口或原始 evaluator 的参考身份 |
| protocol reimplementation | 按冻结输入边界重建的共同协议身份 |
| development-only | 用于调试和接口验证，不能进入论文主表 |
| transfer | 独立的跨任务接口迁移诊断，不是主线结果 |

## 15. 结语

这套系统的研究价值不在于目录里有多少脚本，而在于能否把“观测什么、知道什么、如何提出候选、如何分配、如何执行、何时算成功、失败如何保留”写成可重复的实验合同。HM3D 提供语义搜索问题，C2 提供几何探索和执行反馈地基；二者可以共享候选和执行接口的工程经验，但必须保留任务、真值、指标和证据身份的边界。后续最重要的工作不是继续扩展术语，而是完成论文规模运行、逐失败链核验和 source-bound 统计冻结。只有当这些证据闭合后，才适合把本项目从“研究复现系统”提升为论文中的正式方法与实验结果。

## 附录 A：文件级审计矩阵

下面的矩阵用于第二次阅读或交接时快速定位文件。它不是把每个历史产物都提升为当前入口，而是说明每一类文件在证据链中的位置。

| 层 | 代表文件/目录 | 需要回答的问题 | 通过标准 |
| --- | --- | --- | --- |
| 数据 | `datasets/hm3d_val_formal_*` | scene、navmesh、episode 是否来自同一正式归档 | tar/hash/staging/episode index 一致 |
| 任务 | `configs/objectnav_hm3d.yaml`、`multi_objectnav_hm3d.yaml` | 传感器、机器人数量、步数、成功距离是什么 | 配置 read-back 与 manifest 一致 |
| 环境 | `multi-robot-setting/habitat/` | 两个 agent 如何 reset、step、结束和计算 evaluator | 逐 agent 状态与官方行为对齐 |
| 感知 | `utils/detection_segmentation.py` | 检测框、掩码和有效深度怎样进入目标点云 | 保留每帧阈值、点数和坐标审计 |
| 地图 | `utils/mapping.py`、`explored_map_utils.py` | obstacle、explored、frontier 如何更新 | 地图尺寸、分辨率、坐标变换固定 |
| 规划 | `utils/fmm_planner.py`、`shortest_path_follower.py` | 候选如何转成路径和动作 | 路径、动作、重规划原因可追溯 |
| 分配 | `agents/vlm_multi_agents.py`、`system_prompt.py` | 模型/弱基线如何选择候选 | 输入字段、输出 schema、fallback 可验证 |
| 通信 | `utils/chat_utils.py` | 请求、token、延迟和失败如何记录 | 每次调用有 request/response/audit 记录 |
| 运行 | `run_conavgpt2_policy_smoke.py`、`run_hm3d_formal_dev_batch.py` | episode 如何绑定和批量执行 | source index、scene、stem、状态完整 |
| 统计 | `aggregate_hm3d_formal_full.py` 及聚合 JSON | 缺失、失败和分层结果怎样处理 | 不删除行、不填成功、保留 failure ledger |
| C2 | `source_latest/`、`scripts/`、WSL 官方仓库 | 地图、图、分配、轨迹和执行回执怎样串联 | source/binary/launch/telemetry 四者一致 |
| 证据 | `reproduction_tracking/`、`logs/`、`manifests/` | 结果是否能回到具体运行 | hash、版本、seed、原始日志齐全 |

### A.1 Co-NavGPT2 一次 step 的字段流

一次完整 step 至少包含以下逻辑字段。不同 runner 的 JSON 名称可能不同，但语义应保持一致：

```text
episode identity
  -> agent runtime pose and sensor packet
  -> detector boxes / masks / confidence
  -> valid depth count and projected points
  -> local map / explored map / frontier candidates
  -> assignment request and response audit
  -> selected goal and FMM path
  -> discrete action
  -> after-state / evaluator-side observation
  -> stop decision / failure reason
```

审计时应记录字段来自哪一层。比如 `success=1` 是 evaluator 输出，不能倒推 detector 真的正确；`found_goal=true` 是 agent 内部状态，不能倒推 evaluator 已成功；`action=STOP` 只是动作调用，不能倒推距离合同已经满足。

### A.2 C2 一次任务回执的字段流

C2 的计划—执行链可以抽象为：

```text
map update
  -> frontier/region candidate
  -> connectivity and reachability
  -> proposal
  -> commit / lease
  -> path and trajectory
  -> collision / communication / timeout check
  -> finalize or cancel
  -> receipt and coverage update
```

`Proposal` 只代表候选计划，`Commit` 代表被某个机器人接收，`Finalize` 才代表执行阶段结束；`Cancel` 和失败原因必须保留。这样的状态机可以防止“规划成功”被错误统计成“探索完成”。

## 附录 B：从零复现的检查清单

### B.1 HM3D 运行前

- [ ] Python、PyTorch、Habitat-Sim、Habitat-Lab 版本已记录。
- [ ] 正式 HM3D tar 的大小和 SHA-256 已记录。
- [ ] staging 与 runtime overlay 不是 example 或旧 minival 的软链接。
- [ ] scene、navmesh、episode index 和 category binding 通过检查。
- [ ] `scene_id` 与 `source_file/source_episode_index` 同时存在。
- [ ] `objectgoal`、目标坐标、goal annotation 和 evaluator 状态不在 policy 输入。
- [ ] API key 只从外部凭据文件或环境变量读取，绝不写进日志和文档。
- [ ] 5-step smoke 通过后才启动 30/100/500 step。

### B.2 HM3D 运行后

- [ ] raw JSON、stdout、stderr、逐步 trace 和 LLM audit 均存在。
- [ ] 每个 episode 有独立 output stem，没有 gzip 覆盖。
- [ ] 缺失行和失败行仍在 manifest/ledger 中。
- [ ] success、SPL、DTG 与 path/steps 的口径明确。
- [ ] 模型是 GPT-4o、DeepSeek 或其他替换，身份写入结果。
- [ ] 结果是否 development-only 已写入聚合文件。

### B.3 C2 运行前

- [ ] B0 源码为只读官方 commit；B1/B2 使用独立工作副本。
- [ ] Xvfb、Mesa、OpenGL/GLSL、ROS 和 simulator 版本已 read-back。
- [ ] 地图、传感器范围、速度、加速度、通信范围和统计端点固定。
- [ ] manifest 中的 method mode 与运行期 `mode_initialized` 一致。
- [ ] 运行前清理残留进程，但不删除历史证据。

### B.4 C2 运行后

- [ ] 每架机器人有完整状态时间线。
- [ ] 每个 A*、no-path、collision、handoff、replan 事件都有上下文。
- [ ] completion 的定义和终止时刻可以回到代码。
- [ ] 路径、速度、coverage 和通信统计使用同一统计端点。
- [ ] 失败局未被删除、覆盖或只保留摘要。
- [ ] 若没有新运行证据，不把新的文档段落写成实验进展。

## 附录 C：适合论文方法部分的叙述顺序

写论文时建议按“问题—信息边界—系统—执行—实验—证据”顺序，而不是按本地目录顺序罗列文件。

1. **问题定义**：先说明 HM3D 语义目标搜索和 C2 几何探索分别优化什么。
2. **信息边界**：明确策略可见的 RGB-D、GPS、compass、文本、局部地图和候选，列出禁止的目标真值。
3. **候选与分配**：解释 frontier/region 的生成、可达性筛选、分配策略和局部规划。
4. **执行反馈**：说明动作如何执行、何时 replan、哪些条件产生成功或取消回执。
5. **指标**：分别给出 SR/SPL/DTG 与 completion/time/path/coverage 的定义，不能共用一张表。
6. **公平性**：固定数据、场景、预算、传感器、控制器和统计脚本，只改变一个预定变量。
7. **结果身份**：将 native-reference、protocol reimplementation、model substitution 和 development-only 写在表头或脚注。
8. **失败分析**：给出漏检、错误 STOP、不可达 frontier、碰撞、通信过期和超时的分类统计。
9. **迁移实验**：最后说明区域级接口能否迁移，以及迁移不改变主任务真值和指标的理由。

这样的顺序能让读者先理解科学问题，再理解代码为什么这样组织；它也能避免把“有一个很长的运行目录”误读成“已经完成了论文实验”。
