# AeroCityBench：项目技术说明

## 300 字项目概括

AeroCityBench 是一个围绕程序化城市、四架 CF2X 和多 UAV 搜索/巡检任务设计的 benchmark 工程。它把 CitySpec、布局索引、目标无关的公共 inspection atlas、动作/观测 ABI、L0 几何控制器、L1 外部策略适配、Isaac/PhysX capture、release gate 和审计工具拆成独立组件。公共代码不暴露 evaluator-private target、support site、witness 或 seed；候选策略只能通过清洗后的 JSONL/process 边界访问公开观测，并由 manifest、源代码 revision、运行时 Python、checkpoint/solver lock 和文件哈希绑定。L0 负责校准和回放，不具备正式 score 资格；L1 需要真实 CF2X 原生执行、完整返回和安全证据，不能用 cuboid capability probe 或一段视频替代。项目的开发重点是科学治理：把 visual-only 装饰与 task geometry 分开、把 formal-score eligibility 写成机器可判定状态、拒绝跨布局复制和 stale receipt，并用 pytest、Ruff、资产来源和 release audit 守护公开边界。它展示的是 benchmark 与实验合同的工程能力；现有 native/preflight 结果应按 capability、development 或 formal 资格分别解释。

## 项目定位

- **任务族**：`exploration-3d / G1-U`、`geometry-search-3d / G2I`，以及单独的 perception-search 方向。
- **核心设置**：四 UAV、300 秒模拟执行预算、CF2X 候选执行器和程序化城市布局。
- **证据等级**：L0/CPU 合同、L1 外部/原生执行、visual review、development preflight、formal score eligibility 分开。

## 技术架构

```text
release config + CitySpec + asset registry
          -> layout materialization / public atlas
          -> ObservationPacket / MessagePacket ABI
          -> centralized or external planner adapter
          -> L0 runtime or Isaac/CF2X process boundary
          -> receipts / replay / audit
          -> readiness status and evaluator-facing report
```

`adapters.py` 负责 planner、Gymnasium/PettingZoo 包装器、JSONL 外部进程桥接和 replay writer；`runtime.py` 负责 L0 fleet step 与动作执行；builder/release 模块物化 CitySpec、presets 和 schema；tools 目录提供 capture、native gate、资产审计、统计冻结、manifest 构建和故障诊断。

## 核心代码结构

| 路径 | 作用 |
| --- | --- |
| `src/aerocity_bench/adapters.py` | public observation/action ABI、外部 process boundary、manifest、replay |
| `src/aerocity_bench/runtime.py` | L0 fleet runtime、动作投影、step receipts |
| `src/aerocity_bench/builder_v3.py`、`release.py` | CitySpec、布局、release 配置和哈希锁 |
| `src/aerocity_bench/atlas_leakage.py` | 公共 atlas 与 evaluator-private 字段泄漏检查 |
| `tools/isaac_capture.py`、`isaac_native_gate.py` | Isaac capture 和能力闸门 |
| `tools/audit_*`、`build_*_manifest.py` | 几何、资产、统计、外部适配、就绪状态审计 |
| `configs/`、`schemas/`、`tests/` | 版本化协议、schema、release preset 和回归测试 |

## 开发过程

1. 先定义公开/私有数据投影和 task geometry hash，避免 benchmark 产生目标泄漏。
2. 建立 ordinary release、layout materialization 和 L0 控制器，先在 CPU 上验证 geometry、返回、碰撞和 receipt。
3. 为外部 planner 增加真实进程边界，限制响应大小、超时、字段和非 ASCII key，并绑定上游 revision/许可证。
4. 接入 Isaac native gate、CF2X preflight、visual review 和资产许可审计；把 capability 证据与 formal score 分离。
5. 用 readiness status、统计协议和 release audit 作为提交前闸门，拒绝复制布局、过期回执和未绑定文件。

## 涉及知识点

- 程序化城市与场景几何、3D 搜索/巡检可见性
- 多 UAV 动作 ABI、集中式/分布式 planner 和 Gym/PettingZoo 适配
- Isaac Sim/PhysX、CF2X 执行、原生进程与安全超时
- JSON Schema、SHA-256、manifest、receipt chain 和 SBOM
- evaluator-private 数据隔离、target/atlas leakage 审计
- benchmark readiness、统计协议、formal eligibility 和复现治理

## 测试与验证

```powershell
uv sync --extra dev
uv run python -m pytest
uv run ruff check src tests tools
uv run aerocity-bench list-presets
uv run aerocity-bench show-preset ordinary-v1-mini
```

当前 CLI 还提供 `init-config`、`assets-verify`、`list-baselines`、`run-baseline`、`evaluate`、`probe-isaac`、`native-gate`、`capture-review` 和批处理命令。原生运行需使用明确验证的 IsaacLab Python；CPU test 只能证明合同和审计逻辑。`formal_score_eligible` 必须以当前 manifest 和完整 native episode 为准。

## 学术申请中的项目价值

该项目适合申请机器人 benchmark、具身智能评测、强化学习系统和可信 AI 方向。它的核心亮点不是“生成了一座城市”，而是把任务定义、信息泄漏、执行回执、资产许可和结果资格都转成可审计接口，体现研究基础设施与实验方法学能力。

## 运行入口

- CLI：`src/aerocity_bench/cli.py`
- L0 runtime：`src/aerocity_bench/runtime.py`
- 外部 planner：`src/aerocity_bench/adapters.py`
- Isaac capture/native gate：`tools/isaac_capture.py`、`tools/isaac_native_gate.py`

## 证据边界

L0 校准、cuboid capability probe、visual review 和 CF2X preflight 都不能自动升级为 formal L1 score。私有资产、evaluator target、真实运行时和本地 Python 不属于公开源码输入；任何结果必须保留 manifest、hash、receipt 和 eligibility 状态。

## 源码审计记录

| 层次 | 代表代码 | 复核内容 |
| --- | --- | --- |
| 协议 | `src/aerocity_bench/contracts.py`、`schemas/` | CitySpec、Pose3D、ObservationPacket、Action、Message 和 episode schema 的字段/单位/版本。 |
| 运行 | `src/aerocity_bench/runtime.py` | 四机 L0 step、动作边界、碰撞/返回、receipt 和物理时间。 |
| 适配 | `src/aerocity_bench/adapters.py` | public projection、集中式/分布式 planner、Gym/PettingZoo 包装和外部 JSONL 进程边界。 |
| 构建 | `builder_v3.py`、`release.py`、`config_loader.py` | layout materialization、preset、task geometry hash 和 output 目录拒绝 stale files。 |
| 泄漏 | `atlas_leakage.py`、`public_boundary.py` | 拒绝 target、count、support site、witness、seed 和 evaluator-private 命名泄漏。 |
| 原生 | `tools/isaac_capture.py`、`isaac_native_gate.py` | Isaac 进程、能力检查、before/after state、receipt chain 和超时。 |
| 外部 | `aco3d_g2i_process_adapter.py`、`ortools_g2i_process_adapter.py` | upstream URL/revision/license、source lock、JSONL request/response 和翻译边界。 |
| 审计 | `tools/audit_*`、`build_*_manifest.py` | 资产、几何、统计、baseline、L0/L1、readiness 和可视化审计。 |
| 回归 | `tests/` | schema、hash、泄漏、外部进程、release、native gate 和 formal eligibility。 |

## 一次公开策略调用链

1. release config 经过 schema 和 source lock 检查，materializer 生成公开 layout index、CitySpec 投影和版本哈希。
2. runtime 将每架 UAV 的 public observation 打包为 `ObservationPacket`，隐藏 evaluator-private target 字段。
3. PlannerAdapter 或 ExternalProcessPlannerBridge 接收 JSON-safe projection，检查 request id、action roster、响应大小、超时和禁止字段。
4. L0 runtime 同步执行 action segments，输出 before/after state、clearance、collision、return-home 和 receipt。
5. ReplayWriter 保存公开 episode 和 hash；validator 重新检查 manifest、source revision、upstream lock、receipt chain 和输出 schema。
6. readiness builder 将所有 gate 汇总为 `formal_score_eligible` 或 development-only 状态，不由一项成功的 native probe 自动升级。

## G1-U 与 G2I 的边界

G1-U 是目标无关的 3D exploration 视图，允许 coarse prior 和 occupancy；G2I 是 geometry-search，公共 atlas 只描述 roof/facade/entrance/rubble inspection regions。G2I 的 inspection credit 必须同时满足 range、FoV、surface facing、LOS、dwell、freshness、pose drift、clearance 和 safety，纯距离命中不算确认。当前 builder/native gate 是否接受某一 track，必须以当前 release config 为准，不能因为 adapter 存在就宣称正式 score。

## 资产与发布复核

`audit_release_assets.py`、`capture_polyhaven_provenance.py` 和 package/wheel 检查确保 USD、mesh、NVIDIA/Nucleus 路径、私有场景和 investigation artifacts 不被意外打包。第三方资产的 CC0/BSD/商业条款要与作者源码 license 分栏记录；“本地能加载”不等于“可以上传或再分发”。

## 交付检查表

- [ ] release 输出目录为空或由构建器明确创建，禁止混入旧文件。
- [ ] public projection 不含 target/evaluator/private 字段。
- [ ] 外部方法有真实 process boundary、上游 revision、许可证和 source hash。
- [ ] native capability、visual review、L0 calibration 与 formal L1 资格分开。
- [ ] 每次评测都保存 release/config/source/runtime/replay hash 和 readiness status。

## 原理与理论基础

### 1. Benchmark 的任务定义

一个可比较的 benchmark 至少要固定状态、观测、动作、终止、资源预算和评分函数。AeroCityBench 的 CitySpec 描述场景几何和布局；public projection 只向策略暴露允许的观察；evaluator-private 数据保存目标、合法 witness 和隐藏 split。这样可以把“算法知道什么”和“评测知道什么”分开，避免策略通过目标坐标、数量或随机种子获得捷径。

### 2. 3D 几何搜索

G2I 的观察合法性是多个条件的合取：距离在 range 内、目标/表面落入 FoV、法向 facing 合适、LOS 没有遮挡、dwell 足够、观测新鲜、位姿漂移和 clearance 合格。只有满足这些条件的观察才贡献 represented area。纯距离最小化不是搜索完成，因为它不保证看到了有效表面，也不保证观测可重复。

### 3. 执行证据与可比性

L0 控制器提供受控 oracle/校准路径，L1 需要外部策略在真实进程边界中执行。每个 control step 产生 before/after state 和 receipt，回执链把动作、状态、上一个回执和 evaluator decision 绑定起来。可比性要求方法使用同一动作预算、场景 split、控制器、传感器 profile 和统计脚本；否则差异可能来自接口或资源而非方法。

### 4. readiness 与形式化资格

formal_score_eligible 是多项 gate 的逻辑合取，而不是 native probe 成功的别名。可以把 release、source、runtime、asset、replay、safety 和 statistics gate 写成布尔向量 g，只有 product(g_i)=1 时才允许进入正式主表。development-only、capability-only、visual-only 和 transfer diagnostic 是合法状态，它们防止不完整证据被包装成正式排名。

### 5. 理论假设和失效模式

该 benchmark 假设 public/private 边界可信、场景物化可重建、外部进程不会获得隐藏字段、CF2X 状态可由原生运行时测量。复制 layout、伪造 receipt、超时强行成功、把 visual-only mesh 写进 geometry hash 或把 private target 传给 planner 都会破坏研究结论，因此对应审计必须成为代码路径而不是人工承诺。

---

# 深入实现导览：从 CitySpec 到可审计的多智能体搜索实验

## 0. 阅读范围与证据标记

本文依据当前工作树中的源码、配置、测试、README 和 `docs/` 研究记录整理。重点追踪的是一条可以实际运行的代码链：`ordinary-v1-mini.json` -> `generator_v3.py` -> `builder_v3.py`/`compiler.py` -> 公共任务投影 -> `L0FleetRuntime` 或原生适配器 -> 回执/回放 -> `metrics.py` 和治理审计。仓库同时保存历史版本、失活计划、实验收据和第三方适配器；它们被作为证据或背景读取，不自动等同于当前正式基线。

文中使用以下证据标签：

| 标签 | 含义 | 能支持的结论 |
| --- | --- | --- |
| `S` | 源码、配置、schema 或静态合同 | 说明接口和实现逻辑存在 |
| `T` | CPU/静态回归测试 | 说明输入校验、哈希、边界或确定性逻辑通过测试 |
| `L0` | 受控近似运行时或校准 | 说明合同和指标可以在开发运行时闭环，不能代表原生物理性能 |
| `L1` | 原生 Isaac/PhysX 执行收据 | 说明某个原生执行链在指定条件下运行过，仍须检查资格闸门 |
| `L2` | 视觉审查、资产或场景静态审计 | 说明可视化/依赖检查，不等价于策略得分 |
| `F` | formal score eligible | 只有正式 release、冻结合同、统计和独立运行全部满足时才允许进入主结果表 |

当前配置和 `authoritative-readiness-status-v1.json` 的结论是 `FORMAL_NO_GO`。因此本文不会把历史收据、L0 oracle、cuboid capability probe 或适配器 smoke 写成正式 benchmark 结果。

## 1. 先回答“这个项目到底在研究什么”

### 1.1 研究问题

AeroCityBench 研究的是：在三维城市结构和多智能体执行资源固定的情况下，**公共观测下的探索覆盖是否真的能转化为目标搜索/几何巡检成功**，以及当城市拓扑、目标生成过程和机群执行状态发生变化时，策略是否仍能保持可解释的执行行为。

这个问题包含两个容易被混淆的层次：

1. **覆盖层**：无人机是否经过了更多自由空间、屋顶、立面或入口附近区域。
2. **确认层**：是否在距离、视场、表面朝向、无遮挡、稳定停留和新鲜观测等条件同时满足时完成了合法观察。

只报告路径长度或自由空间覆盖，不能证明完成了目标搜索；只报告隐藏目标命中，又可能掩盖了策略使用了目标真值捷径。项目把这两个层次拆开，并用公开图集和私有评测器分别承载。

### 1.2 非目标

以下内容不是当前项目已经承诺的研究结论：

- 不是现实灾害中人员或目标分布的统计模型；
- 不是现实飞行器动力学、通信链路或传感器噪声的完整替代；
- 不是把任意外部仓库接上 JSONL 就能得到公平排名的通用沙箱；
- 不是已经完成的正式排行榜、PyPI 发布或足够统计功效的论文实验；
- 不是后续质量多样性、强化学习或视觉语言方法的训练场。README 明确要求这些方法在当前合同和正式闸门冻结后才可进入相应实验。

这组非目标很重要：它把 benchmark 基础设施、实验合同和后续方法研究分开，避免用方法代码反向改变任务规则。

## 2. 仓库地图：每个目录在整条链中的位置

### 2.1 顶层目录

| 目录 | 实际职责 | 阅读时的注意点 |
| --- | --- | --- |
| `src/aerocity_bench/` | 核心 Python 包：合同、生成、运行时、评测、适配器、指标、治理 | 当前实现的主线；先读这里再看 tools |
| `configs/` | 版本化 release、治理注册表、统计和 measurement protocol | 配置是实验规则的一部分，不是普通参数文件 |
| `schemas/` | CitySpec、episode、release config 和故障描述的 JSON Schema | 负责结构约束，语义约束仍由 Python validator 承担 |
| `tools/` | 构建、捕获、native gate、资产审计、统计冻结和诊断脚本 | 很多脚本是一次性证据生成器，不能把脚本存在理解为结果已经通过 |
| `tests/` | 合同、边界、原生前置检查、回放、统计和打包回归 | 文件名体现测试责任，执行时应按主机风险分组 |
| `docs/` | 研究问题、实验计划、方法边界、审稿风险和图示 | 计划文档与事实收据要分开阅读 |
| `assets/` | 资产注册表或本地资产入口 | 真实资产根目录可在仓库外，不能因为目录存在就上传第三方资源 |
| `external/` | 外部方法、锁定源码或本地依赖说明 | 外部方法的 URL、revision 和 license 必须与适配器声明配对 |
| `reason/` | 运行收据、审计报告和历史决策材料 | 证据文件可能是未提交或历史版本，需检查哈希和状态 |
| `debug/` | 调试输出 | 不能作为正式输入或发布物 |

### 2.2 核心模块的分层关系

```text
ordinary_config.py / configs/*.json
        |
        v
generator_v3.py -> targets_v3.py -> CitySpec + private episode
        |
        +--> compiler.py -> scene.usda / collision.usda / task projection
        +--> inspection_atlas.py -> target-independent G2I atlas
        +--> builder_v3.py -> split/layout manifest + hashes + readiness inputs
        |
        v
public_boundary.py -> ObservationPacket / ActionPacket -> adapters.py
        |                                           |
        |                                           +--> L0FleetRuntime
        |                                           +--> ExternalProcessPlannerBridge
        |                                           +--> Gymnasium/PettingZoo wrapper
        v
evaluator.py + runtime.py -> Observation/Execution/Confirmation receipts
        |
        +--> measurement_evidence.py / measurement_aggregator.py
        +--> metrics.py -> run metrics
        +--> experiment_governance.py -> development/formal status
```

这张图表达的是数据依赖，不是进程拓扑。真正的 Isaac、CF2X、ROS 或 container 运行都要通过自己的 runner 和环境锁；`ExternalProcessPlannerBridge` 只实现本地 process 声明。

## 3. 配置如何变成一个可复现 release

### 3.1 `ordinary-v1-mini.json` 的关键冻结项

当前普通配置给出了一个小规模开发/发布候选：生成器版本为 `aerocity-generator-ordinary-v3.1`，主种子为 `20260729`，核心城市边长为 80 m，拓扑 OOD 边长为 96 m，机群规模固定为 4，单回合模拟执行时间为 300 s，返航预留 35 s。拆分包括 `train`、`validation`、`calibration`、`test_iid`、`test_topology` 和 `test_process_ood`。

配置还固定了：

- 80 m 布局的私有目标数量范围为 6--8，96 m 为 8--10；
- 开发拓扑族为 `offset_grid`、`t_junction`、`dead_end_courts`，拓扑 OOD 使用 `ring_spokes`、`staggered_loop`；
- 目标生成过程按 split 绑定，`test_process_ood` 使用各向异性聚类过程；
- 资产包仅允许注册表中的五类 CC0 资产，`custom_official_forbidden=true`；
- 控制周期为 0.2 s，规划截止时间为 0.15 s，规划 cadence 为 1 s 的固定速率加公开事件；
- 观察要求连续停留 0.5 s、冷却 0.5 s，线速度不超过 0.25 m/s，角速度不超过 8 deg/s，源观测新鲜度不超过 0.4 s，最大观察距离 3 m，水平/垂直视场为 68/51 deg；
- 车辆半径 0.32 m，水平/垂直速度上限 3/2 m/s，最小净空 0.75 m，能量预算 24000 J；
- 通信范围 65 m，带宽 16000 bytes/s，延迟 0.06 s，丢包概率 0.03，消息 TTL 0.8 s；
- 硬碰撞会使个体终止并使安全排名失格，越界动作拒绝并悬停。

这些字段不是“建议参数”，而是合同的一部分。实验比较时必须使用同一 release、同一执行合同和同一统计脚本，否则方法差异无法与规则差异区分。

### 3.2 生成器的确定性

`generator_v3.py` 的职责是根据 release 配置和 seed 构造 CitySpec。它先确定城市边界、道路图和建筑 grammar，再生成障碍、视觉装饰和起点相关几何。建筑 grammar 包括矩形、 slab、窄 slab、L 形、U 形、庭院、塔台和阶梯塔等；组件被拆成真实碰撞体，因此屋顶、立面和入口的几何检查可以使用同一套 collider。

生成器中有三个值得学习的设计：

1. **几何与装饰分离**：道路、建筑组件和障碍参与碰撞/任务几何；街灯、树、箱体和立面点缀被标记为 visual-only，编译时关闭碰撞。
2. **从真实组件面选择入口**：`_face_anchor` 不直接把 compound building 的外包络当成入口面，而是从实际组件的南/东/北/西面中选择，避免入口落在 U 形开口或庭院空洞上。
3. **哈希绑定**：layout、task geometry、asset set 和 release manifest 均参与后续 hash。装饰可以改变画面，但不应改变隐藏目标、起点或粗先验的任务几何哈希。

### 3.3 目标过程与目标有效性

`targets_v3.py` 负责在私有侧把候选支撑面转成目标记录。配置控制支撑面/目标比例、最小目标间距、周围碰撞体数量、见证点距离、垂直跨度以及单次观察最多确认的目标比例。目标必须在执行前通过有效性检查，写入 `target_validity` 清单并冻结 `validity_hash`；`PrivateEvaluator` 初始化时再次检查目标数量、有效性标志、ID 集合和执行合同哈希。

这一步的研究意义是防止“先跑结果、再挑合法目标”。目标在 run 开始后不能因为策略表现而重采样、移动或改变数量；否则同一个方法的成绩不再对应同一个任务。

### 3.4 `builder_v3.py` 的发布闸门

构建器采取“新目录一次性构建”的策略：如果输出目录已经存在，直接拒绝，而不是覆盖或增量写入。内部流程大致为：

1. 加载 release config 和资产锁，校验版本、split、目标过程和执行合同；
2. 逐 layout 调用生成器，拒绝重复 layout hash；
3. 写 `authority_private/`，包括私有 episode、支撑面和目标有效性摘要；
4. 写 `method_public/`，只包含公开任务、公开 episode、起点和必要哈希；
5. 调用 `compiler.py` 生成 `scene_authority` 下的 USD 和碰撞层；
6. 对每个目录生成文件 SHA-256 清单、layout manifest 和 release manifest；
7. 运行公共边界、资产合法性、native gate 输入和 readiness 检查；
8. 只有满足发布条件时才导出公开 release，否则保留开发构建并明确 `formal_score_eligible=false`。

构建器的核心不是“写很多文件”，而是让任何一个文件被修改后都能在后续验证中暴露。目录级 manifest 解决文件缺失/替换问题，内容 hash 解决字段被改写问题，source commit 和 registry hash 解决证据与代码版本错配问题。

## 4. 场景编译：视觉层、碰撞层和公共投影

`compiler.py` 将 CitySpec 编译为三个层次：

### 4.1 `scene.usda`

该层写出 ground、roads、buildings、obstacles、urban ground detail、visual decorations 和 procedural visual detail。道路根据起点/终点或轴向计算长度、中心、旋转角和前进方向；人行道和道路标线属于视觉增强。建筑每个 component 生成独立 USD Cube，保持与几何审计相同的中心和尺寸。

### 4.2 `collision.usda`

碰撞层只把 ground、建筑组件和障碍体挂上 `PhysicsCollisionAPI`。视觉装饰不进入碰撞组，立面点缀明确 `physics:collisionEnabled=false`。这样在改变资产模型、颜色或装饰时，任务几何和安全净空不会悄然变化。

### 4.3 `stage.usda` 和资产引用

`stage.usda` 以 sublayer 组合场景和碰撞层；装饰资产用相对 `_assets` 路径引用，并通过 `AssetLock` 查找注册表记录。编译器不自动下载 NVIDIA/Nucleus 内容，也不把本机绝对路径写入 release。正式发布时应再运行资产审计，确认 USD 依赖闭合、许可证和再分发策略一致。

## 5. 公共/私有边界：为什么不能把目标坐标放进 observation

### 5.1 公共输入

`ObservationPacket` 是方法可见的核心数据包，包含 episode/agent/sequence/time、车辆姿态、世界系线速度、角速度、剩余能量、局部占据体素、占据原点和分辨率、队友摘要、收到的消息、健康状态以及传感器俯仰角。`ActionPacket` 支持 `HOVER`、`WAYPOINT`、`VELOCITY`、`OBSERVE`、`RETURN` 五类动作。

### 5.2 明确禁止的字段

公共 task/episode 和外部 process wire 不能包含目标坐标、目标数量、支撑面 ID、见证点、私有 split、私有 seed、evaluator witness 或隐藏的完整 ESDF。`public_boundary.py` 递归检查字段名，大小写、连字符和下划线变体都会归一化；非字符串或非 ASCII JSON key 也直接拒绝。

公共 episode 保留 `target_count_public=false`、`target_process_public=false` 这两个审计哨兵，用来证明字段没有公开；发送到外部进程前，`_external_process_episode_projection` 会验证哨兵为 false 后将其移除，避免外部策略利用“字段存在”分支。

### 5.3 为什么“process boundary”仍不是盲评测沙箱

`ExternalProcessPlannerBridge` 通过标准输入/输出维持 JSONL 长连接，限制请求/响应大小、请求 ID、动作 roster、响应 schema 和时间；超时或进程退出时只终止自己创建的进程树。它能隔离许可证和接口错误，但仍运行在同一主机权限模型中。真正的盲评测还需要只读挂载、网络策略、容器 digest、ROS 图隔离和 evaluator 服务治理；不能把本地 process 适配器的存在描述为已经完成的安全沙箱。

## 6. G1-U 与 G2I：两个任务视图必须分开排名

### 6.1 G1-U：目标无关的三维探索

G1-U 主要暴露局部 occupancy、粗粒度公共先验和队友状态，适合研究 frontier、覆盖、通信和路径规划。它不提供定义清晰的 inspection object，因此旧版只做 coverage diagnostic 的 G1-U 结果不应被包装成隐藏目标搜索分数。

### 6.2 G2I：目标无关图集上的几何巡检

G2I 的公共图集由 `inspection_atlas.py` 生成，区域类别包括 `roof`、`facade`、`entrance` 和 `rubble`。图集给出候选 inspection cell、姿态、表面点、表面法向、姿态容差、代表面积和 safe-sky transit graph；它描述“应覆盖哪些结构区域”，不描述“真实目标在哪里”。

图集编译过程包括：

1. 从 CitySpec 提取与目标无关的结构几何并计算 `inspection_geometry_hash`；
2. 根据 sensor rig、FoV、range 和采样策略生成候选 cell；
3. 用同一套 flight bounds、净空、相机姿态、距离、FoV、surface facing、LOS 和 safe-sky 规则做 geometric admission；
4. 拒绝所有不可能合法执行的 cell，避免把永远不可达的点放进分母；
5. 生成带哈希的 transit graph 和 mission sector；
6. 由 `validate_public_inspection_atlas` 与 `validate_public_mission_sector` 检查字段和绑定关系。

### 6.3 合法 observation 的合取条件

G2I 的观察信用可以形式化为：

\[
I(c,t)=R(c,t)\land F(c,t)\land Q(c,t)\land L(c,t)\land D(c,t)\land N(c,t)\land S(c,t),
\]

其中 `R` 是范围，`F` 是视场，`Q` 是表面法向朝向，`L` 是无遮挡视线，`D` 是连续 dwell，`N` 是源观测新鲜度/序列约束，`S` 是速度、姿态漂移、净空和运行时安全。任一条件失败，当前 dwell 会被重置，不贡献 inspection area。

## 7. `contracts.py`：把动作、观测和证据变成版本化数据

### 7.1 `Pose3D`

`Pose3D` 保存位置、yaw、pitch、roll；位置必须是三向量，角度必须落在有限范围。传感器俯仰可以在 `ActionPacket.sensor_pitch_deg` 和 `ObservationPacket.sensor_pitch_deg` 中独立表示，避免把有界云台运动错误地当成机体姿态。

### 7.2 `MessagePacket`

消息绑定发送者、接收者集合、创建/过期时间和 bytes payload。构造函数拒绝空 ID、空目的地、过期时间早于创建时间和非 bytes 载荷。运行时据此统计发送、投递、丢弃、过期和带宽拒绝。

### 7.3 `ActionPacket`

动作在构造阶段就检查 episode/agent/sequence/time。`WAYPOINT` 必须有 pose，`VELOCITY` 必须有 body-frame 速度；只有 `OBSERVE` 能绑定 `source_observation_id`。这个约束把“看到了什么”绑定到一个具体、可审计的旧观测，防止策略发送一个无来源的成功观察。

### 7.4 `ExecutionReceipt`

每个执行段记录请求动作和实际动作、时间、规划延迟、距离、能耗、最小净空、碰撞、越界、安全干预、deadline miss、执行级别、action hash、source observation hash、执行前后状态 hash、上一条 receipt hash 和 confirmation IDs。receipt 的 hash 链可检测删除、重排和跨动作复制；但它只能证明证据结构完整，不能替 evaluator 证明目标真的存在。

## 8. L0FleetRuntime：开发校准用的受控运行时

### 8.1 运行时状态

`runtime.py` 为每架机维护位置、姿态、速度、能量、健康状态、消息队列、覆盖轨迹和上一次 receipt。`reset()` 校验起点净空并生成第一批 observation；`step()` 对所有活跃 agent 的动作执行同一段任务时间，输出新的 observation、`RuntimeStep`、receipt 和失败记录。

L0 的运动更新是可控的近似执行模型，使用 release contract 的速度、加速度、yaw rate、半径、净空和能耗参数。它适合验证动作 ABI、回执链、通信计费、返回家点、coverage trace 和指标聚合；`formal_score_eligible` 明确为 false，不能把 L0 当成 CF2X/PhysX 物理结果。

### 8.2 动作执行与安全顺序

一次 `step()` 的逻辑可概括为：

1. 检查 episode 是否已终止、动作 roster 是否完整、sequence 是否可接受；
2. 根据动作种类计算请求目标或速度，应用传感器俯仰限制；
3. 计算边界和碰撞，越界动作按 contract 拒绝并悬停；
4. 根据离散时间/速度约束生成实际位移，更新能量和路径距离；
5. 处理硬碰撞、净空和 deadline，必要时生成 terminal `FailureRecord`；
6. 投递/丢弃消息并累积通信 ledger；
7. 对 `OBSERVE` 调用私有评测器，更新确认和公共图集 dwell；
8. 计算 measured/public observation，写入 action/state hash 和 receipt；
9. 更新时间、coverage trace、inspection trace 和 terminal 状态。

顺序不能任意交换。例如，必须先经过 runtime safety，再向 measurement evidence 计入观察；否则策略可能在碰撞或净空违规的姿态上获得 inspection credit。

### 8.3 时间与规划 cadence

控制周期是 0.2 s，但规划 cadence 可按 1 s 固定速率触发，并允许 anonymous confirmation、safety intervention、fleet roster change 和 return-reserve-entry 等公开事件提前触发。未触发规划时，策略动作按“最新公共观测”重新绑定；消息不因为 held action 自动重传。规划超时依照 `overrun_policy` 保持上一安全动作或悬停，连续 deadline miss 达到阈值后终止或标记失败。

这使计算延迟成为预算的一部分，而不是被墙钟时间悄悄忽略。适配器还要单独报告接口税（projection、序列化、等待响应、校验和转换），方便判断比较差异来自算法还是桥接层。

## 9. `PrivateEvaluator`：目标真值只在评测侧出现

### 9.1 初始化与冻结

评测器构造时接收 authority-side private episode、城市碰撞体和至少 16 字节的 receipt secret。初始化检查 layout hash、目标数量、所有目标 `valid_before_run=true`、target-validity manifest hash、冻结标记以及 execution contract hash。任何一项不匹配就拒绝运行。

### 9.2 可见性检查

`_visibility()` 使用传感器姿态而非简单 body pose，依次检查最大距离、水平/垂直 FoV、表面法向 facing 和 line-of-sight。遮挡判断使用城市 collider，并忽略目标所属的 collider，避免目标自身把自己判定为遮挡。

### 9.3 dwell、冷却和去重

`process()` 只接受绑定了当前 observation 的 `OBSERVE` 动作。它还检查 source freshness、线速度、角速度、连续 dwell、最大 pose drift、observe session cooldown 和 observation ID 是否重复。目标在全局只计数一次；多机同时看到同一目标时，私有侧会记录冗余/同时确认，但不会把一个目标加成多次。

确认回执只暴露匿名 target handle、confirmation ID、agent ID、确认时间、源 observation ID 和 HMAC receipt token。真实 target ID、坐标和 dwell 细节留在 private audit snapshot，不能进入方法输入或公开报告。

## 10. 公共 measurement evidence 与指标

### 10.1 为什么不能直接从总结表填指标

`measurement_evidence.py` 从测量到的原生位置、公开观察和 receipt 重新构造两类 trace：2 m 体素自由空间覆盖，以及通过安全 `OBSERVE` 后获得的公共 atlas 面积。快照保存输入绑定哈希、状态轨迹哈希、coverage denominator 和三条时间序列。`measurement_aggregator.py` 可以在受保护环境中重建几何可见性，避免手工编辑报告就改变结果。

### 10.2 主要指标

`metrics.evaluate_run()` 计算：

- `confirmed_recall_auc`、最终确认召回率和到达 25/50/75% 的时间；
- 2D/3D 自由空间 coverage AUC；
- G2I 的面积加权 `inspection_footprint_auc` 和 cell-count 诊断；
- 碰撞、越界、安全/净空干预、是否全部返航；
- 路径距离、能耗、每次确认能耗；
- planning latency 的 median/P95/P99 和 deadline miss；
- 通信字节/包的发送、投递、丢弃、过期、重复和带宽拒绝；
- 每机距离、距离 Jain fairness、Gini、不均衡归因；
- 按 support class、altitude band 和 opportunity tercile 的分组召回。

面积加权 inspection footprint 是主要 G2I 几何指标，cell count 只用于解释分辨率/采样影响。每一项指标都应随 denominator、semantics、run hash 和 eligibility 一起保存。

### 10.3 统计协议

当前 G2I 统计配置将 `layout_ancestor` 作为独立单位，以 `layout_hash` 做配对键，主要结果是 `mean_final_confirmed_recall`，设定双侧 alpha 0.05、power 0.8、最小可检测效应 0.1、至少 29 个独立 ancestor、5000 次 bootstrap 和 Holm 校正。失败的已完成 ancestor 必须留在分母，不能只报告成功回合。

这意味着“跑了 29 个 episode”不等于“有 29 个独立样本”。同一 layout 的多个 process 或 episode 可能只是配对重复，不能伪装成独立祖先。

## 11. 外部方法接入的实际步骤

### 11.1 适配器声明

外部方法需声明 adapter ID、能力 profile（G1 或 G2I）、执行边界、集中/分布式属性、上游 URL、完整 revision、许可证和 source hash。`ExternalProcessPlannerBridge` 当前只接受 `process` boundary，拒绝把 container 或 ROS 声明伪装成本地可执行文件。

### 11.2 JSONL 交互

进程启动后先发送 `reset`，再以每行一个 JSON 对象发送 `act`。请求带 schema、request ID、kind 和公共投影；响应必须带相同 request ID、`status=ok` 和 actions。`act` 的 actions 键集合必须与活跃 observation agent 集合完全相同，每个动作再经 `_action_from_dict` 转换为 canonical `ActionPacket`。

### 11.3 失败处理

以下情况都会 fail-closed：非 ASCII key、私有字段、超出行长度、响应超时、JSON 损坏、错误 request ID、额外响应字段、动作 roster 不匹配、进程提前退出。桥接层记录初始化延迟和每次 act 的分段耗时；超时只清理自己创建的进程树。

### 11.4 公平比较的必要条件

比较外部方法时必须统一：公开观测 profile、动作语言、控制器、规划预算、返回预留、通信约束、城市 split、随机种子生成规则和统计脚本。仅比较“外部程序最终返回了一个动作”不构成科学可比性；适配器可通过不当降采样、额外缓存或隐藏预处理改变资源占用。

## 12. 推荐的完整实现与实验顺序

下面是一条适合从零理解和复现的执行顺序。每一步都要保存命令、输入 hash、输出 hash 和证据等级。

### 阶段 A：环境与合同

1. 使用 Python 3.11+ 建立干净环境，安装项目及 dev 依赖。
2. 运行 `list-presets`、`show-preset ordinary-v1-mini`，确认配置被当前 Python 解析。
3. 运行 focused contract tests，先验证 schema、public boundary、hash 和 config validator。
4. 记录 Python 版本、操作系统、Git commit、依赖锁和环境 manifest。

### 阶段 B：资产与 release 构建

1. 准备经过注册表验证的 CC0 资产根目录，设置 `AEROCITY_ASSET_ROOT` 和 `AEROCITY_OUTPUT_ROOT`。
2. 使用新的空输出目录执行 `python -m aerocity_bench build --release ... --allow-uncommitted-development`。
3. 运行 `validate`、`audit_release_assets.py`、`audit_scene_geometry.py` 和 `audit_public_boundary.py`。
4. 对照目录 manifest 检查 `method_public` 与 `evaluator_private` 是否分离，确认视觉装饰没有进入碰撞/任务 hash。

### 阶段 C：L0 合同闭环

1. 用 L0 runtime reset，确认四个起点合法且公共 episode 能通过校验。
2. 先运行简单 HOVER/WAYPOINT/RETURN policy，验证位置边界、能量、返回和 collision handling。
3. 再加入 OBSERVE，检查源 observation、dwell、cooldown、confirmation 和 public atlas trace。
4. 用 ReplayWriter 写回放并调用 `validate_replay`；用 `metrics.evaluate_run` 计算指标。
5. 运行故障注入：缺失 source observation、重复 observation、过期消息、超时、越界和 receipt 删除，确认全部拒绝或 fail-closed。

### 阶段 D：外部策略与原生能力

1. 先用本地 dummy JSONL planner 验证 process bridge 的 request/response、超时和字段清洗。
2. 绑定真实 upstream revision 和 license 后，生成 external adapter manifest。
3. 使用 IsaacLab 独立 Python 运行 `probe-isaac` 或 capture review，只记录能力/视觉证据。
4. 使用 CF2X native preflight 验证控制器状态推进、净空、碰撞和返航，再进行多机短时闭环。
5. 只有当原生 episode、回执链、环境锁和场景/许可证审计均满足时，才讨论 L1 calibration；这仍不自动意味着 F。

### 阶段 E：正式统计前冻结

1. 冻结 release config、execution contract、source commit、asset registry 和统计协议。
2. 使用 calibration split 做方法无关的 A-gate；不能用 calibration 结果改变任务边界。
3. 运行至少三种行为机制、多个独立 layout ancestor 和失败保留策略，审查 route/observation 是否存在捷径。
4. 运行外部方法同权限闭环、场景/许可证检查、clean wheel install 和 recovery drill。
5. 由 `experiment_governance.py` 生成 readiness status。若任一必需项为 false，主表状态必须保持 `FORMAL_NO_GO`。

## 13. 研究实验应该如何设计

### 13.1 研究因素与控制变量

建议把因素分成三组：

| 因素 | 研究含义 | 必须固定的对照 |
| --- | --- | --- |
| 城市拓扑 | IID 与 topology OOD 的结构变化 | 机群、动作、预算、传感器和评测合同 |
| 目标过程 | uniform/clustered/height-stratified 与 OOD 过程变化 | layout ancestor、公共输入和随机规则 |
| 执行状态 | 通信丢包、deadline、能量、返航和安全干预 | 任务几何、方法观测权限和统计端点 |

“方法是否更好”只能在同一行控制变量内解释。若同时改了 planner、控制器、消息编码和地图分辨率，结果不再能归因于单一方法机制。

### 13.2 基线层次

建议按以下顺序建立 baseline，而不是一开始就训练复杂模型：

1. **安全悬停/返航**：验证终止、能量和 return reserve；它不追求搜索得分。
2. **覆盖扫描**：按公共 occupancy/atlas 规划固定航线，作为几何上限和执行下限之间的参照。
3. **区域贪心**：按公共 atlas region 的距离、剩余面积和队友分配做确定性选择。
4. **表面巡检器**：在合法 cell 上加入姿态、FoV、LOS、dwell 和 safe-sky 约束。
5. **外部三维方法**：只有 source/revision/license、同一 process boundary 和同一 action budget 完成后才纳入。

基线的价值不是“制造一个很强的对手”，而是把 coverage-to-search gap、通信税、执行失败和几何可达性分别测量出来。

### 13.3 必做消融

- 去掉 public atlas，只保留 occupancy，测量几何先验是否造成捷径；
- 改变 atlas 采样密度，比较 area-weighted 与 cell-count 的稳定性；
- 固定路线但改变目标过程，检验目标过程 OOD 是否真的可见于指标；
- 固定 planner 但改变通信丢包/延迟，报告确认率、返航和消息统计的变化；
- 保持同一 layout hash，交换 agent ID 或起点，检查是否存在个体编号捷径；
- 在相同动作控制器下比较 L0 与 L1，报告排名偏差而不是直接把两者成绩混排。

### 13.4 失败样本如何处理

碰撞、越界、deadline failure、进程崩溃和没有返航都要保留。不能只保留“完成的漂亮视频”，也不能以重跑后的成功覆盖第一次失败。每一个失败应有 episode、agent、任务时间、category、detail、terminal 标志，并绑定运行 manifest。统计中按协议保留已完成 ancestor 的失败行；未完成的进程错误另列为执行完整性审计，不可无声删除。

## 14. 理论知识点：从代码映射到方法学

### 14.1 部分可观测多智能体决策

每架机的输入是自身姿态/速度/能量、局部 occupancy、队友摘要和已收到消息，而不是全局目标状态。因此它可以抽象为多智能体部分可观测决策过程：全局状态包含城市和私有目标，agent 只能获得公共投影。`MessagePacket` 使通信成为显式动作/资源，消息 TTL、带宽和丢包让“共享信息”有代价。

### 14.2 几何可见性

目标或 atlas cell 的合法观察不是球形邻域，而是距离、视场锥、表面法向半空间、障碍物 LOS 和姿态稳定性的交集。代码中的 `in_field_of_view`、`surface_facing`、`line_of_sight` 和 `minimum_clearance` 对应这四类几何约束。将其拆成独立函数，有利于测试边界条件并解释失败原因。

### 14.3 覆盖与搜索的差异

自由空间覆盖可以写成轨迹访问体素集合的并集；搜索成功则需要访问“任务对象”的合法观察集合。前者是空间占据问题，后者是带传感器和执行状态的几何观察问题。项目分别保留 `coverage_trace`、`inspection_coverage_trace` 和 confirmation receipts，避免用一个指标代替另一个问题。

### 14.4 执行回执的可追溯性

receipt 链相当于对动作序列和 measured state 的 hash-linked log。它不提供机密性，也不替代可信硬件，但可以检测回执删除、顺序变更、动作与观测错绑和跨 episode 复制。配合 `source_commit`、config hash、asset hash 和 replay hash，可以把“一个数字”还原为一组可复核输入。

### 14.5 统计独立单位

布局祖先是更接近任务随机性的独立单位；同一城市上改变 target process 或 episode seed 仍可能共享大量几何结构。以 layout hash 配对并在 ancestor 层做 bootstrap，可以减少把同一结构的重复测量误当独立证据的风险。

## 15. 常见错误与源码中的防线

| 错误 | 结果 | 代码防线 |
| --- | --- | --- |
| 把 target 坐标写进 public observation | 方法可使用评测真值 | `public_boundary.py` 递归字段拒绝 |
| 用距离命中代替合法观察 | 夸大搜索能力 | evaluator/atlas 的 FoV、facing、LOS、dwell 和稳定性检查 |
| 把视觉装饰写进碰撞体 | 画面变化偷偷改任务难度 | `compiler.py` 分离 visual/collision layer |
| 在已有输出目录上增量构建 | stale 文件混入 release | `builder_v3.py` 拒绝已存在输出目录 |
| external process 超时后继续等 | 规划计算不计入预算 | bridge 有界超时、终止和 latency receipt |
| 把旧 v16 panel 当当前正式结果 | 合同 hash 不一致 | readiness/status 和 governance registry 标为 superseded/blocked |
| 只展示成功视频 | 忽略安全失败和失败分母 | `FailureRecord`、ledger、manifest 和统计保留规则 |
| 用 L0 oracle 校准后训练复杂方法 | 方法反向适应 benchmark | governance 中 `method_results_may_tune_benchmark=false` 和 training blocked |

## 16. 如何自己读懂这套代码

推荐以“数据流”而非文件字母顺序阅读：

1. 先看 `configs/releases/ordinary-v1-mini.json`，列出所有会影响几何、动作、时间、通信和评分的字段。
2. 阅读 `ordinary_config.py`，理解每一项如何被精确 key、正数范围、split 映射和安全合同校验。
3. 阅读 `contracts.py`，把每个 packet 的字段、单位、生命周期和哈希关系画成表。
4. 阅读 `generator_v3.py` 与 `targets_v3.py`，追踪一个 building 从 grammar 到 collider、支撑面和私有目标。
5. 阅读 `compiler.py` 与 `builder_v3.py`，观察同一 CitySpec 如何变成 USD、public task 和 private episode。
6. 阅读 `public_boundary.py` 与 `inspection_atlas.py`，验证目标相关信息在哪里被拒绝、公共 cell 如何被 admission。
7. 阅读 `runtime.py`、`evaluator.py`，追踪一个 `OBSERVE` 从动作构造到 receipt、confirmation 和 coverage credit。
8. 阅读 `adapters.py`，检查外部方法实际拿到的 JSON，而不是只看 adapter 名称。
9. 阅读 `metrics.py`、`measurement_evidence.py`、`experiment_governance.py`，确认每个数字的原始 trace、分母、哈希和状态。
10. 最后读 `tools/` 和 `tests/`，把每条命令对应到一项合同或失败模式。

建议在本地用一个最小 episode 做手工追踪：记录一次 HOVER、一次 WAYPOINT、一次 `OBSERVE` 失败、一次成功 `OBSERVE`、一次 RETURN，逐字段比较 observation、action、receipt 和最终 report。这样比直接阅读数万行历史报告更容易理解设计意图。

## 17. 可复现命令与检查清单

以下命令来自当前 README 的开发路径，具体 Python、资产根目录和输出根目录必须替换为本机已验证路径：

```powershell
$env:PYTHONPATH = (Resolve-Path .\src)
$assetRoot = (Resolve-Path $env:AEROCITY_ASSET_ROOT)
$outputRoot = Join-Path $env:AEROCITY_OUTPUT_ROOT 'ordinary-v1-mini'
python -m aerocity_bench build --release .\configs\releases\ordinary-v1-mini.json `
  --asset-root $assetRoot `
  --output $outputRoot `
  --allow-uncommitted-development
python -m aerocity_bench validate $outputRoot
```

质量检查应分组运行：

```powershell
python -m pip install -e ".[dev]"
.\tools\run_python_quality_gate.ps1 -Python python
python -m pytest tests/test_public_boundary.py tests/test_ordinary_v3.py tests/test_inspection_atlas.py
python -m pytest tests/test_experiment_governance.py tests/test_statistical_protocol.py
```

不要在本机把所有会生成子进程或启动 Isaac 的测试直接合并成一个无边界的 `pytest -q`；README 已说明 Windows 主机上应先运行 focused groups，再对 process/native 任务使用 host guard 和独立收据。

## 18. 目前可以怎样准确介绍这个项目

### 面向导师或简历的两句话

可以写成：

> 设计并实现一个程序化三维多智能体搜索基准，将城市几何生成、目标无关公共观测、私有评测器、执行回执、外部策略适配和版本化实验治理统一到可复现合同中。重点解决覆盖指标与真实几何观察之间的偏差，以及多智能体执行、通信和安全失败如何进入同一评测链。

> 当前已完成源码级合同、L0 运行时、公共图集、回放与审计工具以及原生能力前置检查；正式 L1 主结果仍需当前合同下的独立 ancestor、外部方法闭环、场景/许可证和清洁发布复核，不能把开发收据写成正式排名。

### 不建议使用的表述

- “已经完成真实城市四机 benchmark 排行榜”；
- “所有外部方法都已公平复现”；
- “L0 结果证明算法在真实飞行器上有效”；
- “视觉 review 视频就是正式实验”；
- “native gate 通过等于 formal score eligible”。

这些表述与当前 readiness 文件中的 `FORMAL_NO_GO`、历史合同失配和统计阻断相矛盾。

## 19. 结论

AeroCityBench 最值得学习的地方，是把一个看似简单的“三维多机搜索场景”拆成了可检查的研究对象：城市生成是否确定、公开输入是否干净、动作是否受控、观察是否真的合法、执行是否安全、结果能否重放、资产能否发布、统计是否独立、旧证据是否仍适用。源码中的 validator、hash、receipt、manifest 和 readiness 并不是外围装饰，而是 benchmark 结论可信度的一部分。

如果把它作为学习项目，应优先掌握三条主线：

1. `CitySpec -> public/private release`：理解场景、目标和信息边界如何分离；
2. `Observation -> Action -> ExecutionReceipt`：理解多智能体执行、通信、稳定观察和安全失败如何被记录；
3. `raw trace -> metrics -> governance`：理解一个结果怎样从测量轨迹变成带资格状态的统计量。

这三条线贯通后，再阅读强化学习包装器、外部三维方法和 Isaac/CF2X 适配器会更容易，也能清楚区分“代码已经具备的能力”和“仍需实验证据才能成立的论文主张”。
