# Rivermark Benchmark：项目技术说明

## 300 字项目概括

Rivermark 是面向多智能体三维搜索研究的基准基础设施，而不是单一演示脚本。项目围绕 Isaac Sim 中的八架 CF2X 四旋翼，建立从场景、协议、传感器采集到数据准入和评测的可追溯链路。采集侧同步记录 RGB、深度、语义标签、RayCaster LiDAR、IMU、机体状态、动作、接触/安全状态、相机外参和固定世界见证；每个文件、运行时配置、源代码和回执均以 SHA-256 绑定。CPU 路径可以在不启动 Isaac Sim 的情况下生成受控夹具、检查 schema、执行重复性和泄漏审计；原生 Isaac 路径负责真实 PhysX 采集。评测器将搜索事件、可见性、轨迹和安全约束映射到统一 ABI，并为经典规划、RL/MARL、离线 RL、QD 和 VLA 保留适配入口。该项目体现的核心能力是实验治理：区分可复现实验、数据集准入、仿真证据与正式 benchmark 资格，避免用一段视频或一个指标替代完整证据链。

## 项目定位

- **任务**：程序化 City-Lite 场景中的多 UAV 三维 stealth-search / Search3D。
- **主执行对象**：八架 PhysX CF2X；项目源码不重新分发 NVIDIA Isaac Sim、CF2X USD 或私有 evaluator 数据。
- **设计目标**：可重跑、可审计、可比较、可向多种算法范式提供稳定输入输出接口。
- **核心边界**：CPU 测试验证合同和数据治理；Isaac Sim 才能提供原生物理采集证据，二者不混写。

## 技术架构

```text
protocol/runtime lock + seed
        -> City-Lite scene and CF2X fleet
        -> synchronized sensor/action capture
        -> manifest, receipt and SHA-256 binding
        -> admission / leakage / provenance audits
        -> evaluator and baseline ABI
        -> RLDS / Zarr / Parquet projections and reports
```

`code/src/rivermark_benchmark/` 是可安装的核心包。采集、验证、评测、重复性、资产供应链和基线适配分别由独立模块承担；`config/` 保存采集协议、运行时锁和标签本体，`schemas/` 以 JSON Schema 约束每类产物，`tests/` 提供不依赖 Isaac Sim 的 CPU 回归。`media/` 和 `evidence/` 是展示和证据入口，不应被误读为全部训练数据。

## 核心代码结构

| 路径 | 作用 |
| --- | --- |
| `code/src/rivermark_benchmark/isaac_capture.py` | Isaac Sim 原生采集流程、传感器绑定、episode 写出 |
| `code/src/rivermark_benchmark/reproducibility.py` | 协议、运行时、源树和回执的哈希绑定及重复性分析 |
| `code/src/rivermark_benchmark/researcher_entry.py` | 不启动 Isaac 的研究者 smoke 入口 |
| `code/src/rivermark_benchmark/evaluator.py`、`search_event_evaluator.py` | 统一提交接口、搜索事件和指标评测 |
| `code/src/rivermark_benchmark/metrics.py` | 轨迹、搜索覆盖和安全指标计算 |
| `code/src/rivermark_benchmark/baseline_harness.py`、`methods.py`、`marl.py`、`qd_train.py` | 不同算法范式的适配和基线执行 |
| `code/config/`、`code/schemas/` | 协议锁、标签本体、JSON Schema 和供应链合同 |
| `code/tests/` | schema、路径、哈希、泄漏、资产和 evaluator 的 CPU 测试 |

关键设计是“先写证据再评分”。采集程序生成 manifest 和 receipt；准入程序检查未绑定文件、过期回执、跨 split 泄漏和非法 lineage；只有通过这些检查的 episode 才进入 evaluator。重复性报告使用相同 episode seed 的两次独立采集，并按 camera-local、frame-aligned、class/agent ID 比较，结论是 bounded same-seed variation，而不是 bitwise determinism。

## 开发过程

1. 先固定运行时 profile、CF2X 标定、场景协议和 seed，避免“脚本版本不同”成为隐性变量。
2. 再定义 schema、manifest 和 receipt，让采集、验证、评测共享同一数据契约。
3. 先实现 CPU researcher smoke 和结构测试，再接入 Isaac Sim 原生采集，缩短迭代反馈。
4. 增加崩溃遗留恢复、失败 ledger、资产来源和 SBOM 检查，支撑长时间批量收集。
5. 最后提供 RLDS/Zarr/Parquet 投影和多范式 baseline harness，使算法比较不依赖私有采集脚本。

## 涉及知识点

- Isaac Sim / PhysX 场景与传感器同步
- 多智能体轨迹、动作和观测 ABI
- JSON Schema、数据血缘、SHA-256 内容寻址
- 数据集 split leakage、可复现实验和 provenance
- Search3D 可见性、空间覆盖、轨迹与安全指标
- RL/MARL、QD、离线 RL、VLA 的统一适配
- SBOM、资产许可证和公开/私有边界治理

## 测试与验证

CPU smoke 和测试：

```powershell
cd code
python -m pip install -e ".[cpu-ci]"
$output = Join-Path $env:TEMP "rivermark-researcher-smoke"
python -m rivermark_benchmark.researcher_entry $output
python -m unittest discover -s tests -v
```

`evidence/same-seed-repeatability*.json` 是已保留的受控重复性证据。它不能替代原生 Isaac 物理正确性、真实飞行安全或正式数据集发布审查；这些结论需要绑定对应运行时、资产和实验记录。

## 学术申请中的项目价值

该项目适合展示“研究基础设施设计”能力：不仅能写仿真脚本，还能把实验变量、数据质量、统计口径和公开边界固化为机器可检验合同。对机器人、强化学习、具身智能和实验方法学方向，最有说服力的亮点是跨算法 ABI、同 seed 重复性分析和数据准入机制，而不是单独宣传八架无人机的视觉效果。

## 运行入口

- 研究者 smoke：`code/src/rivermark_benchmark/researcher_entry.py`
- Isaac 采集：`code/src/rivermark_benchmark/isaac_capture.py`
- 重复性分析：`code/src/rivermark_benchmark/repeatability.py`
- 评测：`code/src/rivermark_benchmark/evaluator.py`

## 证据边界

README 中的 Isaac/CF2X 描述、媒体和 evidence 文件各自对应不同证据等级。CPU 测试只能证明接口、schema 和审计逻辑；媒体只能证明渲染路径；正式 benchmark 结论还需要 evaluator-private 分区、完整运行锁和通过准入的原生 episode。第三方资产和 NVIDIA 内容遵循各自许可，不能从源码仓库的 Apache-2.0 直接推导出可再分发权。

## 源码审计记录

以下记录按源码职责而不是按文件数量组织。阅读时优先追踪一条 episode 从配置进入、传感器写出、receipt 生成、准入检查到评测报告的路径。

| 审计层 | 代表文件/目录 | 逐文件要点 |
| --- | --- | --- |
| 采集 | `code/src/rivermark_benchmark/isaac_capture.py` | 创建场景和车辆、绑定传感器、采样 action/observation、写 episode 文件和回执；任何超时或进程异常都必须留下失败信息。 |
| 公共入口 | `researcher_entry.py`、`__main__.py` | researcher smoke 不启动 Isaac；CLI 负责把路径、协议和输出目录显式化。 |
| ABI | `observation.py`、`action.py`、`episode.py` | 统一字段、shape、时间语义和版本，供 planner、RL、QD、VLA 适配。 |
| 复现 | `reproducibility.py`、`repeatability.py` | 对 protocol/runtime/source/receipt 做 SHA-256 绑定，并对同 seed 的两次结果做 camera-local、frame-aligned 比较。 |
| 准入 | `admission.py`、`provenance.py`、`lineage.py` | 拒绝未绑定文件、过期 receipt、split leakage、非法 lineage 和不完整 manifest。 |
| 评测 | `evaluator.py`、`search_event_evaluator.py`、`metrics.py` | 从提交 ABI 读取轨迹和事件，计算搜索覆盖、可见性、时序与安全指标并返回 schema 化报告。 |
| 基线 | `baseline_harness.py`、`methods.py`、`learned.py`、`marl.py`、`qd_train.py` | 让不同范式共享同一 episode/action/observation 约定，避免每个算法使用一套隐藏预处理。 |
| 供应链 | `asset_provenance.py`、SBOM 工具、`schemas/` | 记录 USD、第三方包和构建输入；许可证和内容来源与运行结果分离。 |
| 测试 | `code/tests/` | CPU 测试覆盖 schema、路径、manifest、哈希、故障恢复、泄漏、评测和公开投影。 |

## 端到端调用链复核

1. 采集 profile 读取固定 runtime lock、protocol、label ontology 和 seed，生成 scene/camera/vehicle binding。
2. 每个 control tick 同时推进物理状态和传感器采样；观测记录时间戳、agent ID、动作和外参版本。
3. episode writer 先写数据，再写 receipt/manifest；写入过程中异常时由 failure ledger 记录，不把半个 episode 标为成功。
4. admission 重新计算文件、协议、运行时和 source tree 哈希，并检查 split、lineage、文件集合和 schema。
5. 通过准入的 episode 投影到公开 ABI；evaluation 只读允许公开的字段，evaluator-private target 不进入算法输入。
6. evaluator 输出 metrics/report；报告中同时保留证据等级、runtime、seed、数据版本和失败样本统计。

## 学习复核顺序

建议按以下顺序复现，而不是先打开视频：先读 `schemas/` 理解字段；再读 `researcher_entry.py` 看 CPU 最小闭环；随后读 `reproducibility.py` 和 admission 模块理解哈希与失败语义；最后读 `isaac_capture.py`、evaluator 和 baseline harness。每一步都回答三个问题：输入是否公开、输出是否绑定、异常是否可追溯。

## 常见误读与纠正

- **“同 seed 一样”**：报告声明的是 bounded same-seed variation，不是逐 bit 相同。
- **“八机视频就是数据集”**：媒体是可视化证据，数据集还需要完整 manifest、schema、split 和准入回执。
- **“CPU smoke 验证了 Isaac”**：CPU 只验证合同和审计逻辑；PhysX、传感器同步和资产加载必须在 native runtime 验证。
- **“有 evaluator 就能公开目标”**：evaluator-private 字段故意不进入 public projection，泄漏检查是 benchmark 合同的一部分。

## 交付检查表

- [ ] protocol、runtime lock、source revision、asset registry 和 seed 已绑定。
- [ ] episode 文件集合与 schema 完整，异常路径有 failure ledger。
- [ ] 公开投影不含 evaluator-private target/witness。
- [ ] 重复性报告给出比较口径、失败样本和环境信息。
- [ ] baseline、RL、QD 或 VLA 结果沿用同一 ABI 和 evaluator。
- [ ] NVIDIA/第三方资产许可证与 Apache-2.0 源码边界分开说明。

## 原理与理论基础

### 1. Search3D 的部分可观测决策模型

多 UAV 搜索可以抽象为部分可观测马尔可夫决策过程（POMDP）：隐藏状态 `s_t` 包括城市几何、目标状态、车辆位姿和动态障碍，观测 `o_t` 由 RGB、深度、语义、LiDAR、IMU 和接触传感器组成，联合动作 `a_t` 由八架车辆的控制命令构成。策略根据历史 `h_t=(o_0,a_0,...,o_t)` 形成 belief `b_t(s)=P(s_t|h_t)`，再选择动作。Rivermark 的 ABI 和同步采集正是为这个定义服务：如果不同传感器的时间戳、agent ID 或外参不一致，观测就不再对应同一 belief，跨算法比较也失去意义。

### 2. 多传感器同步与坐标变换

传感器融合可以写成 `z_i(t_i)=h_i(x(t_i))+v_i`。当图像、LiDAR 和 IMU 的采样时刻不同，需要用状态传播或时间插值把它们映射到统一参考时刻；外参 `T_world_sensor` 决定传感器坐标中的点如何投影到固定世界。项目同时保存相机外参、body state 和 fixed-world witness，是为了让评测能够区分“模型没有看到目标”和“坐标/时间绑定错误”。接触/安全状态也不能只作为展示字段，因为它决定轨迹是否满足物理约束。

### 3. Provenance 的哈希链

数据准入的核心是内容寻址。对文件或 JSON canonical serialization 计算 `H(x)=SHA256(x)`，再把 protocol、runtime、source tree、asset registry、episode 文件和上一个 receipt 的 hash 写入当前 receipt，可以得到一条有向证据链。哈希不能证明物理仿真正确，但能证明“报告引用的输入没有在事后被静默替换”。这也是为什么项目把失败 ledger、过期 receipt、未绑定文件和 split leakage 视为一等错误，而不是普通 warning。

### 4. 搜索指标与统计边界

搜索任务的指标应来自事件和轨迹，而不是只看最终是否“找到”。可用覆盖面积/体积、首次发现时间、单位时间信息增益、航迹长度、能耗、碰撞和返回率组成向量 `m=(coverage, latency, cost, safety, closure)`。不同方法比较时必须锁定场景、seed、传感器、episode cap 和 evaluator-private split；否则看到的差异可能来自预算或数据泄漏。`same-seed repeatability` 只估计受控运行的变化分布，不等于证明模拟器逐 bit 确定。

### 5. 理论假设和失效模式

## 开发过程深拆

### 阶段 A：先固定合同，再写采集器

项目先确定 episode 的最小可审计单位：场景/运行时/源树版本、车辆集合、传感器集合、动作序列、时间范围、文件清单和失败状态。这个顺序避免采集器先写出大量无法解释的二进制数据，再事后猜测它们属于哪个配置。

### 阶段 B：CPU 路径隔离复杂依赖

`researcher_entry` 构造小型 fixture，验证 schema、manifest、公开数组和 report 生成。它不启动 Isaac Sim，因此可在 CI 或没有 GPU 的审阅环境运行。路径、JSON、哈希错误在 CPU 阶段暴露，传感器、PhysX 和资产错误留给 native 阶段。

### 阶段 C：原生采集与恢复

Isaac capture 需要处理 timeout、进程崩溃、半写文件和重复 episode。failure ledger 绑定协议、seed、stage 和最后一个有效 receipt；恢复程序只重启拥有的进程和未完成 cell，不能覆盖已通过准入的文件。

### 阶段 D：跨范式评测

经典 planner、RL/MARL、QD 和 VLA 的输入形态不同，但 evaluator 只接受稳定 observation/action ABI。适配器负责转换而不改变任务定义；任何额外传感器、未来帧或 private 字段都必须在 manifest 中声明。

## 源码阅读路线

推荐按 `schemas -> researcher_entry -> reproducibility -> admission -> isaac_capture -> evaluator -> baseline_harness` 阅读。每读完一层，检查输入是否有版本、输出是否有 hash、异常是否有失败语义、文件是否能被下一层验证。

该基准假设传感器消息可对齐、车辆状态可被仿真器真实推进、public ABI 不泄露 evaluator-private 信息。传感器延迟、进程崩溃、资产版本变化、目标定义改变或碰撞模型不一致都会破坏这些假设。工程上因此要求 capture、admission、evaluator 三层重复检查，并把“不能评分”作为合法输出，而不是用缺失字段填零。

## 阅读范围与证据等级

本说明以本地项目 `rivermark/` 为对象，重点核对了根目录说明、`docs/` 下的任务/采集/评测/复现文档、`code/config/` 中的运行时锁与采集协议、`code/schemas/` 的结构约束，以及 `code/src/rivermark_benchmark/` 的核心实现。源码阅读的重点不是把文件名重新抄一遍，而是沿着一条 episode 的生命周期追踪：配置如何进入运行时，车辆和传感器如何被创建，数据如何落盘，失败如何被登记，公开投影如何去除私有信息，最后评测器如何从事件得到指标。

文档中的“已实现”只表示在源码或配置中可以直接找到对应逻辑；“已有证据”表示仓库里存在相应 JSON、日志或媒体；“建议实验”表示下一步应运行的验证，不能写成已经完成。当前可以确定的证据等级如下：

| 证据等级 | 能说明什么 | 不能说明什么 |
| --- | --- | --- |
| 静态源码和配置 | 字段、流程、边界和默认值已经被写入工程 | 不能证明当前机器成功加载了所有资产 |
| CPU researcher smoke | ABI、schema、manifest、哈希和逻辑可在无 Isaac 环境下运行 | 不能证明 PhysX、渲染器或传感器物理正确 |
| `evidence/preflight.json` | 某次预检记录中的依赖、资产和运行锁检查通过 | 不能替代完整多机采集或正式数据集发布 |
| 媒体文件 | 某些录制或验证路径有可视化输出 | 不能仅凭视频判断目标真值未泄漏、统计有效或物理逼真 |
| native T1/T2 receipt | 绑定了场景、运行时、传感器和采集结果的原生 episode | 仍需要按协议完成 cohort、准入和统计分析 |

这种证据分层是项目研究方法的一部分。它使读者可以准确知道某个结论来自代码、来自一份运行回执，还是仅仅来自展示素材。

## 研究问题与非目标

### 研究问题

Rivermark 关注的问题可以写成：在一个具有遮挡、有限可见性、机间安全约束和通信限制的程序化三维城市中，如何比较不同多智能体搜索策略的发现效率、轨迹代价和安全性，同时保证不同策略看到的是同一类信息、使用的是同一时间预算，并且结果可以追溯到原始仿真输入？

这不是单纯的“八架飞行器能否飞起来”。真正的实验对象是一个由场景合同、传感器合同、动作合同、事件评测和数据治理共同定义的任务。若只改变策略而没有锁定场景资产、传感器参数、episode seed、重置条件和评测端点，最后的数值差异就无法归因于策略本身。

### 明确的非目标

1. 项目不是公开的真实飞行控制器，也不声称可以直接控制实体飞行器。
2. 项目不是把旧有路线、目标和轨迹迁移进新场景的可视化工具；原生采集明确拒绝导入外部历史工件。
3. 项目不是用 OpenCV 后处理生成“看起来像仿真”的视频。CPU 路径可以产生结构化夹具，但不冒充 Isaac 渲染或 PhysX 证据。
4. 项目不把 evaluator-private 的目标真值、奖励或未来状态提供给策略。
5. 项目不把某一次成功的 canary 运行写成已完成的大规模正式 benchmark。

这些非目标限制了项目的宣传边界，却提高了实验结论的可信度。

## 目录与职责的完整地图

### 根目录和文档层

- `README.md`：面向英文读者的入口，说明项目目的、依赖边界、运行方式和证据状态。
- `README.zh-CN.md`：中文说明，适合快速理解任务、采集分层和数据访问边界。
- `docs/overview.md`：把项目拆成场景、运行时、采集、准入和评测几个层次。
- `docs/task-and-scene.md`：解释 City-Lite 场景、路线族、起点锚点、目标区域和 holdout 关系。
- `docs/capture.md`：描述 native capture 的启动、传感器、写盘、收尾和失败处理。
- `docs/observation-abi.md`：规定 policy 看到的观测字段、单位、shape、时间戳和缺失语义。
- `docs/methods.md`：说明经典规划、学习方法、MARL、QD 等范式如何接入同一接口。
- `docs/scoring.md`：定义搜索事件、可见性、安全违规和汇总指标。
- `docs/validation-and-admission.md`：解释为什么一个 episode 必须经过结构、血缘、split 和内容准入。
- `docs/determinism.md`：给出运行锁、哈希、同 seed 重复性和环境记录的口径。
- `docs/governance.md`、`docs/data-access.md`、`docs/limitations.md`：分别处理公开边界、数据获取和局限性。

### 配置、schema 与供应链

- `code/config/isaac_runtime.windows-5.1.json` 锁定 Isaac Sim、IsaacLab、Python、GPU、时间步长、设备、相机开关和资产哈希。
- `code/config/isaac_repeatability.citylite_t1_v2.json` 锁定同 seed 重复性比较的对齐方式和阈值。其声明是有界变化，不是 bitwise determinism。
- `code/config/collection_protocol.citylite_t1_expert_coverage_v2.json` 用于 T1 专家覆盖采集；它要求训练/验证 holdout 和最低尝试数，但不用于 policy ranking。
- `code/config/collection_protocol.citylite_native_t2_canary_v3.json` 是开发阶段原生 T2 canary，要求运行时标定、完整传感器 smoke、action causality、同步和私有事件评估。
- `code/config/label_ontology.citylite_v1.json` 固定语义类别和标签 ID，避免同一颜色或类别名称在不同 episode 中含义变化。
- `code/config/supply_chain.pending.json`、`local_assets.example.json` 记录外部 USD、材质、私有路径和待补齐的来源信息。
- `code/schemas/` 中的 JSON Schema 将 manifest、receipt、episode、观测、动作和评测报告变成机器可检查的接口，而不是仅靠 README 约定。

### 源码层

`abi.py` 是跨模块的语义基础；`schema.py` 和 `validate.py` 负责结构检查；`runtime/` 提供不依赖 Isaac 的动力学和传感器夹具；`isaac_capture.py`、`isaac_dataset.py`、`isaac_evidence.py` 负责 native 采集和证据封装；`eight_cf2x_fleet.py` 处理八机实例；`citylite_scene/` 和 `citylite_task.py` 负责场景权威和任务状态；`collection_protocol/` 把 T1、T2、seed、功耗和覆盖要求变成代码；`admission` 相关实现检查数据是否可以进入评分；`search_event_evaluator.py` 和 `metrics.py` 只在准入后的数据上计算任务指标；projection 模块为不同研究框架导出受控视图。

## City-Lite 场景为什么要有 authority

### 场景不是一个目录

在普通仿真脚本中，找到 USD 文件并成功打开似乎就足够了。但 benchmark 的场景必须回答更多问题：几何层来自哪个版本？碰撞代理和可视网格是否一致？路线和起点是否属于同一 split？目标区域是否在 policy 可见信息之外？资产是否有可公开的许可证？因此 `citylite_scene/authority.py` 不把“路径存在”当作场景有效，而是同时检查层文件、资产 registry、版本、receipt、哈希和几何证明。

`scene.py` 负责在 authority 允许之后构造场景，`constants.py` 固定坐标和类别语义，`aabb.py` 提供包围盒几何，`materials.py` 处理可视材质，`routes.py` 保存路线族与起点/目标区域关系。职责分开以后，改变材质不会悄悄改变碰撞几何，改变一条路线也不会让旧 episode 继续声称属于原 split。

### 路线族和 holdout

路线不是随机散点。一个 episode 至少需要携带 route family、start anchor、target region、episode seed 和 split。训练与验证可以在相同场景风格下使用不同起点锚点或目标区域；更严格的实验还可以留出整条路线族。这样做的原因是：如果只随机打乱帧，策略可能记住相同的空间结构，得到很高的帧级分数却没有真正泛化。

建议的几何审计包括：检查起点与目标区域是否重复、检查验证路线是否穿过训练轨迹的近邻半径、检查目标可见性分布是否相同、检查每个 cohort 的距离和遮挡难度是否平衡。`cohort_audit.py` 的作用不是给算法加分，而是防止数据划分让某一组天然更容易。

## Search3D 任务的形式化

设世界状态为

```text
s_t = (G, X_t, O_t, Y, C_t, B_t)
```

其中 `G` 是固定城市几何，`X_t` 是八架飞行器的位姿和速度，`O_t` 是动态障碍或接触状态，`Y` 是 evaluator-private 目标集合，`C_t` 是通信和连接状态，`B_t` 是各智能体对已搜索空间的内部 belief。策略只能通过观测历史更新 `B_t`，不能直接读 `Y`。

对第 `i` 架飞行器，公开观测可以表示为：

```text
o_t^i = (rgb_t^i, depth_t^i, semantic_t^i, lidar_t^i,
         imu_t^i, body_state_t^i, contact_t^i, message_t^i)
```

联合动作是八个个体动作的集合，动作的有效时间不是命令写入时间，而是仿真器应用并完成一个 step 后的 effective timestamp。这个因果关系由 ABI 记录为“command write -> simulation step -> state update -> sensor read -> storage”。若把命令写入时间误当成生效时间，轨迹延迟和控制稳定性会被系统性高估。

目标发现也应被定义为事件，而不是一个随意的布尔标签。一个候选事件至少包含 agent、时间、空间位置、类别或匿名目标 ID、观测证据引用和置信度；private evaluator 再根据真值、空间容差、时间顺序和去重规则决定它是否为 confirmed event。这样可以区分“看到了疑似目标”“重复报告同一目标”和“在禁止区域内得到的无效确认”。

## 原生 Isaac 采集：从启动到收尾

### 1. 启动前检查

`preflight.py` 和相关 runtime safety 模块在启动前检查运行锁、GPU、Isaac/IsaacLab 版本、场景 authority、资产哈希、输出目录归属、磁盘和 Windows system commit。采集进程还要确认没有外部 native 进程占用同一资源，避免上一轮崩溃留下的仿真器继续写入当前目录。

### 2. 创建应用与 stage

采集器使用一个 `AppLauncher`，并根据 runtime lock 设置 `headless=true`、`enable_cameras=true`、`device=cuda:0`。启动后新建 stage，再由 City-Lite authority 绑定允许的几何层和碰撞代理。采集器拒绝把旧 MD-QD-Swarm 的 routes、targets、traces 或 artifacts 直接带入当前 episode，这个限制是为了确保任务定义来自当前 protocol，而不是来自展示脚本的隐式状态。

### 3. 构建八机 fleet

`eight_cf2x_fleet.py` 为八个 agent 分配稳定的 public agent ID，实例化 CF2X PhysX 资产，写入初始位姿、碰撞和控制参数。每个 agent 的 ID 必须跨帧稳定，不能按当前可见性或渲染顺序重新编号。fleet 初始化完成后，采集器还要做静态起点分离和禁飞区检查，防止 episode 一开始就产生碰撞或越界。

### 4. 绑定传感器

正式采集至少包含以下传感器和状态：

| 传感器/状态 | 用途 | 关键审计点 |
| --- | --- | --- |
| onboard Camera/TiledCamera | 策略可用的 RGB、深度或语义观测 | 相机相对机体位姿、内参、曝光和时间戳 |
| overview Camera/TiledCamera | 固定世界坐标的可视化与见证 | 固定相机位姿，不能跟随机体或按状态切换 |
| MultiMeshRayCaster LiDAR | 几何感知和障碍距离 | 射线配置、坐标系、点数、缺失值和噪声模式 |
| IMU | 角速度、线加速度和状态估计输入 | body frame、单位、采样时间和 bias 配置 |
| ContactSensor | 碰撞、接触和安全状态 | 碰撞事件与物理 step 的对应关系 |
| body state | 位姿、速度、姿态和控制结果 | world/body frame、四元数顺序和有效时间 |

机载相机的 body offset 记录为 `(0.12, 0.0, 0.04)`，下俯角为 15 度；overview 相机使用固定世界坐标和 1920×1080、45 mm 配置。overview 不能作为 policy 输入，否则会把一个本来不具备的全局上帝视角混入算法比较。

### 5. 控制与采样循环

runtime lock 中的 `dt_s=0.005` 意味着物理推进和动作因果链以 5 ms 为基础步长；`render_interval=1` 表示每个步长都推进渲染采样配置，保留帧时再按协议做 stride。一个可靠的循环应该先读取当前状态和待执行动作，写入 action record，再调用仿真 step，等待状态更新，读取各传感器，最后把统一时间戳和 frame index 写入 archive。不能先写“预测状态”，再把它当成传感器观测。

### 6. 写出和收尾

项目保留帧 stride 为 10，以控制存储预算；这不会改变物理推进频率。写出过程中同步更新 frame archive、episode manifest 和 checkpoint。若进程在中途崩溃，failure ledger 记录最后一个有效 frame、最近的 receipt、异常类型和输出目录；收尾阶段只有在文件集合、schema、哈希和事件统计都完成后才生成成功 receipt。半写 episode 不允许被下游脚本自动发现为正式样本。

## 传感器同步、坐标和 ABI

### 坐标约定

项目使用右手坐标系，世界坐标为 `x_east_y_north_z_up`，机体坐标为 FLU，相机采用 OpenCV optical frame，长度用米、角度用弧度，四元数按 `wxyz` 存储，变换语义为 `T_parent_child`。这些细节看似是格式问题，实际上直接决定点云投影、碰撞距离和轨迹误差的正负号。

对于一条点云观测，工程上需要明确它究竟是 `p_world`、`p_body` 还是 `p_lidar`。若使用

```text
p_parent = T_parent_child * p_child
```

却把 `T_child_parent` 当成同一对象，城市墙体会在地图中出现镜像或漂移。Rivermark 因此同时保存传感器外参、body state 和固定世界 overview witness，使错误可以通过几何闭环被发现，而不是等训练失败后猜原因。

### 时间语义

每条记录至少需要区分 `command_timestamp`、`effective_timestamp`、`sensor_timestamp` 和 `storage_timestamp`。它们不一定相同，但必须满足因果顺序，并在 manifest 中声明时钟来源。多传感器对齐时，不能只按数组下标拼接；应按 frame index 和时间容差检查，必要时使用状态传播或插值。`policy_projection.py` 会拒绝 command timestamp 晚于 effective timestamp、数组长度不一致或字段时间戳无法对齐的投影。

### 缺失数据语义

缺失值不是简单填零。零深度可能代表真实的近距离表面，也可能代表传感器无回波；零点云可能代表无障碍，也可能代表采集失败。因此 ABI 要区分 valid mask、missing reason 和值本身。压缩方式、dtype、shape、单位、frame、缺失策略都应随字段描述一起保存，使下游算法不会把“无数据”当成“世界为空”。

## CPU pilot 与原生 Isaac 的边界

`runtime/engine.py` 的 profile 名称是 `rivermark-kinematic-pilot-v1`。它不启动 Isaac Sim，也不模拟完整 PhysX，而是用固定速度/yaw controller 推进状态，再生成结构化 RGB-D、语义、LiDAR、雷达、IMU、body state 和通信数据。它的价值是让研究者在没有 GPU 或 Isaac 安装的机器上验证：

1. action 是否能按 ABI 被接受；
2. 状态是否单调推进且时间戳合理；
3. public observation 是否不含目标 ID、奖励、未来状态和 seed；
4. 安全 guard 是否阻止明显的碰撞、越界和机间过近；
5. manifest、projection、evaluator 和失败状态是否可串联。

它不能证明渲染器、碰撞网格、CF2X 电机模型、IMU 噪声或 LiDAR 射线与 Isaac native 一致。论文或申请材料应把它称为 CPU contract/pipeline pilot，不能称为“在 Isaac 中完成了八机实验”。要获得原生证据，必须运行同一 protocol 的 `isaac_capture.py`，留下 native receipt、传感器 smoke 和可审计的 episode 文件。

## 信息隔离：为什么目标真值必须在 evaluator 私有侧

Search3D 的难点来自不完整观测。如果 policy observation 直接包含目标坐标、目标 ID、完整地图、评测奖励或 future state，任务就从搜索变成了读取答案。`search_event_evaluator.py` 的设计把目标真值保留在 evaluator-private manifest 中，公开提交只包含动作、观测引用和候选事件。

这不是为了增加形式上的复杂度，而是为了防止三类常见泄漏：

- **直接泄漏**：字段名或数组中出现 target position、target ID、reward。
- **间接泄漏**：以“调试可视化”名义输出只在目标附近出现的颜色、框或额外通道。
- **时序泄漏**：把未来帧、最终地图或 episode 结束状态拼入当前观测。

`policy_projection.py` 用 allow-list 从原始 capture 生成外部策略视图，原始数据不被复制覆盖；输出 manifest 明确标记 development-only、formal benchmark admission false 和 T2 score not permitted。这个 projection 是 provenance boundary，不是操作系统级沙箱；如果研究者自行读取原始 capture，工程层面无法替代访问控制，所以数据访问文档和私有 evaluator 仍然是必要的。

## 事件评测和指标设计

### 事件匹配

评测器接收候选事件、观测证据和私有目标真值。匹配通常需要同时满足类别/匿名身份、空间距离、时间窗口和未被确认过的唯一性。匹配失败不应简单归因于策略低分，报告应区分：没有观测证据、证据超出空间容差、重复确认、时间顺序错误和安全违规。

### 主指标

`metrics.py` 的主指标是 `normalized_confirmed_auc`：以时间为横轴、confirmed target recall 为纵轴，计算曲线归一化面积。它比只报告 episode 末尾的 recall 更能区分“很早找到目标”和“直到结束才找到目标”。同时保留 final recall、false confirmations、time to all targets 和 truncation。

设 episode 在时间 `T` 内有 `K` 个目标，确认召回为 `R(t)`，则可以写成：

```text
NC-AUC = (1 / T) * integral_0^T R(t) dt
```

离散实现应使用实际事件时间或 frame timestamp，而不是默认每帧间隔完全相同。不同 episode 的统计单位是 episode；八架 agent 和数千帧是同一 episode 内的重复观测，不能当成独立样本直接计算显著性。

### 安全指标

安全报告至少包含 collision、geofence、visual_intrusion、最小机间距离、越界持续时间和是否安全收尾。安全指标不是 reward 的附属项，而是任务约束的一部分。一个策略如果通过穿越禁区获得更高 AUC，应该被 evaluator 标为不合格或单独分组，而不是与安全策略混在一个平均分中。

## 数据血缘、manifest 与失败账本

### Manifest 的作用

episode manifest 是数据的目录，而不是简单的文件列表。它应包含 protocol ID、runtime lock ID、source revision、scene contract、asset registry、episode seed、split、route family、agent 列表、传感器列表、时间范围、文件 hash、schema version、privacy tier 和 completion status。它回答“这条数据是什么、由什么产生、谁可以读取、是否可以评分”。

### Receipt 与 SHA-256

对 canonical JSON、配置文件、源树摘要、资产和 episode 文件计算 SHA-256，再把这些摘要写入 receipt，能够检测事后替换或遗漏。哈希不是物理正确性的证明，也不是许可证证明；它只证明被引用的内容与报告生成时的内容一致。许可证应由 `asset_provenance.py`、`NOTICE`、`LICENSE` 和 supply-chain registry 单独说明。

### Failure ledger

失败记录需要保留：run ID、protocol、seed、stage、最后有效 frame、进程退出码、异常类型、已写文件、最近 checkpoint、资源状态和恢复建议。最危险的做法是捕获异常后删除输出目录，让批量统计只看见成功样本。Rivermark 的失败 ledger 使失败率本身成为可以分析的指标，也让重跑时知道应该从头开始还是从未完成 cell 恢复。

## 准入、split 泄漏和 cohort 审计

一个 episode 只有通过以下检查，才可进入正式评测：

1. manifest 的 schema 和文件集合完整；
2. receipt 中的 protocol、runtime、source 和 asset hash 可重新计算；
3. 公开 projection 不含 private target、reward 或 future state；
4. route family、start anchor、target region 与 split 规则一致；
5. 时间戳、shape、dtype、frame 和 agent ID 在全 episode 内一致；
6. 采集未被 crash、磁盘不足或资源保护截断；
7. evaluator 能够明确判断完成、失败或不可评分。

split leakage 不限于完全相同的文件。还应审计空间近邻、相同路线的不同 seed、同一目标区域的重复视角、相同轨迹前缀和从 train 生成 validation 的缓存。`cohort_audit.py` 负责把这些检查转换为可重复的报告；它不替算法选择最优数据，也不自动把不平衡 cohort 修正成“公平”。

## T1、T2 与正式数据集的实验设计

### T1：专家覆盖和传感器验证

`collection_protocol.citylite_t1_expert_coverage_v2.json` 是 T1。它使用八机 multisensor expert coverage，设置 train/validation、route family、start anchor 和 target region holdout，要求最低尝试数和最低 admitted 数。T1 的主要产物是 sensor quality、route coverage、visibility realization、failure rate 和 episode bytes，目的在于证明场景和采集链条可用，而不是给学习策略排名。

建议的 T1 表格按 episode 记录：

| 组 | 路线族 | 起点锚点 | 目标区域 | 尝试数 | 准入数 | 传感器缺失率 | 安全违规率 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| train | A | seen | seen | 4+ | 4+ | 统计 | 统计 |
| validation | B | held out | held out | 4+ | 4+ | 统计 | 统计 |

### T2：原生 policy canary

`collection_protocol.citylite_native_t2_canary_v3.json` 是 development-only native T2 canary。它的检查顺序应是：CF2X runtime calibration、full sensor smoke、独立两次通过、action causality、sensor sync、camera pose closure、fixed-world witness、private event evaluation、native replay 和 route timing feasibility。任何一项未通过，都只能输出 canary failure 或 not-admitted，而不是补一个零值继续算正式分数。

### 正式数据集

正式数据集需要在 T1/T2 的基础上增加固定 cohort、数据访问权限、公开投影、统计计划、版本冻结和发布前审计。应为每个正式版本保留 release manifest、源树 revision、资产许可范围、schema 版本、失败清单摘要和可复现命令。没有这些材料时，最准确的说法是“具备正式数据集的工程骨架”或“正在进行 native admission”，而不是“已发布 benchmark”。

## 重复性实验怎么做

重复性配置将比较同一 protocol、同一 episode seed、同一相机坐标系和 frame 对齐下的两次独立运行。建议流程如下：

1. 固定 `isaac_runtime.windows-5.1.json`、City-Lite contract、CF2X asset hash 和 seed。
2. 分别建立两个全新输出目录，禁止第二次读取第一次的临时状态。
3. 记录 GPU、驱动、Isaac Sim、IsaacLab、Python、CUDA、源树和资产哈希。
4. 用 public agent ID 和 frame index 对齐状态、IMU、LiDAR、RGB、深度和语义标签。
5. 按 `isaac_repeatability.citylite_t1_v2.json` 中的 position、orientation、velocity、LiDAR、RGB、depth 和 semantic agreement 阈值统计。
6. 同时报告失败、缺失帧、超阈值样本和资源告警。

已有配置中的示例阈值包括 position 0.05 m、orientation 0.02 rad、linear velocity 0.15 m/s、angular velocity 0.3 rad/s、LiDAR mean 0.05 m、LiDAR p95 0.25 m、RGB mean 8、RGB p95 32、depth mean 0.1 m、depth p95 0.5 m、semantic agreement 0.98。它们是本项目的受控比较标准，不是对所有机器和所有场景的普适物理误差保证。最终报告必须使用“bounded same-seed repeatability”措辞，并保留变异原因。

## 面向不同算法范式的接入

### 经典规划

经典 planner 只需要读取几何、可见性、机体状态和通信约束，通过 action ABI 写回控制命令。它适合作为可解释 baseline，帮助验证指标、路线难度和安全规则没有被学习器特化。

### RL/MARL

RL 或 MARL 训练时可以通过 `train.py`、`learned.py`、`marl.py` 和 SB3 相关入口接入。训练环境应从公开 projection 获得观测，奖励只能在训练实验内部定义，不能被带入评测输入。集中训练、分散执行需要在 manifest 中声明：训练时是否可见全局状态，执行时每个 agent 只保留哪些字段，通信是否有延迟或丢包。

### 质量多样性和离线学习

`qd_train.py` 适合研究不同行为描述符下的多样性档案；档案更新要引用真实执行后的指标，而不是只根据策略网络输出预测“可能达到”的质量。离线 RL 则使用已准入 episode，必须防止把 evaluator-private 结果、未来帧或失败后状态作为 observation。每一种投影都应生成新的 manifest，不能修改原始 capture 来迎合算法。

### VLA 和跨模态方法

RGB、深度、语义、LiDAR 和通信可以投影到研究者自己的数据格式，但需要保留原始 frame index、timestamp、agent ID 和 episode lineage。若模型额外使用语言指令或文本目标，应在 protocol 中显式声明指令来源和是否包含目标类别先验，避免不同工作在“输入信息量”上不可比。

## 建议的复现实验步骤

### 阶段一：无 GPU 的合同复现

```powershell
cd C:\Users\Administrator\Desktop\作品集\MultiAgent-Robotics-Portfolio\rivermark
python -m pip install -e ".[cpu-ci]"
$out = Join-Path $env:TEMP "rivermark-researcher-smoke"
python -m rivermark_benchmark.researcher_entry $out
python -m unittest discover -s tests -v
```

先确认输出目录中的 manifest、schema、projection、failure ledger 和 report，再看是否有测试失败。这个阶段应检查 public observation 是否缺少 target/private 字段、动作时间是否满足因果关系、哈希是否能复算。

### 阶段二：小规模 CPU 方法对照

选择固定的 fixture、固定 episode cap 和固定 agent 数，运行一个规则 planner、一个随机或贪心策略和一个最小学习 baseline。此处目的不是得到论文分数，而是检查 evaluator 是否能区分早发现、晚发现、重复确认、碰撞和截断。所有结果按 episode 汇总，并保存配置、源树摘要和输出 manifest。

### 阶段三：native preflight 与传感器 smoke

在满足 Isaac Sim 5.1.0.0、IsaacLab 2.3.x、CUDA 和 GPU 条件的 Windows 机器上执行 preflight。确认 City-Lite contract 和 CF2X USD 的 SHA-256 与 runtime lock 一致，再启动一个短 T2 canary。重点查看 camera pose closure、LiDAR shape、IMU frame、contact event 和 fixed-world overview；不能只观察窗口是否出现画面。

### 阶段四：T1 coverage 和 T2 canary

按 protocol 生成新 run ID，分别记录 train/validation holdout，不复用上一轮的临时输出。每次运行完成后先做 admission，再做 evaluator。对于失败 episode，保留 failure ledger 并在 cohort 统计中报告；不要为了达到 minimum admitted 而删除失败样本。

### 阶段五：冻结版本和方法比较

待场景、传感器和准入稳定后冻结 release manifest，再比较 planner、RL/MARL、QD 或 VLA。每个方法使用同一 policy projection、episode cap、seed 集合和安全规则；若某方法需要额外信息，另开实验组，不得混入主表。

## 本次本机核验记录

本次只修改了本说明文档，没有上传 GitHub，也没有修改 Rivermark 源码。使用现有环境运行 `python -m unittest discover -s tests -v`，全量测试共发现 411 项：377 项通过，21 项因 `jsonschema`、PyTorch、Parquet、Zarr 或 QD 等可选依赖未安装而跳过，13 项在 formal dataset 和 supply-chain 相关测试中因缺少 `cryptography` 模块而无法执行。随后针对项目核心合同单独运行了 runtime、metrics、ABI 和 policy projection 四组测试，共 32 项，全部通过。

这组结果说明当前源码的 CPU 合同、指标、观测 ABI 和公开投影路径可以在本机复核；它不说明 Isaac Sim、PhysX、CUDA、CF2X USD 和 native multisensor capture 已在本机完成验证。要使 native 结论成立，还需要在满足 runtime lock 的 Isaac 环境中补齐依赖，执行 preflight、传感器 smoke、T1/T2 canary、receipt 校验和 cohort admission，并保存对应的原始运行证据。

## 故障排查思路

### 采集启动即失败

先检查 runtime lock、Isaac 版本、GPU 驱动、资产哈希和输出目录 ownership，再检查是否有遗留 native 进程。不要先修改任务或关闭安全检查。若 authority 报错，优先修复 scene contract 或 registry，而不是把路径硬编码成另一个 USD。

### 有 RGB 但没有有效深度或 LiDAR

检查传感器是否在 AppLauncher 中启用、是否真正 attach 到对应 prim、时间戳是否在同一 step 写出、数组 shape/dtype 是否符合 ABI。overview 画面正常并不代表 onboard sensor 已经有效。

### 相机见证与机体轨迹不闭合

检查 body offset、相机坐标系、四元数顺序和 `T_parent_child` 方向。固定 overview 不能跟随机体；若它的轨迹随 agent 状态改变，说明可视化证据已经污染了固定世界假设。

### 评分结果为不可评分

查看 evaluator 报告中的 private event、时间顺序、候选事件匹配、重复确认、安全违规和 truncation 字段。不要把 `not_scored` 当作失败分数，也不要用缺失值填零后重新平均；先修复证据或明确排除原因。

### 同 seed 差异过大

先按 frame 和 public agent ID 对齐，再检查传感器时间、GPU/驱动、资产版本、reset 状态和资源压力。报告应保留差异分布，不能只保留通过阈值的 episode。若差异来自渲染非确定性，应在结论中说明它属于有界变化，而不是宣称完全确定。

## 适合论文和申请材料的表述

可以这样描述项目：

> 我实现了一个面向多智能体三维搜索的实验基础设施，将程序化场景、八机多传感器采集、公开观测 ABI、私有事件评测和数据准入统一到可追溯的 episode 协议中。系统同时提供不依赖 Isaac 的 CPU contract smoke 和原生 Isaac/PhysX 采集路径，并用运行锁、资产哈希、失败账本和同 seed 有界重复性分析控制复现实验风险。

如果 native 数据集尚未完成，应写成“已实现采集和准入框架，正在完成原生 T1/T2 验证”，不要写“已完成真实飞行验证”或“已发布正式 benchmark”。如果只运行过 CPU smoke，应明确说“验证了 ABI、schema 和审计逻辑”。这种措辞看起来比夸张的结果保守，但对导师或审稿人更有价值，因为他们能够分辨系统设计、工程实现和实验结论各自的证据。

## 术语表

| 术语 | 本项目中的含义 |
| --- | --- |
| City-Lite | 项目维护的程序化三维城市场景及其版本、几何和资产合同 |
| episode | 一次具有固定 protocol、seed、场景、agent 集合和时间范围的实验单元 |
| T1 | 面向专家覆盖、传感器质量和路线覆盖的采集阶段，不用于 policy ranking |
| T2 canary | 原生 Isaac policy 路径的开发验证，不等同正式 benchmark |
| public observation | 策略可以访问的观测字段集合 |
| evaluator-private | 只供评测器使用的目标真值、安全真值或其他隐藏字段 |
| receipt | 对运行输入、输出、哈希和状态的可验证回执 |
| provenance | 数据从资产、配置、源树到 episode 和报告的血缘记录 |
| policy projection | 从原始 capture 生成的 allow-list 研究者视图 |
| fixed-world witness | 固定世界位姿的 overview 见证，用于检查轨迹和空间闭环 |
| bounded repeatability | 同 seed 运行在明确阈值下的有界变化，不是逐 bit 确定性 |
| split leakage | 训练/验证之间通过路线、空间、缓存或未来信息发生的隐性泄漏 |

## 最终理解

Rivermark 最值得学习的地方，不是“使用了多少传感器”或“能渲染多少架飞行器”，而是把一个容易被演示视频掩盖的研究问题拆成了可以逐项验证的合同：场景必须可追溯，动作必须有因果时间，观测必须信息隔离，失败必须被保留，评测必须基于事件和 episode，结果必须能回到原始配置和资产。它把仿真研究从一次性脚本提升为可复核的实验系统。

因此，阅读或继续开发时应始终坚持三条线并行：

1. **物理线**：Isaac/PhysX、CF2X、传感器、碰撞和时间步是否真实推进。
2. **数据线**：schema、manifest、receipt、hash、split 和 projection 是否闭合。
3. **研究线**：任务定义、基线、公平比较、指标和统计是否回答明确问题。

只有三条线都通过，Rivermark 才能从“有代码、有画面”变成可用于论文实验和学术申请的研究项目。
