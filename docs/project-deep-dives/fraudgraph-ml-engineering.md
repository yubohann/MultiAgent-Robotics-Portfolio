# FraudGraph ML Engineering：项目技术说明

## 300 字项目概括

FraudGraph ML Engineering 是一个面向金融交易欺诈检测的可复现实验工程。项目没有把“图神经网络”当成单个模型文件，而是把同一数据集构造成图结构、关系序列和事件序列三种视图，再由 SplitGNN 图编码器、序列 Transformer 和融合分类头共同学习。数据层提供 IEEE-CIS、Elliptic、AMLSim、信用卡、DeFi、Ethereum phishing/Ponzi 等适配器；训练层支持 CPU smoke 和 CUDA 12.1 profile；运行层保存 seed、配置、数据修订、验证选模规则、held-out test 约束和 Git revision 到 manifest；评估层输出分类指标、阈值、embedding 分析、TensorBoard 审计和可比较报告。仓库还保留 vendored SplitGNN 代码及第三方归属，数据、图缓存、checkpoint 和日志默认留在本地，不通过代码静默下载。实验协议围绕三个可证伪问题展开：图编码器是否捕获序列模型遗漏的欺诈信号，关系/事件序列是否带来稳定增益，以及低标注比例下哪些模块仍然有效。它适合展示图学习、时序建模和实验治理的交叉能力，而非宣称某个数据集上的统一 SOTA。

## 项目定位

- **任务**：多数据集金融欺诈/恶意账户分类。
- **建模视图**：交易图、关系序列、事件序列和融合表示。
- **核心模型**：SplitGNN encoder + Transformer/sequence branch + hybrid fusion head。
- **工程原则**：数据由用户合法放置；代码不自动下载，测试与训练输出隔离。

## 技术架构

```text
raw dataset -> dataset registry / adapter
            -> graph cache + relation/event sequence views
            -> SplitGNN branch + Transformer branch
            -> hybrid fusion / low-label variants
            -> validation checkpoint and threshold selection
            -> held-out evaluation, embeddings, reports, manifest
```

`runtime_dataset_policy.py`、`paths.py` 和 dataset adapters 负责路径及数据版本；`hybrid_task_model/` 将编码、序列、loss、SSL 和 forward 拆成 mixin；`trainer_local.py`、`main.py` 和 `paper_runner_runtime.py` 负责训练运行；`evaluator.py` 和 `run_artifacts.py` 负责结果与证据落盘。

## 核心代码结构

| 路径 | 作用 |
| --- | --- |
| `src/fraud_ml_engineering/main.py`、`__main__.py` | 训练 CLI、数据集选择和运行参数 |
| `src/fraud_ml_engineering/dataset_registry.py` | 数据集名称、适配器和配置注册 |
| `src/fraud_ml_engineering/*_dataset.py` | IEEE-CIS、Elliptic、AMLSim、Ethereum、DeFi 等数据适配 |
| `src/fraud_ml_engineering/vendor/splitgnn/` | SplitGNN 模型、训练、预处理和指标工具 |
| `src/fraud_ml_engineering/hybrid_task_model/` | 图/序列编码、融合、loss、forward 和 SSL mixin |
| `src/fraud_ml_engineering/evaluator.py`、`embedding_analysis.py` | 测试评估和表示分析 |
| `src/fraud_ml_engineering/run_artifacts.py`、`checkpointing.py` | manifest、checkpoint 兼容和产物路径 |
| `scripts/`、`configs/`、`tests/` | 主线、消融、tuning、报告、结构和协议测试 |

训练协议明确先用 validation 选择 checkpoint/threshold，再只在最后评估 held-out test；历史 best 记录不能自动晋升为新实验结果。`record_run_manifest.py` 只记录元数据，不启动训练，这一边界有助于审查命令和结果是否被混淆。

## 开发过程

1. 先做 dataset registry 和路径策略，确保不同数据集不会共享隐式缓存。
2. 引入 vendored SplitGNN，并保留第三方版权与 `THIRD_PARTY_NOTICES.md`。
3. 增加 relation/event sequence 和 hybrid fusion，形成图、序列、融合的可比较主线。
4. 为低标签、融合、IEEE acceptance 和 on-chain 任务编写独立脚本和配置。
5. 通过 CPU smoke、manifest、checkpoint compatibility、comparison report 和 CI 固化发布前检查。

## 涉及知识点

- 图神经网络、heterophily、消息传递和 SplitGNN
- Transformer/序列建模、事件时间和关系序列
- 多模态/多视图融合、低标签学习、SSL
- 类别不平衡、阈值选择、验证集与 held-out test
- DGL/ PyTorch 设备管理、CPU/CUDA 12.1 环境
- 数据集适配、缓存、checkpoint 兼容和实验 manifest
- 可复现性、第三方代码归属和不自动下载数据策略

## 测试与验证

```powershell
python scripts/validate_repository.py
python -m pytest
python scripts/run_splitgnn_smoke_suite.py --dataset comp --device cpu
python -m fraud_ml_engineering --dataset elliptic --rounds 1 --local_epochs 1 --disable_tb
```

长实验前先记录 manifest：

```powershell
python scripts/record_run_manifest.py --output artifacts/elliptic/manifest.json --dataset elliptic --seed 42 --config configs/experiments/onchain_main_selection.yaml -- python -m fraud_ml_engineering --dataset elliptic --rounds 20 --local_epochs 2 --disable_tb
```

这些命令证明的是仓库合同、轻量路径或指定数据上的运行，不自动证明论文级指标；真实结果必须连同数据 revision、seed、配置和 split policy 保存。

## 学术申请中的项目价值

它能把“模型想法”落实成可审查研究流程：多视图输入、第三方模型集成、低标注消融和报告生成均有独立入口。对图学习、金融 AI、时序异常检测和 ML systems 申请，manifest、数据治理和验证选模规则是比单一 AUC 更可靠的工程信号。

## 运行入口

- CLI：`python -m fraud_ml_engineering`
- SplitGNN smoke：`scripts/run_splitgnn_smoke_suite.py`
- Hybrid 主线：`scripts/run_hybrid_mainline_protocol.py`
- 消融：`scripts/run_hybrid_fusion_ablation.py`、`run_hybrid_low_label_mechanism_ablation.py`
- 报告：`scripts/generate_auditable_comparison_report.py`

## 证据边界

仓库不包含公开数据集、生成图、预训练模型或 TensorBoard 输出。README 的研究问题是协议目标，不是已经由每个数据集统一验证的结论；任何性能数字必须回到具体数据版本、训练配置、seed、测试划分和 artifacts。SplitGNN 的许可证与归属按 `THIRD_PARTY_NOTICES.md` 处理。

## 源码审计记录

| 层次 | 代表文件 | 逐文件审计问题 |
| --- | --- | --- |
| CLI | `main.py`、`__main__.py`、`cli_contract.py` | dataset、device、rounds、local epochs、TB、cache-only 和异常退出是否有明确语义。 |
| 路径 | `paths.py`、`runtime_dataset_policy.py` | source data、graph cache、artifact 和 checkpoint 是否隔离，是否拒绝隐式下载。 |
| 注册 | `dataset_registry.py`、各 `*_dataset.py` | 每个数据集的字段、标签、时间/节点关系、split 和缓存版本。 |
| 图模型 | `model.py`、`vendor/splitgnn/model.py`、`metric_utils.py` | 节点特征、边/关系、heterophily 处理、输出 logits 和第三方实现边界。 |
| 序列模型 | `hybrid_task_model/sequence_mixin.py`、`encoding_mixin.py` | relation/event token、padding/mask、时间顺序和序列长度。 |
| 融合/loss | `forward_mixin.py`、`loss_mixin.py`、`_core.py`、`ssl_mixin.py` | graph/sequence branch 的拼接、监督/SSL loss、低标签开关和预测输出。 |
| 训练 | `trainer_local.py`、`paper_runner_runtime.py`、`experiment_protocol.py` | seed、选模、checkpoint、validation/test 时序和资源限制。 |
| 评估 | `evaluator.py`、`embedding_analysis.py`、`run_artifacts.py` | threshold、指标、embedding、manifest 和 report schema 是否绑定。 |
| 实验脚本 | `scripts/run_*`、`configs/experiments/` | 主线、融合消融、低标签、IEEE tuning/acceptance 是否相互独立。 |
| 质量门 | `tests/`、`validate_repository.py` | 结构、路径、CLI、实验协议、结果 artifact 和 comparison report。 |

## 数据到预测的调用链

1. CLI 解析数据集和实验配置，registry 选择 adapter；路径策略定位原始数据，禁止代码擅自下载。
2. adapter 标准化实体、边、关系和事件，构造或读取版本化 graph cache，并把 dataset revision 写进 manifest。
3. graph branch 生成节点/子图表示；sequence branch 按时间构造 relation/event tokens；mask 保证 padding 不产生信号。
4. fusion head 合并图与序列表征，计算分类 loss；低标签和 SSL 模式只通过显式 config 开启。
5. validation 阶段选择 checkpoint 和阈值；held-out test 被延后到选择完成后，避免 test 泄漏。
6. evaluator 写出指标、混淆矩阵/阈值、embedding 和 runtime metadata，comparison generator 只接受 schema 合法记录。

## 研究问题的可证伪实现

- 图编码器增益：比较 graph-only、sequence-only、hybrid，并保持数据 revision、seed 和 test split 相同。
- 序列增益：关闭 relation/event branch 或替换为固定聚合，查看是否在多个 seed 中稳定。
- 低标签稳健性：用显式 label fraction 和 mechanism ablation，不把一次最好结果当作普遍结论。

## 运行异常的排查顺序

先运行 `validate_repository.py` 检查路径和仓库合同；若 import 失败，确认 Python 与 requirements profile；若 graph cache 不一致，清理仅对应 dataset 的 local cache 并重新记录 revision；若指标异常，首先核对 split、threshold、label mapping 和 validation 选择，而不是直接修改模型。GPU 训练异常应另存资源、设备、CUDA/PyTorch 版本和命令，不以 CPU smoke 代替。

## 交付检查表

- [ ] 每次 run 都有 seed、dataset revision、config、commit 和 command manifest。
- [ ] graph cache、checkpoint、TB 和 report 不进入 source package。
- [ ] validation/test 选择顺序可在日志中复核。
- [ ] vendored SplitGNN 的版权、许可证和修改点被保留。
- [ ] 主线与消融使用同一指标 schema，历史 best 不自动晋升。

## 原理与理论基础

### 1. 交易图与异质关系

把账户、商户、设备或交易视为节点/边后，欺诈信号可以通过邻域传播表达。对节点 v 的一层消息传递可抽象为 h_v(l+1)=phi(h_v(l), AGG({psi(h_v(l),h_u(l),e_uv)}))；e_uv 保留交易类型、金额、方向或时间关系。金融图常是异质、稀疏且高度不平衡的，简单同质 GCN 可能把正常邻居的统计特征过度平滑，SplitGNN 分支的意义在于对关系/结构差异保留更细粒度的信号。

### 2. 关系序列和事件序列

图结构回答“谁与谁相连”，序列回答“这些关系以什么顺序发生”。对用户或实体的事件序列 e_1,...,e_T，Transformer 通过 softmax(QK^T/sqrt(d))V 建立远距离依赖，时间编码和 padding mask 决定模型是否能区分先后、缺失和无效 token。关系序列、事件序列与 graph branch 不是简单拼接：应明确每个分支看到的字段、时间窗口和标签可见性，才能做公平 ablation。

### 3. 融合目标与不平衡评估

hybrid 模型通常最小化 L=lambda_g L_graph+lambda_s L_seq+lambda_f L_fusion+lambda_ssl L_ssl，但 loss 权重不等于业务价值。欺诈数据常有极低正类比例，accuracy 会被多数类主导；应结合 precision/recall、PR-AUC、阈值、成本矩阵和 calibration。阈值只能由 validation 选择，held-out test 只在流程末端读取，否则调参会把测试集变成训练信号。

### 4. 低标签与实验因果

低标签实验的关键不是随机删几条标签后报一次结果，而是固定未标注数据、label fraction、seed、训练步数和评价集，比较 graph-only、sequence-only、hybrid 及机制消融。若多个数据集、多个 seed 和相同 protocol 下仍有一致增益，才可以讨论模块的稳健贡献；单个 dataset 的最好结果只支持局部观察。

### 5. 理论假设和失效模式

模型假设实体映射、时间顺序、标签定义和 split 没有泄漏；缓存、未来事件、重复账户或跨 split 邻边会破坏这个假设。图过平滑、序列截断、类别漂移、threshold 偏置、GPU 非确定性和错误的 label mapping 都可能导致虚高/虚低结果，因此 repository validation、manifest、dataset revision 和 comparison schema 是模型理论之外的必要组成。
