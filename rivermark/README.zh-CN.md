# Rivermark

<p align="center">
  <img src="assets/demos/rivermark-search.gif" alt="Rivermark 多智能体三维搜索" width="78%" />
</p>

**面向多智能体三维隐蔽搜索的物理仿真基准工具链，基于 Isaac Sim 原生采集，八架 CF2X 四旋翼在程序化 City-Lite 城市场景中协同飞行。**

城市会藏住目标。四旋翼可能从建筑旁飞过而错过庭院，在障碍物边缘丢掉目标视线，或者在有效窗口内通过得太快。Rivermark 把这些情形记录成同步多传感器回合，每个仿真步先写入控制指令再推进状态，观测与动作的因果链保持完整。场景，采集协议，运行环境与源代码都有明确的内容标识，回合通过独立验证后进入正式数据集。隐藏目标真值由计分器保管，搜索方法依靠飞行中的真实观测取得确认。

**状态。** 采集协议 `citylite-t1-expert-coverage-v2` 已冻结，四个训练回合与四个验证回合全部采集完成，当前协议下不再接受新的采集绑定。原生采集路径面向 Isaac Sim 5.1 与 Isaac Lab 2.3.2，CPU 路径可以在干净检出上完成验证。

## 我的职责

Rivermark 由我从零设计与实现，以下条目都对应仓库中可查证的材料。

- **基准设计。** Search3D 任务合同，观测与动作 ABI，City-Lite 场景合同，内容标识与运行环境锁方案，以及冻结的 `citylite-t1-expert-coverage-v2` 采集协议及其 `config/` 记录与逐产物 `schemas/` 合同。
- **场景与任务实现。** `src/rivermark_benchmark/citylite_scene/` 中的程序化 City-Lite 组合：路线族，安全起点，材质与几何实现以及条件检查。
- **采集与计分管线。** `src/rivermark_benchmark/isaac/` 中的 Isaac Sim 原生采集通路，`data/` 中的回合合同与投影，`audit/` 中的独立验证器与正式数据集准入，`runtime/` 中的失败账本，以及 `score/` 中带公开指标定义的计分器。
- **学习基线。** `train/` 中的试点方法集（经典规划，RL，MARL，质量多样性与 torch 训练路径），SB3 迁移路径，以及 `config/baseline_suite.cpu_pilot_v1.json` 的受限基线套件与带出处标注的报告。
- **实验执行。** 冻结批次的采集与捕获运行，同种子重复性采集与几何扫描，`media/` 中的路线见证与单元视频以及 `evidence/` 中的报告。
- **文档。** README 全集，`docs/` 指南，schema，changelog 与引用元数据。

## 已核验结果

以下为冻结批次的开发级证据。

- 同一运行环境锁下对回合种子 1751072442 的两次原生采集通过了全部声明比较，语义标签一致率 1.0，下限 0.98，机载 RGB 帧平均绝对误差 2.24，上限 8.0。
- 原生几何扫描在两个路线配对上均实现了全部 4 个直接可见目标。
- 冻结的 City-Lite 合同组合了约 20000 个活动 prim 与 276 个使用的 USD 图层，含 4807 个可行驶表面碰撞体与 4 个任务障碍碰撞体。
- 指令体覆盖 92 米乘 92 米乘 5.25 米，两个路线族相交于 5 个点，共享 0 个航点与路段。

## 记录内容

每个回合是一段面向八机编队的同步多智能体时序。

- 机载 RGB 与深度
- 原生语义分割学习标签
- RayCaster 激光雷达测距
- IMU，接触与安全状态
- 机体位姿与速度
- 每个仿真步之前写入的控制指令，以及公开路线状态与显式队伍消息
- 相机标定，时间戳与世界，机体，相机变换闭环

固定世界俯瞰相机充当路线见证，在每个保留帧上渲染并检查。

## 确定性与准入

场景，协议，运行环境与源代码树都以 content identity 绑定。运行环境锁配置 `citylite-windows-isaacsim-5.1.0.0-local-isaaclab-2.3.2` 固定解释器，依赖版本，显卡下限以及渲染与物理配置。同种子分析器在预先声明的容差下比较同一回合种子的两次采集，并按类别与智能体编号做帧对齐。

独立验证器重新打开原始产物，检查场景身份，传感器同步，动作因果，视觉与激光雷达侵入门禁，接触，路线实现与目标可见性证据。通过检查的回合进入正式数据集，失败产物保留在失败账本中，长时间采集期间支持崩溃遗留恢复。

## 计分

计分器接受带时间戳的确认事件，每个事件绑定来源观测编号。计分器掌握可见性，对隐藏目标做匹配，并返回经过验证的报告，包含召回率，确认面积曲线，首次确认时间，错误确认，碰撞，超时，任务成本与失败率。指标定义版本化并公开，计分器输入在准入之前保持私密。

## 媒体与证据

`media` 目录保存训练单元 0，1，2，3，9，11 与验证单元 13 到 19 以及 route-witness-r31 的俯瞰与合成 MP4 和关键帧。`evidence` 目录保存同种子重复性报告与公开及计分器私有两种回合清单样例。

## 快速开始

需要 Python 3.10 或更高版本。所有命令都在仓库根目录执行。

```powershell
python -m pip install -e ".[cpu-ci]"
$output = Join-Path $env:TEMP 'rivermark-researcher-smoke'
python -m rivermark_benchmark.researcher_entry $output
Get-Content "$output\researcher_smoke_report.json"
```

运行 CPU 测试套件。

```powershell
python -m unittest discover -s tests -v
```

## 源码结构

仓库根目录即独立源码包，面向 Isaac Sim 原生的多智能体三维隐蔽搜索采集，验证与计分。

- `src/rivermark_benchmark/` 按功能分层：`isaac/`（原生采集与证据），`pack/`（打包描述，规格，就绪检查与公开清单），`data/`（合同，回合，归档与投影），`train/`（策略方法与训练），`score/`（计分，评估，诊断与同种子分析），`audit/`（验证，正式数据集准入与发布审计），`runtime/`（仿真运行时，预检，运行环境锁与失败账本），`ops/`（运维与发布媒体工具），以及 `citylite_scene/` 与 `collection_protocol/` 合同包
- `config/` 保存采集协议，运行环境锁，标签本体与基线套件
- `schemas/` 保存每个产物的 JSON Schema 合同
- `docs/` 保存本文档集，以及 API 与 Schema 稳定性，资产政策与安全完整性政策
- `tests/unit/` 保存合同，完整性与质量门禁的 CPU 测试
- `tests/integration/` 保存原生采集，验证，打包与证据管线测试
- `tools/` 保存原生视频规划与编码脚本

## 文档

- [总览](docs/overview.md)，Rivermark 是什么，发布哪些证据，范围如何。
- [任务与场景](docs/task-and-scene.md)，搜索任务与 City-Lite 环境。
- [观测 ABI](docs/observation-abi.md)，回合数据的字段级合同。
- [计分](docs/scoring.md)，指标定义与提交合同。
- [原生采集](docs/capture.md)，运行一次原生 Isaac 采集。
- [视频录制](docs/multistart-native-video-recording.md)，规划并编码两个路线族的原生视频。
- [准入](docs/validation-and-admission.md)，独立验证与正式数据集准入。
- [确定性](docs/determinism.md)，同种子运行，运行环境锁与净室重放。
- [方法](docs/methods.md)，支持的方法族与证据规则。
- [数据访问](docs/data-access.md)，读取回合，投影与研究入口。
- [治理](docs/governance.md)，资产来源，许可与接口稳定性。
- [范围](docs/limitations.md)，当前范围与发展路线。

## 许可

Rivermark 自研源代码，Schema 与文档遵循 **Apache-2.0** 许可。NVIDIA Isaac Sim，Rivermark 内容，CF2X USD 与第三方资产适用各自条款。完整文本见 `LICENSE`。
