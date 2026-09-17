# AeroCityBench

<p align="center">
  <img src="assets/demos/aerocity-bench-overview.gif" alt="AeroCityBench 概览" width="49%" />
  <img src="assets/demos/multi-uav-exploration.gif" alt="多无人机探索" width="49%" />
</p>

**面向城市拓扑，目标过程与机群韧性变化的物理约束多无人机三维目标搜索开放基准。**

城市中的覆盖指标经常失真。无人机可能飞过建筑却错过屋顶，检查错误立面，被几何遮挡截断视线，或因穿越过快错过有效观测窗口。AeroCityBench 把这些情形变成可计分的任务，程序化三维城市，公开任务合同，计分器私有目标真值，确认必须由飞行中的合法观测取得。

**状态。** `v0.2.0.dev0` 原型。生成器，合同，计分器，审计工具与校准基础设施已经实现，正式盲测在验证门禁通过后开放，见[研究记录](docs/research-notes.md)。

## 检验内容

基准检验覆盖更多空间能否转化为发现更多目标。目标只有在计分器接受一次真实观测时才计分。

```mermaid
flowchart LR
    A[程序化三维城市] --> B[公开任务合同]
    B --> C[四机协同方法]
    C --> D[物理执行与 OBSERVE 动作]
    D --> E[计分器私有确认]
    E --> F[回执与分项指标]
```

一次合法 OBSERVE 必须同时满足距离，视场，朝向，视线，允许表面侧，持续观测时间，新鲜度，位姿稳定，净空与运行时安全，并且计数唯一，回执绑定真实传感器帧。

| 方法可见 | 计分器私有 |
| --- | --- |
| 自身状态，允许的传感器，时间与能量预算 | 目标坐标，数量，标签与生成过程 |
| 公开起点，通信消息与目标无关的粗先验 | 合法观测见证与确认判定 |
| 由几何生成的 G2-I 检查图谱 | 测试划分，城市族与随机种子 |

## 基准内容

- 受约束的程序化城市生成器，版本化发布配置与开放资产许可检查。
- 公开与私有任务投影，JSON Schema，内容哈希与发布验证。
- 由几何独立编译的 G2-I 检查图谱与递归泄漏检查。
- 绑定真实观测的 `OBSERVE` 合同与计分器私有确认。
- 基线适配器以及外部方法的 CPU 与原生预检工具。
- 在 CPU 上运行的合同，完整性，主机保护与质量门禁测试。

## 已核验的证据

当前原型的开发级证据。正式分数只来自带 `formal_score_eligible=true` 的冻结记录。

- 三祖先四机 CF2X 与 PhysX 校准面板安全闭合，全员返航，L0 与 L1 排名相关为 1.0。
- L2 视觉审查在冻结的 960x640 配置下通过，覆盖训练，校准与验证城市。
- 十二座开发城市通过静态几何，上下文与完整回合准入审查。
- CC0 迷你资产集闭合，32 个 USD 层，零远程引用，零未解析路径。
- mission-sector 证书从公开分配独立重算飞行，观测，爬升与返航下界，任何不一致即失败关闭。

## 快速开始

需要 Python 3.11 或更高版本。

```powershell
python -m pip install -e ".[dev]"
python -m pytest tests/test_public_boundary.py tests/test_inspection_atlas.py tests/test_ordinary_v3.py -q
```

本地开发发布需要经过验证的开放资产包与可写输出目录。

```powershell
$env:PYTHONPATH = (Resolve-Path .\src)
$assetRoot = (Resolve-Path $env:AEROCITY_ASSET_ROOT)
$outputRoot = Join-Path $env:AEROCITY_OUTPUT_ROOT "ordinary-v1-mini"

python -m aerocity_bench build --release .\configs\releases\ordinary-v1-mini.json `
  --asset-root $assetRoot --output $outputRoot --task-track G2-I --allow-uncommitted-development
python -m aerocity_bench validate $outputRoot
```

包含原生与外部适配器的完整路径见[运行指南](docs/run-guide.md)。

## 文档

- [基准设计](docs/benchmark-design.md)，任务，确认合同，城市生成，划分与指标。
- [研究记录](docs/research-notes.md)，研究问题，带日期的设计决策，验证门禁与证据纪律。
- [运行指南](docs/run-guide.md)，安装，测试，构建与适配器入口。

## 许可

仓库自有代码与文档遵循 [LICENSE](LICENSE)。第三方引擎与资产按各自条款处理，清单见资产登记表。
