# HM3D Realised-QD 文档

当前文档集覆盖方法设计，正式结果与研究范围。历史过程材料已从本目录移除，需要追溯时见 git 历史。

| 从这里开始 | 文档 |
|---|---|
| 组件级结果表，基线对照与机制消融，2026-08-08 | [P10 主表结果](P10主表结果_2026-08-08.md) |
| 论文级冻结记录，主张，实验体系与数据规模 | [论文实验结果汇总](论文实验结果汇总_2026-08-08.md) |
| realised-QD，RB-SF-SAC 与 RFG 的方法合同与差距清单 | [主方法严格设计](主方法严格设计_realised_QD_RFG_RB_SF_SAC_2026-08-08.md) |

## 目录范围

```text
src/aerocity_method/  contracts, adapters, realised-QD, RL, runtime, safety, scoring
configs/              HM3D protocols, experiment manifests and external-method contracts
scripts/              assembly, audit, training, replay and Isaac launch wrappers
tests/                unit, property, leakage, performance and integration contracts
assets/               demo captures used by the READMEs
archive/              retired pre-freeze material and superseded selection code
debug/                engineering findings, hypotheses and eliminated options
reason/               dated split and outcome decision rounds
docs/                 this documentation set
```

HM3D 资产，Isaac 内容，检查点，私有计分数据与原始运行输出保存在本地工作区，公开源码树只包含代码，合同与文档。

真实运行与 HM3D 单元测试统一使用
`C:\Users\Administrator\anaconda3\envs\env_isaaclab\python.exe`。
Inkscape 自带 Python，裸 `python` 与裸 `pytest` 只用于临时排查。
