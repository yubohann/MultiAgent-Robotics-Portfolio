# 活动文档索引

只以下列文件为当前有效说明：

1. [HM3D QD+RL 大实验整改训练与公平评测执行方案](HM3D_QD_RL大实验整改训练与公平评测执行方案_2026-08-05.md)：当前主合同，包含 P0 门槛、v6/v1 执行结果 schema、控制器冻结、训练/基线公平预算和正式结果边界。
2. [四机完整论文实验执行计划](四机快速实验执行计划_2026-08-03.md)：四机真实执行、执行结果、QD 实时更新和候选级 RL 运行顺序；具体冲突以主合同为准。
3. [HM3D 无目标三维协作探索权威计划](HM3D无目标三维协作探索权威计划_2026-08-02.md)：任务、协议、P01--P10 状态和正式冻结边界。
4. [QD 有效性与丰富性闭环](HM3D_QD有效性与丰富性闭环_2026-08-03.md)：QD 进入正式主实验前的可证伪检查。
5. [P0 裁决与根因修复记录](P0裁决与根因修复记录_2026-08-07.md)：当前协议的根因审查与修复记录。
6. [主方法严格设计](主方法严格设计_realised_QD_RFG_RB_SF_SAC_2026-08-08.md)：realised-QD、RFG 与 RB-SF-SAC 的当前方法合同。
7. [P10 主表结果](P10主表结果_2026-08-08.md)：当前主表汇总与结果边界。
8. [论文实验结果汇总](论文实验结果汇总_2026-08-08.md)：实验记录与论文级汇总入口。

历史性的母论文复现和路线讨论材料不再作为当前实验执行命令；它们保留在归档资料中，或只可作为文献参照。

当前任务是无目标的在线三维协作探索。正式传感器是 `sparse_range_3d`；`physics_only` 只用于 H15 吞吐对照。活动面不再存在目标点、目标数量、目标机会、确认召回或深度/RGB-D 正式合同。

未列入上述索引的材料不能作为当前代码入口、评价依据或论文结果。多数历史材料已位于 `../archive/2026-08-03_retired_contracts/`；少数仍保留在本目录的旧路线讨论仅供文献追溯，后续整理时再归档，不得覆盖主合同。

真实运行和 HM3D 单元测试统一使用：
`C:\Users\Administrator\anaconda3\envs\env_isaaclab\python.exe`。
禁止使用 Inkscape 自带 Python、裸 `python` 或裸 `pytest`。

## Repository Map

```text
src/aerocity_method/  contracts, public adapters, realised-QD, RL, runtime, safety, and evaluation
configs/              HM3D protocols, experiment manifests, and external-method contracts
scripts/              public assembly, audit, training, replay, and Isaac launch wrappers
tests/                unit, property, leakage, performance, and integration contracts
manifests/            versioned protocol and evidence bindings
docs/                 active research plans; historical material remains in archive/
```

`datasets/`, `private_eval/`, `reports/`, `results/`, `figures/`, IsaacLab
scene trees, checkpoints, external checkouts, and generated caches are local
research material. They are not package inputs and are excluded from the public
source boundary by `.gitignore` and the release instructions.

## GitHub Metadata

- Recommended repository name: `hm3d-realised-qd`
- Recommended description: `Outcome-grounded quality-diversity and reinforcement learning for target-free multi-UAV exploration in HM3D-derived 3D environments.`
- Suggested topics: `hm3d`, `multi-uav`, `quality-diversity`, `reinforcement-learning`, `3d-exploration`, `multi-agent-systems`, `isaaclab`, `robotics`
