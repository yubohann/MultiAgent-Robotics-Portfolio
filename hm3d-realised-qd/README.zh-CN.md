# HM3D Realised-QD

<p align="center">
  <img src="assets/demos/hm3d-scene-1.gif" alt="HM3D 多无人机探索场景" width="78%" />
</p>

**面向 HM3D 派生三维场景的目标无关多无人机探索，以真实执行回执驱动质量多样性与强化学习。**

四架 CF2X 四旋翼在未知室内扫描场景中飞行，只使用公共稀疏测距观测。机队融合共享 belief，从公共候选池中挑选团队计划，并在真实 Isaac 与 PhysX 动力学下执行。行为多样性只来自执行回执，档案记录真实飞过的行为，所有方法使用同一观测，同一安全合同与同一物理时间预算。

**状态。** `v0.1.0` 研究快照，2026-08-08。realised-QD 选择器拥有覆盖 42 个真实回合的已核验 P10 组件级结果，RB-SF-SAC 策略与 RFG 片段复用是正式选择路径的下一步接入工作。

## 我的职责

`pyproject.toml` 将 Bohan Yu 列为项目作者，本目录记录了署名背后留档的工作。

- **研究设计。** 形式化目标无关多无人机探索任务、共享感知/通信/安全合同与 `Explored-Free-Flight-Volume-AUC_time` 指标，并在[主方法设计](docs/method-design-realised-qd-rfg-rb-sf-sac-2026-08-08.md)中预注册假设与消融链。
- **世界模型与 QD 学习实现。** 实现稀疏占据 belief、前沿/路线访问/观察候选生成器、带意图审计回退的三维 realised-QD 档案与选择器（`src/realised_qd/runtime/hm3d_realised_qd.py`、`src/realised_qd/archives/`），以及 RB-SF-SAC、masked-PPO 与 replay 学习栈（`src/realised_qd/learning/`）。
- **探索与结果管线。** 实现 CF2X/PhysX 执行器、安全账本、transit 时序校准、持久化采集与训练结果数据集构建（`src/realised_qd/runtime/`、`scripts/exploration/`、`scripts/mechanism/`）。
- **实验执行。** 在真实 Isaac 与 PhysX 上完成 42 个正式 P10 回合、00626 机制消融与 QD 回放校准；realised-QD 组件在 00626 场景领先全部五个基线，消融闭合 no_qd < planned_qd < realised_qd 梯度。
- **工程与文档。** 修复[实验结果汇总](docs/experiment-results-2026-08-08.md)记录的 7 项管线缺陷并配回归测试，撰写方法规格、协议配置与 `tests/` 测试集。

## 核心设计

- 任务为 HM3D 派生室内场景的目标无关在线探索，主指标 `Explored-Free-Flight-Volume-AUC_time`，所有方法共享 CF2X，通信，安全与物理时间合同。
- 公共稀疏测距结果构建稀疏占据 belief。前沿，路线访问与观察候选由该 belief 生成，经静态净空守卫与团队联合守卫准入，池上限冻结为 16。
- realised-QD 档案在三维描述符空间保存行为精英，共 64 个 cell，描述符与质量来自真实执行回执。
- RB-SF-SAC 增加循环 belief 状态共享前沿 SAC 层，为同一批团队候选排序。
- RFG 以完成且溯源干净的执行为门控复用片段，授予实测体积收益，并在片段被后续执行重写时撤销收益。
- PhysX 执行器以 CF2X 动力学运行每个被选中的 manifest，物理预算 40 秒，每个回合产出回执，安全账本与结果哈希。

## 已核验结果

- 42 个正式 P10 回合在真实 Isaac 与 PhysX 中闭合，零碰撞，零越界，零分离违规，零失败片段，全部 transit 完成。
- 在 325.17 m³ 的 train 场景 00626 上，realised-QD 组件取得 0.0984 AUC，领先全部五个基线，random 0.0915，frontier_3d 0.0917，auction 0.0838，gvp_mrep_port 0.0891 与 single_rl 0.0932，覆盖率 0.1418 最高，路程 28.8 m 最长。
- 在 145.13 m³ 的 train 场景 00459 上，该组件取得 0.3436 AUC，仅次于 frontier_3d 的 0.3854。
- 00626 的机制消融形成从计划意图到真实回执的梯度，no_qd 0.0909，planned_qd 0.0926，realised_qd 0.0984。
- QD 回放校准在相同公共起点上重复了 12 次意图模式执行中的 9 次。

## 快速开始

```powershell
uv sync --extra dev --extra rl --extra hm3d
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m ruff check src tests scripts
```

Isaac 与 PhysX 运行通过 `scripts/run_isaac_python.ps1` 启动，需要已验证的 IsaacLab 解释器，并设置 `REALISED_QD_CF2X_USD` 指向本地 CF2X USD 资产。公开源码树只包含代码，合同与文档。HM3D 资产，转换网格，检查点与原始运行输出保存在本地工作区。

## 文档

- [文档索引](docs/README.md)，当前设计，结果与研究记录集合。
- [主方法严格设计](docs/method-design-realised-qd-rfg-rb-sf-sac-2026-08-08.md)，realised-QD，RB-SF-SAC 与 RFG 的方法合同。
- [P10 主表结果](docs/p10-main-results-2026-08-08.md)，组件级结果表与消融。
- [论文实验结果汇总](docs/experiment-results-2026-08-08.md)，上述数字背后的冻结研究记录。

## 许可

源码树保留项目自身的发布条款。第三方代码与资产按各自条款处理。
