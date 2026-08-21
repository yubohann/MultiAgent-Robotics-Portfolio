# 环境配置过程概要

本项目为 Windows/PowerShell 下整理出的 gate-only 最小复现实验代码。纯 2D 逻辑、测试、CSV/JSON 指标处理不依赖 Isaac Sim 图形运行时；IsaacLab/Isaac Sim 主要用于 3D 回放与 MP4 渲染。

## 已核验环境

- 操作系统：Microsoft Windows 11 专业版 `10.0.26100`
- Shell：PowerShell
- Python：`python`
- Python 版本：`Python 3.13.5`
- 项目根目录：`<aerogate_graph>`

## 核心依赖版本

```text
numpy==1.26.4
torch==2.7.0+cu128
matplotlib==3.10.0
pandas==2.2.3
scipy==1.15.3
pytest==8.3.4
gymnasium==1.2.3
networkx==3.4.2
```

## 基本启动

```powershell
cd <aerogate_graph>
python -m pytest tests
```

如果需要运行 3D replay 或重新导出视频，需要先安装并配置 NVIDIA Isaac Sim / IsaacLab，并确保项目中的 `assets/gate/gate.usd` 与 `assets/gate/gate.glb` 可访问。

## 目录策略

- 代码只保留 gate-only 任务，不包含 tree/forest-only 模块。
- 评估附件集中在 `evaluation_artifacts/results/`。
- 运行输出建议统一写入 `outputs/` 或命令行显式指定的 `--output-dir` / `--output-root`。
- 中间训练配置文件不纳入评估附件；如需复现实验，使用 `reproducibility.md` 中的命令入口和随机种子。
