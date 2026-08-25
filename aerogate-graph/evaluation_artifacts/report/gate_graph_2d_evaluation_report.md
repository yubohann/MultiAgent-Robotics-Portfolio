# AeroGate Graph 评估附件报告

生成日期：2026-06-21

项目目录：`aerogate_graph`

## 1. 目录目的

本报告说明 `aerogate_graph` 的复现实验材料结构。目录仅保留 gate-only 实验，不包含树、森林或无关 Markdown 历史文档。重点覆盖动态门柱、单机/多机、静态/动态、Graph-FlashSAC、action/safety shield 和经典规划器基线。

## 2. 组件与代码位置

| 组件 | 主要代码位置 |
|---|---|
| 图观测构建 | `single_gate/env/observation_single.py`，`multi_gate/env/observation_multi.py`，`multi_gate/env/observation_runtime.py`，`multi_gate/graph_rl/graph_policy.py` |
| Graph-FlashSAC actor/critic | `single_gate/graph_rl/graph_sac.py`，`single_gate/graph_rl/graph_flashsac.py`，`multi_gate/graph_rl/graph_masac.py`，`multi_gate/graph_rl/graph_flashsac.py` |
| 动态门任务契约 | `shared/core/dynamic_gate_density_2d.py`，`gate_density_single/core/gate_layout.py`，`multi_gate/env/dynamic_gate_runtime.py` |
| action shield / safety shield | `gate_density_single/core/action_shield.py`，`multi_gate/env/safety_shields.py`，`shared/core/collision_2d.py` |
| 单机/多机统一评估 | `gate_density_single/scripts/run_gate_density_eval.py`，`multi_gate/scripts/run_paper_multi_gate_density_eval.py`，`scripts/run_classic_planner_baselines.py` |
| 训练与压力边界复核 | `multi_gate/imitation.py`，`multi_gate/dagger.py`，`multi_gate/training.py`，`gate_density_single/scripts/train_gate_density_imitation.py` |

## 3. 可复现环境

已核验环境：

- Windows 11 专业版 `10.0.26100`
- PowerShell
- Python `3.13.5`
- Python 路径：`python`

核心依赖：

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

## 4. 测试与烟测

测试命令：

```powershell
cd <aerogate_graph>
python -m pytest tests
```

预期结果：`8 passed`。

导入烟测使用 `utf-8-sig` 解析所有 Python 文件，预期 `parsed=183 failures=0`。

## 5. 关键实验结果

单机动态 42 门基线对比，来自 `results/csv_json/single_dynamic_planner_baseline_eight_metrics.csv`：

| 方法 | seed 数 | 成功率 | 碰撞率 | 超时率 |
|---|---:|---:|---:|---:|
| ours_mainline | 10 | 1.0 | 0.0 | 0.0 |
| astar | 10 | 0.1 | 0.8 | 0.1 |
| theta_star | 10 | 0.0 | 0.9 | 0.1 |
| rrt_star | 10 | 0.0 | 0.0 | 1.0 |
| informed_rrt_star | 10 | 0.0 | 0.0 | 1.0 |
| heuristic | 10 | 0.0 | 0.0 | 1.0 |

多机静态/动态结果，来自 `results/csv_json/multi_static_dynamic_four_metrics_plot_data.csv`：

| 场景 | 门数 | episodes | 成功率 | 碰撞率 | 路径长度 m | 飞行时间 s | 门柱半径 m |
|---|---:|---:|---:|---:|---:|---:|---:|
| multi static | 60 | 3 | 100.0 | 0.0 | 66.5250 | 35.2667 | 0.14 |
| multi dynamic | 36 | 3 | 100.0 | 0.0 | 67.0867 | 47.0000 | 0.24 |
| multi dynamic | 60 | 3 | 100.0 | 0.0 | 66.6690 | 38.0333 | 0.24 |

保留压力边界：单机动态 60 门主方法行在原始 CSV 中为 `real_eval_failed`，成功率 `0.0`、碰撞率 `0.6`。该记录用于说明高密度动态门压力边界，不应改写为成功结果。

## 6. 评估附件

原始指标：

- `results/csv_json/single_dynamic_planner_baseline_eight_metrics.csv`
- `results/csv_json/single_dynamic_planner_baseline_availability.csv`
- `results/csv_json/single_dynamic_gate42_video_manifest.csv`
- `results/csv_json/single_dynamic_gate42_video_manifest.json`
- `results/csv_json/single_dynamic_gate42_video_validation_report.csv`
- `results/csv_json/multi_static_dynamic_four_metrics_plot_data.csv`
- `results/csv_json/multi_static_dynamic_four_metrics_manifest.json`

当前代码包不随附实验图片或 MP4 视频；只保留原始 CSV/JSON 指标、replay manifest 和 validation summary。大型视频输出应作为外部 artifact 管理。

全部附件的来源路径、大小、SHA256 见 `evaluation_artifacts/results_manifest.json`。

## 7. 哈希核验

```powershell
cd <aerogate_graph>\evaluation_artifacts
@'
import hashlib, json
from pathlib import Path

root = Path.cwd()
manifest = json.loads((root / "results_manifest.json").read_text(encoding="utf-8-sig"))
bad = []
for item in manifest["files"]:
    path = root / item["relative_path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != item["sha256"]:
        bad.append((item["relative_path"], digest, item["sha256"]))
print(f"checked={len(manifest['files'])} mismatches={len(bad)}")
raise SystemExit(1 if bad else 0)
'@ | python -
```

预期结果：`checked=7 mismatches=0`。

## 8. 结论

`aerogate_graph` 当前保留了可复现环境说明、原始结果、基线对比和 SHA256 完整性清单。后续发布前应重新核对 `results_manifest.json` 与实际附件是否一致。
