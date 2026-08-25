# 可复现实验包

## 环境版本

- 日期：2026-06-21
- 操作系统：Microsoft Windows 11 专业版 `10.0.26100`
- Shell：PowerShell
- Python：`python`
- Python 版本：`Python 3.13.5`
- 项目根目录：`<aerogate_graph>`

## 依赖版本

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

## 测试命令

```powershell
cd <aerogate_graph>
python -m pytest tests
```

预期输出：

```text
8 passed
```

## 导入烟测命令

该仓库有部分历史文件带 UTF-8 BOM，因此烟测使用 `utf-8-sig` 读取源码。

```powershell
cd <aerogate_graph>
@'
import ast
from pathlib import Path

root = Path.cwd()
failures = []
count = 0
for path in sorted(root.rglob("*.py")):
    if "__pycache__" in path.parts:
        continue
    rel = path.relative_to(root)
    try:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(rel))
        count += 1
    except Exception as exc:
        failures.append((str(rel), repr(exc)))

print(f"parsed={count} failures={len(failures)}")
for rel, exc in failures:
    print(rel, exc)
raise SystemExit(1 if failures else 0)
'@ | python -
```

预期输出：

```text
parsed=183 failures=0
```

## 单机动态门评估入口

需要提供训练好的 checkpoint。随机种子与示例视频一致时使用 `--seed 0`，动态门密度示例使用 `--gate-count 42`。

```powershell
cd <aerogate_graph>
python gate_density_single\scripts\run_gate_density_eval.py `
  --checkpoint <checkpoint.pt> `
  --gate-count 42 `
  --seed 0 `
  --random-yaw `
  --moving-gates `
  --moving-gate-amplitude-m 0.24 `
  --moving-gate-speed-mps 1.0 `
  --episodes 10 `
  --output-dir outputs\single_dynamic_gate42_seed0
```

预期输出：`outputs\single_dynamic_gate42_seed0` 下生成 JSON/CSV 评估摘要，字段包含 `success`、`collision`、`timeout`、`done_reason`、`progress_distance_m`、`flight_time_s`、`actual_gate_motion_range_m`、`moving_gate_swept_clearance_m_min` 等。

## 多机静态/动态评估入口

```powershell
cd <aerogate_graph>
python multi_gate\scripts\run_paper_multi_gate_density_eval.py `
  --checkpoint <checkpoint.pt> `
  --experiments E4_static_multi_8d E5_dynamic_multi_8d `
  --methods full `
  --gate-counts 36 60 `
  --team-sizes 8 `
  --seeds 0 `
  --episodes 3 `
  --workers 1 `
  --output-root outputs\paper_2d_repro
```

预期输出：`outputs\paper_2d_repro` 下生成多机静态/动态结果 CSV/JSON，字段覆盖 `success_rate_pct`、`collision_rate_pct`、`path_length_m`、`flight_time_s`、`gate_post_radius_m`。

## 经典规划器基线入口

该命令读取已有主方法结果目录，并输出规划器 rows、summary、comparison、audit、metric contract。动态门基线可固定速度/幅值以对齐主方法设置。

```powershell
cd <aerogate_graph>
python scripts\run_classic_planner_baselines.py `
  --mode smoke `
  --results-root outputs\paper_2d_repro `
  --experiment E2_dynamic_single_gate_density `
  --gate-count 42 `
  --seed 0 `
  --fixed-dynamic-gate-speed-mps 1.0 `
  --fixed-dynamic-gate-amplitude-m 0.24 `
  --output-dir outputs\planner_baseline_gate42_seed0
```

预期输出：

- `planner_baseline_rows.jsonl`
- `planner_baseline_summary.csv`
- `planner_vs_completed_mainline.csv`
- `planner_baseline_audit.json`
- `planner_baseline_metric_contract.json`
- `planner_baseline_run_manifest.json`

## 随机种子

评估附件中已记录的主要种子：

- 单机动态门 demo：`gate_count=42`，`seed=0`
- 多机静态门 demo：`gate_count=60`，`team_size=8`，`seed=0`
- 多机动态门 demo：`gate_count=36`，`team_size=8`，`seed=0`
- 多机四指标图表数据：每个门密度 `episodes=3`
- 单机规划器基线 CSV：每个门密度 `seed_count=10`

## 关键结果摘要

来自 `results/csv_json/single_dynamic_planner_baseline_eight_metrics.csv`：

| 场景 | 方法 | seed 数 | 成功率 | 碰撞率 | 超时率 |
|---|---:|---:|---:|---:|---:|
| 单机动态 42 门 | ours_mainline | 10 | 1.0 | 0.0 | 0.0 |
| 单机动态 42 门 | astar | 10 | 0.1 | 0.8 | 0.1 |
| 单机动态 42 门 | theta_star | 10 | 0.0 | 0.9 | 0.1 |
| 单机动态 42 门 | rrt_star | 10 | 0.0 | 0.0 | 1.0 |
| 单机动态 42 门 | informed_rrt_star | 10 | 0.0 | 0.0 | 1.0 |
| 单机动态 42 门 | heuristic | 10 | 0.0 | 0.0 | 1.0 |

来自 `results/csv_json/multi_static_dynamic_four_metrics_plot_data.csv`：

| 场景 | 门数 | episodes | 成功率 | 碰撞率 | 路径长度 m | 飞行时间 s | 门柱半径 m |
|---|---:|---:|---:|---:|---:|---:|---:|
| multi static | 60 | 3 | 100.0 | 0.0 | 66.5250 | 35.2667 | 0.14 |
| multi dynamic | 36 | 3 | 100.0 | 0.0 | 67.0867 | 47.0000 | 0.24 |
| multi dynamic | 60 | 3 | 100.0 | 0.0 | 66.6690 | 38.0333 | 0.24 |

注意：同一 CSV 中单机动态 60 门主方法行标记为 `real_eval_failed`，成功率 `0.0`、碰撞率 `0.6`。该失败样本用于记录高密度压力边界，不应改写为成功结果。

## 结果哈希核验

全部附件文件的 SHA256 固化在 `results_manifest.json`。可用以下命令复核：

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
for row in bad:
    print(row)
raise SystemExit(1 if bad else 0)
'@ | python -
```

预期输出：

```text
checked=11 mismatches=0
```
