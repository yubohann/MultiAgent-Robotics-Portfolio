# Deterministic Replay Package

## Environment Versions

- Date, 2026-06-21
- OS, Microsoft Windows 11 Pro `10.0.26100`
- Shell, PowerShell
- Python, `python`
- Python version, `Python 3.13.5`
- Project root, `<gate_graph_2d_minimal>`

## Dependency Versions

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

## Test Command

```powershell
cd <gate_graph_2d_minimal>
python -m pytest tests
```

Expected output,

```text
```

## Import Smoke Command

Some historical files carry a UTF-8 BOM, and the smoke test reads source as `utf-8`.

```powershell
cd <gate_graph_2d_minimal>
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
        ast.parse(path.read_text(encoding="utf-8"), filename=str(rel))
        count += 1
    except Exception as exc:
        failures.append((str(rel), repr(exc)))

print(f"parsed={count} failures={len(failures)}")
for rel, exc in failures:
    print(rel, exc)
raise SystemExit(1 if failures else 0)
'@ | python -
```

Expected output,

```text
parsed=148 failures=0
```

## Single-Agent Dynamic Gate Scoring Entry

A trained checkpoint is required. The seed and sample video pair with `--seed 0`, and the dynamic gate density sample uses `--gate-count 42`.

```powershell
cd <gate_graph_2d_minimal>
python -m tasks.density_single.scripts.run_gate_density_eval `
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

Expected output, `outputs\single_dynamic_gate42_seed0` holds JSON and CSV scoring summaries with fields such as `success`, `collision`, `timeout`, `done_reason`, `progress_distance_m`, `flight_time_s`, `actual_gate_motion_range_m`, `moving_gate_swept_clearance_m_min`.

## Multi-Agent Static and Dynamic Scoring Entry

```powershell
cd <gate_graph_2d_minimal>
python -m tasks.multi.scripts.run_paper_multi_gate_density_eval `
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

Expected output, `outputs\paper_2d_repro` holds multi-agent static and dynamic CSV and JSON results with fields covering `success_rate_pct`, `collision_rate_pct`, `path_length_m`, `flight_time_s`, `gate_post_radius_m`.

## Classic Planner Baseline Entry

This command reads a completed mainline results directory and writes planner rows, summary, comparison, audit and metric contract files. Dynamic gate baselines can pin speed and amplitude to match the mainline settings.

```powershell
cd <gate_graph_2d_minimal>
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

Expected output,

- `planner_baseline_rows.jsonl`
- `planner_baseline_summary.csv`
- `planner_vs_completed_mainline.csv`
- `planner_baseline_audit.json`
- `planner_baseline_metric_contract.json`
- `planner_baseline_run_manifest.json`

## Seeds

Recorded seeds in the artifact set,

- Single-agent dynamic gate demo, `gate_count=42`, `seed=0`
- Multi-agent static gate demo, `gate_count=60`, `team_size=8`, `seed=0`
- Multi-agent dynamic gate demo, `gate_count=36`, `team_size=8`, `seed=0`
- Multi-agent four-metric figure data, `episodes=3` per gate density
- Single-agent planner baseline CSV, `seed_count=10` per gate density

## Key Result Summary

From `results/csv_json/single_dynamic_planner_baseline_eight_metrics.csv`,

| Scene | Method | Seeds | Success rate | Collision rate | Timeout rate |
| --- | --- | --- | --- | --- | --- |
| Single-agent dynamic 42 gates | ours_mainline | 10 | 1.0 | 0.0 | 0.0 |
| Single-agent dynamic 42 gates | astar | 10 | 0.1 | 0.8 | 0.1 |
| Single-agent dynamic 42 gates | theta_star | 10 | 0.0 | 0.9 | 0.1 |
| Single-agent dynamic 42 gates | rrt_star | 10 | 0.0 | 0.0 | 1.0 |
| Single-agent dynamic 42 gates | informed_rrt_star | 10 | 0.0 | 0.0 | 1.0 |
| Single-agent dynamic 42 gates | heuristic | 10 | 0.0 | 0.0 | 1.0 |

From `results/csv_json/multi_static_dynamic_four_metrics_plot_data.csv`,

| Scene | Gates | Episodes | Success rate | Collision rate | Path length m | Flight time s | Gate post radius m |
| --- | --- | --- | --- | --- | --- | --- | --- |
| multi static | 60 | 3 | 100.0 | 0.0 | 66.5250 | 35.2667 | 0.14 |
| multi dynamic | 36 | 3 | 100.0 | 0.0 | 67.0867 | 47.0000 | 0.24 |
| multi dynamic | 60 | 3 | 100.0 | 0.0 | 66.6690 | 38.0333 | 0.24 |

In the same CSV the single-agent dynamic 60-gate mainline row carries the `real_eval_failed` mark with success rate `0.0` and collision rate `0.6`. That row records the high-density stress limit and stays recorded as a failure sample.
