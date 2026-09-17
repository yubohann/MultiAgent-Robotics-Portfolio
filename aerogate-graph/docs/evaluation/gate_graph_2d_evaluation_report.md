# Gate Graph 2D Minimal Scoring Artifact Report

Generated on 2026-06-21

Project directory, `gate_graph_2d_minimal`

## 1. Purpose

This report describes the deterministic replay material for `gate_graph_2d_minimal`. The directory keeps gate-only experiments and covers dynamic gate posts, single-agent and multi-agent runs, static and dynamic scenes, Graph-FlashSAC, action and safety shields, and classic planner baselines.

## 2. Components and Code Locations

| Component | Main code locations |
| --- | --- |
| Graph observation construction | `tasks/single/env/observation_single.py`, `tasks/multi/env/observation_multi.py`, `tasks/multi/env/observation_runtime.py`, `tasks/multi/graph_rl/graph_policy.py` |
| Graph-FlashSAC actor and critic | `tasks/single/graph_rl/graph_sac.py`, `tasks/single/graph_rl/graph_flashsac.py`, `tasks/multi/graph_rl/graph_masac.py`, `tasks/multi/graph_rl/graph_flashsac.py` |
| Dynamic gate task contract | `shared/core/dynamic_gate_density_2d.py`, `tasks/density_single/core/gate_layout.py`, `tasks/multi/env/dynamic_gate_runtime.py` |
| Action shield and safety shield | `tasks/density_single/core/action_shield.py`, `tasks/multi/env/safety_shields.py`, `shared/core/collision_2d.py` |
| Single-agent and multi-agent scoring | `tasks/density_single/scripts/run_gate_density_eval.py`, `tasks/multi/scripts/run_paper_multi_gate_density_eval.py`, `scripts/run_classic_planner_baselines.py` |
| Training and stress limit review | `tasks/multi/imitation.py`, `tasks/multi/dagger.py`, `tasks/multi/training/`, `tasks/density_single/scripts/train_gate_density_imitation.py` |

## 3. Deterministic Environment

Verified environment,

- Windows 11 Pro `10.0.26100`
- PowerShell
- Python `3.13.5`
- Python path, `python`

Core dependencies,

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

## 4. Tests and Smoke Test

Test command,

```powershell
cd <gate_graph_2d_minimal>
python -m pytest tests
```

Expected result, `8 passed`.

The import smoke test parses every Python file as `utf-8-sig` with expected output `parsed=183 failures=0`.

## 5. Key Experimental Results

Single-agent dynamic 42-gate baseline comparison from `results/csv_json/single_dynamic_planner_baseline_eight_metrics.csv`,

| Method | Seeds | Success rate | Collision rate | Timeout rate |
| --- | --- | --- | --- | --- |
| ours_mainline | 10 | 1.0 | 0.0 | 0.0 |
| astar | 10 | 0.1 | 0.8 | 0.1 |
| theta_star | 10 | 0.0 | 0.9 | 0.1 |
| rrt_star | 10 | 0.0 | 0.0 | 1.0 |
| informed_rrt_star | 10 | 0.0 | 0.0 | 1.0 |
| heuristic | 10 | 0.0 | 0.0 | 1.0 |

Multi-agent static and dynamic results from `results/csv_json/multi_static_dynamic_four_metrics_plot_data.csv`,

| Scene | Gates | Episodes | Success rate | Collision rate | Path length m | Flight time s | Gate post radius m |
| --- | --- | --- | --- | --- | --- | --- | --- |
| multi static | 60 | 3 | 100.0 | 0.0 | 66.5250 | 35.2667 | 0.14 |
| multi dynamic | 36 | 3 | 100.0 | 0.0 | 67.0867 | 47.0000 | 0.24 |
| multi dynamic | 60 | 3 | 100.0 | 0.0 | 66.6690 | 38.0333 | 0.24 |

Stress limit record, the single-agent dynamic 60-gate mainline row in the raw CSV carries `real_eval_failed` with success rate `0.0` and collision rate `0.6`. The record documents the high-density dynamic gate stress limit.

## 6. Artifacts

Raw metrics,

- `results/csv_json/single_dynamic_planner_baseline_eight_metrics.csv`
- `results/csv_json/single_dynamic_planner_baseline_availability.csv`
- `results/csv_json/single_dynamic_gate42_video_manifest.csv`
- `results/csv_json/single_dynamic_gate42_video_manifest.json`
- `results/csv_json/single_dynamic_gate42_video_validation_report.csv`
- `results/csv_json/multi_static_dynamic_four_metrics_plot_data.csv`
- `results/csv_json/multi_static_dynamic_four_metrics_manifest.json`

The source package carries raw CSV and JSON metrics, replay manifests and validation summaries. Large MP4 video outputs live as external artifacts.

Source paths, sizes and descriptions for every artifact live in `artifacts/evaluation/results_manifest.json`.

## 7. Conclusion

`gate_graph_2d_minimal` retains the environment description, raw results and baseline comparison. A release refresh rechecks `results_manifest.json` against the shipped artifact set.
