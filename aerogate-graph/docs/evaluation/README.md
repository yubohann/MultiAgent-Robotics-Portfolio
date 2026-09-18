# Gate Graph 2D Minimal Scoring Artifacts

Generated on 2026-06-21

This directory collects deterministic replay material for the gate-only scoring set. It covers dynamic gates, single-agent and multi-agent scenarios, safety constraints, Graph-FlashSAC results, and classic planner baselines.

## Contents

- `reproducibility.md` documents environment, commands, seeds, expected outputs, and tests.
- `environment_setup.md` documents the environment setup.
- `demo_explanations.md` documents replay and video metadata retained as CSV and JSON.
- `../../artifacts/evaluation/results_manifest.json` lists source path, purpose, and size for each retained artifact.
- Retained CSV and JSON metrics and manifests are external artifacts recorded by relative path in that manifest.
- `gate_graph_2d_evaluation_report.md` holds the compact scoring summary.

## Verification

```powershell
cd <gate_graph_2d_minimal>
python -m pytest tests
```

