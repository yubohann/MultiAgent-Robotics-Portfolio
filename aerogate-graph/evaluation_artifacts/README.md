# AeroGate Graph Evaluation Artifacts

Generated on: 2026-06-21

This directory collects reproducibility material for the gate-only evaluation set. It covers dynamic gates, single-agent and multi-agent scenarios, safety constraints, Graph-FlashSAC results, and classic planner baselines.

## Contents

- `reproducibility.md`: environment, commands, seeds, expected outputs, tests, and hash checks.
- `environment_setup.md`: environment setup notes.
- `demo_explanations.md`: notes for replay/video metadata retained as CSV/JSON.
- `results_manifest.json`: source path, purpose, size, and SHA256 for each retained artifact.
- `results/csv_json/`: retained CSV/JSON metrics and manifests.
- `report/gate_graph_2d_evaluation_report.md`: compact evaluation summary.

## Verification

```powershell
cd <aerogate_graph>
python -m pytest tests
```

Expected result: `8 passed`.
