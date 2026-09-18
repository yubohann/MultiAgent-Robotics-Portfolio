# HM3D Realised-QD

<p align="center">
  <img src="assets/demos/hm3d-scene-1.gif" alt="HM3D multi-UAV exploration scene" width="78%" />
</p>

**Outcome-grounded quality-diversity and reinforcement learning for target-free multi-UAV exploration in HM3D-derived 3D scenes.**

Four CF2X quadrotors explore an unknown indoor scan with public sparse-range sensing. The fleet fuses a shared belief, picks team plans from a common candidate pool, and executes them under real Isaac and PhysX dynamics. Behavioral diversity comes only from execution receipts, so the archive records what flew, and every method faces the same observations, safety contracts and physical time budget.

**Status.** v0.1.0 research snapshot, 2026-08-08. The realised-QD selector holds a verified P10 component result across 42 real episodes, with RB-SF-SAC and RFG integration in progress.

## My Role

`pyproject.toml` lists Bohan Yu as the project author, and this directory records the work behind that authorship.

- **Research design.** Formalised the target-free multi-UAV exploration task, the shared sensing, communication and safety contracts, and the `Explored-Free-Flight-Volume-AUC_time` metric, then pre-registered the hypotheses and ablation chain in the [method design](docs/method-design-realised-qd-rfg-rb-sf-sac-2026-08-08.md).
- **World model and QD learning implementation.** Built the sparse occupancy belief, the frontier, route-access and observation candidate generator, the 3D realised-QD archive and selector with intent-audit fallback (`src/realised_qd/runtime/hm3d_realised_qd.py`, `src/realised_qd/archives/`), and the RB-SF-SAC, masked-PPO and replay stack (`src/realised_qd/learning/`).
- **Exploration and outcome pipeline.** Implemented the CF2X/PhysX executor, safety ledger, transit-timing calibration, persistent collection and train-outcome dataset builders (`src/realised_qd/runtime/`, `scripts/exploration/`, `scripts/mechanism/`).
- **Experiment execution.** Ran the 42 formal P10 episodes, the 00626 mechanism ablation and the QD replay calibration on real Isaac and PhysX; the realised-QD component led all five baselines on scene 00626 and closed the no_qd < planned_qd < realised_qd gradient.
- **Engineering and documentation.** Fixed the seven recorded pipeline defects with regression tests ([results summary](docs/experiment-results-2026-08-08.md), section 5), and wrote the method specification, protocol configs and the test suites under `tests/`.

## Core Design

- The task is target-free online exploration of HM3D-derived indoor scenes, with `Explored-Free-Flight-Volume-AUC_time` as the primary metric under a shared CF2X, communication, safety and physical-time contract.
- Public sparse-range outcomes build a sparse occupancy belief. Frontier, route-access and observation candidates are generated from that belief, admitted by a static clearance guard and a joint team guard, and capped at a frozen pool of 16.
- The realised-QD archive keeps behavioral elites in a 3D descriptor space of 64 cells, with descriptors and quality that derive from real execution receipts.
- RB-SF-SAC adds a recurrent, belief-state, shared-frontier SAC layer that ranks the same team candidates.
- RFG gates fragment reuse on completed, provenance-clean execution, credits measured volume gains, and revokes credit when a later execution rewrites the fragment.
- The PhysX executor runs every selected manifest with CF2X dynamics at a 40 s physical budget, and each episode emits receipts, safety ledgers and outcome hashes.

## Verified So Far

- 42 formal P10 episodes on real Isaac and PhysX closed with zero collisions, zero flight-limit excursions, zero separation violations and zero failed fragments, with every transit completed.
- On the 325.17 m³ train scene 00626, the realised-QD component scored 0.0984 AUC and led all five baselines, random 0.0915, frontier_3d 0.0917, auction 0.0838, gvp_mrep_port 0.0891 and single_rl 0.0932, with the highest coverage 0.1418 and the longest path at 28.8 m.
- On the 145.13 m³ train scene 00459, the same component scored 0.3436 AUC, second to frontier_3d at 0.3854.
- The 00626 mechanism ablation forms a gradient from planned intent to realised receipts, no_qd 0.0909, planned_qd 0.0926, realised_qd 0.0984.
- QD replay calibration repeated nine of twelve intent-mode executions from identical public resets.

## Quick Start

```powershell
uv sync --extra dev --extra rl --extra hm3d
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m ruff check src tests scripts
```

Isaac and PhysX runs go through `scripts/run_isaac_python.ps1` with a verified IsaacLab interpreter and `REALISED_QD_CF2X_USD` set to the local CF2X USD asset. The public source tree holds code, contracts and documentation. HM3D assets, converted meshes, checkpoints and raw run outputs stay in the local workspace.

## Documentation

- [Documentation index](docs/README.md), the current design, results and notes set.
- [Method design](docs/method-design-realised-qd-rfg-rb-sf-sac-2026-08-08.md), the realised-QD, RB-SF-SAC and RFG contract.
- [P10 main table](docs/p10-main-results-2026-08-08.md), the component result table and ablation.
- [Results summary](docs/experiment-results-2026-08-08.md), the frozen research record behind the numbers above.

## License

The source tree retains its project-specific release terms. Third-party code and assets retain their own licenses.
