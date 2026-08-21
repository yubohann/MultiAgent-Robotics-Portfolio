# Architecture

AeroGate Graph is organized around a simulator-independent 2D core. The core tracks
fixed-height motion and circular gate-post collision geometry; higher layers compose it
into single-agent and multi-agent tasks. PyTorch training and Isaac Lab rendering are
adapters rather than requirements for basic environment operation.

The source diagram is [ARCHITECTURE.mmd](ARCHITECTURE.mmd). Render it with a Mermaid-aware
Markdown viewer or Mermaid CLI when a bitmap or SVG artifact is needed.

## Module Responsibilities

| Area | Responsibility | Runtime dependency |
| --- | --- | --- |
| shared/core | Kinematics, collision checks, fixed-height invariants, dynamic gates | NumPy where required |
| single_gate | One-drone environment, graph observations, rewards, and Graph-SAC | NumPy; PyTorch for training |
| multi_gate | Team environment, slots, planning, action shields, and Graph-MASAC | NumPy; PyTorch for training |
| gate_density_* | Gate-density curricula and benchmark/evaluation drivers | NumPy; PyTorch for learned policies |
| assets | Gate and drone USD/URDF data plus layout descriptions | Git LFS for USD |
| shared/visualization | Isaac Lab scene and replay adapters | Isaac Lab, optional |
| aerogate | English public API, CLI, scenario registry, and smoke commands | NumPy |

## Data Flow

1. A scenario config defines bounds, gate geometry, motion limits, rewards, and observation size.
2. The environment creates a collision map and advances fixed-height kinematic state from actions.
3. Multi-agent tasks compute a route, virtual-structure slots, team-separation metrics, and
   safety-shielded controls.
4. Graph observations expose agents, route context, obstacles, and active-agent masks to policies.
5. Training/evaluation modules write checkpoints and metrics outside source directories.
6. The public CLI can produce seeded, JSON-safe rollout reports; CI stores one as a reviewable
   artifact.
7. Optional Isaac Lab adapters render the same task-level geometry; they do not replace core tests.

## Evidence Path

The source graph distinguishes runtime dependencies from evidence-producing paths. The
public `smoke` and `reproduce` commands exercise the NumPy core; the latter compares two
rollouts per explicit seed and records runtime provenance. Learning scripts and retained
checkpoints sit outside that contract because they require larger budgets and optional
PyTorch or simulator dependencies.

This separation makes it possible to review environment behavior, safety metrics, and
configuration changes even when a reviewer cannot run GPU training or Isaac Lab.

## Public Boundary

The aerogate package is the supported import and CLI namespace. Existing research modules
remain available for compatibility and reproducibility, but new callers should prefer the
public scenario helpers. Core code must not import Isaac Lab or PyTorch at module import time.

For research questions, reporting guidance, and explicit non-claims, see
[RESEARCH_OVERVIEW.md](RESEARCH_OVERVIEW.md).
