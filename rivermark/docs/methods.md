# Methods

The benchmark separates the **data interface** from a **method claim**. A method may use the information profile it declares, emit actions to the real Isaac control loop, produce time-stamped public confirmation events, and be scored by the same private scorer.

## Supported families

The repository ships local reference implementations across the families the benchmark is designed to study.

| Family | Reference provided | Evidence still needed |
|---|---|---|
| Classical planning | Random, frontier, submodular, A*, MPC | A native Isaac result and leaderboard rank |
| RL | Actor-critic reference | A trained external-framework result |
| MARL | Shared decentralized actor-critic | MAPPO, skrl, and RLlib execution |
| Quality diversity | pyribs MAP-Elites trainer | Isaac task performance, pending a native rollout |
| VLM, VLN, VLA | Small local RGB-D grounder, grounded-language route, action-chunk network | OpenVLA, LLaVA, and a foundation-model result |
| World model | Local action-conditioned MPC reference | Dreamer and TD-MPC execution |
| External models | Strict adapters for selected families | A result with pinned weights, license, preprocessing, and an Isaac receipt |

## What counts as evidence

A method becomes an active baseline when one native Isaac run binds under a single receipt chain.

- the exact policy-observation projection identity and field allow-list
- the action-before-step trace and action-hold interval
- candidate events tied to opaque source-observation IDs
- scorer-owned visibility witnesses and v3 event scoring
- collision, separation, false-confirmation, timeout, and resource outcomes
- config, source, dependency, seed, and checkpoint identities
- every attempted task, including failures and aborts

Import statements, offline inference, and adapter classes describe interfaces, and execution evidence requires the bound native run.

## Intended comparisons

The design supports controlled ablations across information profiles, from state-only to RGB-D to RGB-D with LiDAR and IMU to message-aware policies, and across centralized and decentralized controllers under matched observation, action, communication, and compute budgets. Any comparison reports the exact profile, model revision, checkpoint identity, training seeds, compute budget, wall time, GPU memory, failure rate, and scoring split.
