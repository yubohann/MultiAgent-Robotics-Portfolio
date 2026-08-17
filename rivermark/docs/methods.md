# Methods

The benchmark separates the **data interface** from a **method claim**. A method may use only the information profile it declares, emit actions to the real Isaac control loop, produce time-stamped public confirmation events, and be evaluated by the same private evaluator.

## Supported families

The repository ships local reference implementations across the families the benchmark is designed to study:

| Family | Reference provided | What it does not establish |
|---|---|---|
| Classical planning | Random, frontier, submodular, A*, MPC | A native Isaac result or leaderboard rank |
| RL | Actor-critic reference | A trained external-framework result |
| MARL | Shared decentralized actor-critic | MAPPO, skrl, or RLlib execution |
| Quality diversity | pyribs MAP-Elites trainer | Isaac task performance without a native rollout |
| VLM / VLN / VLA | Small local RGB-D grounder, grounded-language route, action-chunk network | OpenVLA, LLaVA, or a foundation-model result |
| World model | Local action-conditioned MPC reference | Dreamer or TD-MPC execution |
| External models | Fail-closed adapters for selected families | A result without pinned weights, license, preprocessing, and Isaac receipt |

## What counts as evidence

A method is not an active baseline until one native Isaac run binds, under a single receipt chain:

- the exact policy-observation projection hash and field allow-list
- the action-before-step trace and action-hold interval
- candidate events tied to opaque source-observation IDs
- evaluator-owned visibility witnesses and v3 event evaluation
- collision, separation, false-confirmation, timeout, and resource outcomes
- config, source, dependency, seed, and checkpoint hashes
- every attempted task, including failures and aborts

An import statement, offline inference, or an adapter class is not execution evidence.

## Intended comparisons

The design is built for controlled ablations: state-only versus RGB-D versus RGB-D plus LiDAR/IMU versus message-aware policies, and centralized versus decentralized controllers under matched observation, action, communication, and compute budgets. Any comparison must report the exact profile, model revision, checkpoint hash, training seeds, compute budget, wall time, GPU memory, failure rate, and evaluation split.
