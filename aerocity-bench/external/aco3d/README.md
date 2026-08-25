# Locked ACO3D inspection-ordering candidate

This directory records the source lock for the MIT-licensed SII 2024 method
"Ant Colony Optimization for 3D Inspection Path Planning with Multiple
Unmanned Aerial Vehicles" (DOI `10.1109/SII58957.2024.10417512`).  The
upstream repository is locked at `c395f5b61f6746b2d39310dbc55a7ec3e1eae2d5`.
Its source is intentionally kept outside the AeroCityBench repository.

`tools/aco3d_g2i_process_adapter.py` is a separate-process, source-locked
translation of the published point-ordering loop.  It is not a copy of the
upstream MATLAB files.  The adapter accepts each drone's already frozen,
target-independent public sector and orders only the public 3-D inspection
cell positions within that sector.  CF2X control, collision handling,
observation validation, anonymous confirmations, return, and scoring remain
owned by the benchmark.

The source code's executable `main.m` computes one tour and does not contain
four-vehicle work allocation.  The public G2-I sector assignment consequently
does not become an ACO3D result.  This is a truthful input-semantics match for
the ordering subproblem, but not yet a substantive external G2-I result or a
Gate C closure.  In particular, the host currently lacks MATLAB and GNU
Octave, so no native-upstream equivalence claim is made.

On 2026-08-03, the source lock and public process completed a 12-step L0
calibration smoke on the current bounded-gimbal task boundary. It produced 48
four-vehicle execution receipts with zero collision, boundary, or planning
deadline failures. It deliberately ended before any observation, anonymous
confirmation, or return, and is recorded at
`reason/benchmark-external-methodology-audit-20260802/aco3d-g2i-l0-smoke-ancestor-00-20260803.json`.
It is therefore an ABI/safety smoke only, remains calibration-only and
`formal_score_eligible=false`, and does not close Gate C.

A 128-step follow-up on the same city (25.6 s of simulated time) also had zero
collision, boundary, and deadline failures, but reached no OBSERVE action and
covered zero public cells. Its `pass` field therefore means only execution
integrity; it is negative evidence against claiming route feasibility or search
performance and is retained for the next adapter review.

The subsequent complete 300-second calibration replay required return and
passed: all four vehicles returned, 46 of 59 public cells (132.7516 of
160.1449 square metres) were covered, two anonymous confirmations were issued,
and no collision, boundary, deadline, or terminal failure occurred. Evidence:
`reason/benchmark-external-methodology-audit-20260802/aco3d-g2i-l0-full-calibration-ancestor-00-20260803.json`.
This establishes one current-boundary L0 end-to-end calibration, not native
MATLAB execution, L1 CF2X evidence, a three-ancestor panel, or Gate C closure.

The same complete replay now passed on three independent calibration ancestors.
The machine-checked panel records a minimum public-area coverage of 74.26%, a
mean of 79.58%, four total anonymous confirmations, all vehicles returned, and
no collision, boundary, deadline, or terminal failure. It is available at
`reason/benchmark-external-methodology-audit-20260802/aco3d-g2i-l0-calibration-panel-20260803.json`.
The panel is deliberately marked `formal_score_eligible=false` and
`gate_c_eligible=false`: it establishes only the source-locked translation's
L0 calibration path.
