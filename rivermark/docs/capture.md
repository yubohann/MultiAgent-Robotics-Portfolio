# Native Isaac Capture

`rivermark_benchmark.isaac_capture` runs the native collection. One long-lived Isaac AppLauncher process builds a fresh stage with the approved City-Lite roots and eight physical CF2X vehicles. This page covers capture preparation and each physics step.

## The Step Loop

At every physical step the collector performs this sequence.

1. writes the physical command
2. steps the simulation
3. updates physics and control state
4. reads synchronized RGB, depth, semantic, LiDAR, IMU, contact, and extrinsics
5. persists the frame, action, state, timestamp, and agent ID

Command-before-step timing makes causal scoring possible.

## Before Launch

A capture starts after preflight checks pass, covering a clean source tree, disk reservation, the Windows system-commit guard, GPU and driver capacity, the City-Lite contract, the CF2X asset hash, the runtime lock, and the scorer-private manifest binding. A repository-wide lease allows one AppLauncher at a time.

The collection protocol, cell, and episode index are bound before Isaac starts.

```powershell
rivermark-isaac-capture --output-dir E:\rivermark-runs\run-001 `
  --collection-protocol .\collection-protocol.json `
  --collection-cell-id train-route-0 `
  --collection-episode-index 1 `
  --evaluator-private-manifest E:\rivermark-private\evaluator.json `
  --runtime-lock .\config\isaac_runtime.windows-5.1.json `
  --headless
```

The capture resolves the protocol once, stores a path-free binding in the receipt, and derives the episode seed deterministically from the protocol, cell, and episode index. A runtime seed that differs from the bound seed is rejected.

## Fail-Closed Behavior

A capture is discarded when any gate fails, covering missing or stale sensor frames, pose-closure error, unresolved scene references, visual intrusion, unsafe obstacle proximity, route-contract violations, private-truth leakage, insufficient disk, runtime-lock mismatch, and resource-guard breaches. Failed artifacts stay outside the formal dataset and enter a redacted failure ledger.

## The Evidence Path

- **Overview witness**. A fixed public-world camera pose, checked live at every retained frame. The sparse archive keeps RGB, native semantics, and pose at the first, every tenth, and final retained frame.
- **Onboard gates**. Every RGB-D and LiDAR sample passes a visual-intrusion gate. Close meshes, foliage, near-surface depth domination, or anomalous near-range LiDAR fail the capture at this gate.
- **Video**. Overview and composite MP4s encode after an independent validator passes, and each file is fully decoded and sampled at the first, 25%, 50%, 75%, and final frames. Episode validity rests on the validator pass.

## Private Manifest Retention

For every future `fixed_public_route` capture, the operator supplies an existing private retention directory outside both the repository and the capture. The collector snapshots the exact scorer manifest there under its SHA-256 filename and loads the retained snapshot in place of the mutable source. The public receipt records the retention kind, hash, and byte count, while the private root, path, and manifest bytes remain local.
