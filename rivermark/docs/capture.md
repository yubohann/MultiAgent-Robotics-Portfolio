# Native Isaac Capture

`rivermark_benchmark.isaac_capture` runs the native collection: one long-lived Isaac AppLauncher process builds a fresh stage with the approved City-Lite roots and eight physical CF2X vehicles. This section covers how a capture is prepared and what happens at each physics step.

## The step loop

At every physical step the collector:

1. writes the physical command
2. steps the simulation
3. updates physics/control state
4. reads synchronized RGB, depth, semantic, LiDAR, IMU, contact, and extrinsics
5. persists the frame, action, state, timestamp, and agent ID

Command-before-step timing is what makes causal evaluation possible.

## Before launch

A capture is not allowed to start until a set of preflight checks pass: clean source tree, disk reservation, Windows system-commit guard, GPU/driver capacity, the City-Lite contract, the CF2X asset hash, the runtime lock, and the evaluator-private manifest binding. A repository-wide lease prevents two AppLaunchers running at once.

The collection protocol, cell, and episode index are bound before Isaac starts:

```powershell
rivermark-isaac-capture --output-dir E:\rivermark-runs\run-001 `
  --collection-protocol .\collection-protocol.json `
  --collection-cell-id train-route-0 `
  --collection-episode-index 1 `
  --evaluator-private-manifest E:\rivermark-private\evaluator.json `
  --runtime-lock .\config\isaac_runtime.windows-5.1.json `
  --headless
```

The capture resolves the protocol once, stores a path-free binding in the receipt, and derives the episode seed deterministically from protocol + cell + index. A runtime seed that differs from the bound seed is rejected.

## Fail-closed behavior

A capture is discarded when any gate fails: missing or stale sensor frames, pose-closure error, unresolved scene references, visual intrusion, unsafe obstacle proximity, route-contract violation, private-truth leakage, insufficient disk, runtime-lock mismatch, or a resource-guard breach. Failed artifacts stay outside the formal dataset and are recorded in a redacted failure ledger.

## The evidence path

- **Overview witness**: a fixed public-world camera pose, checked live at every retained frame. The sparse archive keeps only RGB, native semantics, and pose at the first, every tenth, and final retained frame — no overview depth on disk.
- **Onboard gates**: every RGB-D and LiDAR sample passes a visual-intrusion gate. Close meshes, foliage, near-surface depth domination, or anomalous near-range LiDAR fail the capture; no cropping or good-frame selection can repair it.
- **Video**: overview and composite MP4s may be encoded only after an independent validator passes, then must be fully decoded and sampled at the first, 25%, 50%, 75%, and final frames. A playable video alone does not establish a valid episode.

## Private manifest retention

For every future `fixed_public_route` capture, the operator supplies an existing private retention directory outside both the repository and the capture. The collector snapshots the exact evaluator manifest there under its SHA-256 filename and loads the retained snapshot rather than the mutable source. The public receipt records only the retention kind, hash, and byte count — never the private root, path, or manifest bytes.
