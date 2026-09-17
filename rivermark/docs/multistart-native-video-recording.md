# Multi-Start Native Isaac Video Recording

Rivermark's City-Lite contract defines two public route families. Each family contains eight literal CF2X start poses, so one episode shows the swarm spread across the map. Family A covers the train condition, and family B is the mirrored validation condition.

## Plan the batch

Run from a clean checkout.

```powershell
$matrix = Join-Path $env:TEMP 'rivermark-multistart-video-matrix.json'
python tools/plan_multistart_video_matrix.py --output $matrix
Get-Content $matrix
```

The output contains eight initial world poses and the complete waypoint route for every episode, with placeholders for the external CF2X USD, City-Lite contract, scorer-private manifest, retention directory, IsaacLab source, and sensor-smoke receipt.

## Record

Replace the placeholders in each matrix row with paths from the local asset package, then run the command through the IsaacLab Python interpreter. The private manifest must match the cell's route family and stay outside both the repository and the episode output directory.

```powershell
& C:\Users\Administrator\IsaacLab\python\python.exe `
  -m rivermark_benchmark.isaac_capture `
  --output-dir E:\rivermark-native-video\train\episode-0000 `
  --drone-usd <external-cf2x.usd> `
  --scene-contract <external-city-lite-contract.json> `
  --collection-protocol .\code\config\collection_protocol.citylite_t1_expert_coverage_v2.json `
  --collection-cell-id train-citylite-direct-v2 `
  --collection-episode-index 0 `
  --evaluator-private-manifest <external-private-manifest.json> `
  --evaluator-private-manifest-retention-root E:\rivermark-private-retention `
  --runtime-lock .\code\config\isaac_runtime.windows-5.1.json `
  --isaaclab-source C:\Users\Administrator\IsaacLab\source `
  --sensor-physics-smoke-receipt <external-isaac-smoke-receipt.json> `
  --control-mode fixed_public_route `
  --headless
```

The capture's native RGB frames, depth, semantic labels, pose, actions, and runtime receipt are the source of truth. A later encoder may combine the native RGB frames with a transparent diagnostics panel, and the Isaac render stays the image source.

## Encode Native Frames

After a capture passes its receipt and independent validation, encode the native RGB archive directly.

```powershell
$env:PYTHONPATH = (Resolve-Path .\code\src)
python .\tools\encode_native_video.py `
  --capture-dir E:\rivermark-native-video\train\episode-0000 `
  --view overview `
  --output E:\rivermark-native-video\videos\train-citylite-direct-episode-0000-overview.mp4 `
  --fps 20
```

Use `--view onboard` for the multi-camera onboard archive. The encoder rejects missing archives, wrong RGB shape or dtype, a missing FFmpeg, and partial output, and writes a SHA-256-bound `.manifest.json` beside the MP4.

## Acceptance

A video is accepted when its episode receipt and native payload pass these checks.

- `route_family_id`, cell, seed, and eight initial poses are hash-bound.
- the first retained frame shows the City-Lite scene and all visible swarm members as native renders.
- the waypoint route is executed by the physical CF2X runtime.
- timestamps are monotonic and frame count agrees with the native capture.
- camera pose closure, RGB, depth, and semantic freshness, LiDAR and IMU synchronization, and collision and clearance gates pass.
- the final MP4 hash is recorded alongside the capture receipt and Git commit.
- family A and family B videos are labelled separately, and private targets and scorer coordinates stay out of the public video and manifest.

The local NVIDIA assets, private manifests, and generated videos live in the operator's local evidence store under the applicable NVIDIA and asset-package terms.
