# Multi-Start Native Isaac Video Recording

Rivermark's City-Lite contract already defines two public route families. Each
family contains eight literal CF2X start poses, so one episode shows a swarm
initialised across the map rather than eight drones placed at one shared point.
Family A covers the train condition; family B is the mirrored validation
condition.

## Plan the batch

Run from a clean checkout:

```powershell
$matrix = Join-Path $env:TEMP 'rivermark-multistart-video-matrix.json'
python tools/plan_multistart_video_matrix.py --output $matrix
Get-Content $matrix
```

The output contains eight initial world poses and the complete waypoint route
for every episode. It deliberately contains placeholders for the external
CF2X USD, City-Lite contract, evaluator-private manifest, retention directory,
IsaacLab source, and sensor-smoke receipt.

## Record

Replace the placeholders in each matrix row with paths from the local asset
package, then run the command through the IsaacLab Python interpreter. The
private manifest must match the cell's route family and stay outside both the
repository and the episode output directory:

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

The capture's native RGB frames, depth, semantic labels, pose, actions, and
runtime receipt are the source of truth. Do not create a video by drawing a
trajectory over a blank canvas. A later encoder may combine the native RGB
frames with a transparent diagnostics panel, but it may not replace the Isaac
render.

## Encode Native Frames

After a capture passes its receipt and independent validation, encode the
native RGB archive directly:

```powershell
$env:PYTHONPATH = (Resolve-Path .\code\src)
python .\tools\encode_native_video.py `
  --capture-dir E:\rivermark-native-video\train\episode-0000 `
  --view overview `
  --output E:\rivermark-native-video\videos\train-citylite-direct-episode-0000-overview.mp4 `
  --fps 20
```

Use `--view onboard` for the multi-camera onboard archive. The encoder refuses
missing archives, wrong RGB shape/dtype, missing FFmpeg, and partial output; it
writes a SHA-256-bound `.manifest.json` beside the MP4.

## Acceptance

Accept a video only when its episode receipt and native payload pass all of the
following checks:

- `route_family_id`, cell, seed, and eight initial poses are hash-bound;
- the first retained frame shows the City-Lite scene and all visible swarm
  members, with no camera-only placeholder;
- the waypoint route is executed by the physical CF2X runtime, not a post-hoc
  trajectory animation;
- timestamps are monotonic and frame count agrees with the native capture;
- camera pose closure, RGB/depth/semantic freshness, LiDAR/IMU sync, collision
  and clearance gates pass;
- the final MP4 hash is recorded alongside the capture receipt and Git commit;
- family A and family B videos are labelled separately; no private targets or
  evaluator coordinates appear in the public video or manifest.

The repository does not include the local NVIDIA assets, private manifests, or
generated videos. Those remain in the operator's local evidence store under the
applicable NVIDIA and asset-package terms.
