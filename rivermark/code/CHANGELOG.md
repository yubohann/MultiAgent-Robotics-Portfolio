# Changelog

## Unreleased

- Added a real Isaac Lab eight-CF2X Search3D capture path with online RGB-D,
  semantic segmentation, MultiMeshRayCaster LiDAR, IMU, contact, actuator,
  communication, and high-level-action streams.
- Added an independent, import-free Isaac capture validator and a strict
  public candidate packer with scorer-truth separation.
- Added a closed multisensor information profile that excludes radar.
- Added dataset documentation and CI scaffolding.
- Corrected the native T2 physical trace ABI to distinguish a held policy
  decision's pre-command time from each physics step's actuator-command time.
- Deduplicate repeated RGB-D and semantic views by their capture-local anonymous
  semantic slot before event serialization, keeping the slot within the capture.

The current release is a research preview. A clean-source Isaac episode and an
independent validation receipt mark the next milestone, and hardware-radar,
real-flight, and foundation-model results stand outside this preview.
