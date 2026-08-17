# Changelog

## Unreleased

- Added a real Isaac Lab eight-CF2X Search3D capture path with online RGB-D,
  semantic segmentation, MultiMeshRayCaster LiDAR, IMU, contact, actuator,
  communication, and high-level-action streams.
- Added an independent, import-free Isaac capture validator and a fail-closed
  public candidate packer with evaluator-truth separation.
- Added a closed no-radar multisensor information profile.
- Added release gates, dataset documentation, checksums, and CI scaffolding.
- Corrected the native T2 physical trace ABI to distinguish a held policy
  decision's pre-command time from each physics step's actuator-command time.
- Deduplicate repeated RGB-D/semantic views by their capture-local anonymous
  semantic slot before event serialization; the slot is never released in a
  candidate event or provided to the private evaluator.

The current release remains a research preview until a clean-source Isaac
episode, an independent validation receipt, and an operator-approved artifact
allowlist are present. No hardware-radar, real-flight, or foundation-model
result is implied.
