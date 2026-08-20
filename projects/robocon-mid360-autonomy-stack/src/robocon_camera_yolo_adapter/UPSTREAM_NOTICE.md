# Source and Adaptation Notice

The adapter reuses target semantics and an inference-interface pattern from a
private `Camera_Yolo` integration. No detector source, model weight, SDK
binary, calibration, or runtime configuration was copied into this workspace.
The adapter accepts an explicit runtime integration path and adapts per-frame
detections to a versioned JSON observation. Private source and device-specific
parameters remain outside this public package.
