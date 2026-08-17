# Observation ABI

The observation ABI is the field-level contract for a formal episode, defined in `schemas/observation_abi_v1.schema.json`. `rivermark_benchmark.abi` validates the same rules without `jsonschema`, so a reader can reject a payload before decoding it.

A file hash cannot establish these five facts, so the ABI fixes them explicitly:

1. **Timing.** A command is written before the simulation step; state is updated after the step; synchronized sensors are read after the state update; only then is the frame stored. Action fields use `command_before_step` semantics.
2. **Conventions.** World coordinates are right-handed `x_east_y_north_z_up`; body coordinates are FLU; camera optical coordinates are OpenCV right/down/forward; lengths are metres, angles are radians, quaternions are `wxyz`.
3. **Field metadata.** Every field declares dtype, shape, units, frame, agent and timestamp fields, missing-value policy, valid range, compression, and time semantics.
4. **Calibration.** Intrinsics, distortion model, and the closed extrinsic equation `T_world_camera = T_world_body * T_body_camera` are recorded. A sensor marked `unavailable` cannot be faked with a proxy.
5. **Hash binding.** A canonical JSON hash ties the ABI document to a manifest or release receipt. Changing a unit, shape, timing rule, or calibration changes the hash.

## Fidelity labels

ABI 1.1 requires every stream to declare a `fidelity` label — one of `simulator_consistent`, `noise_modeled`, or `hardware_calibrated` — plus a list of error sources it does *not* represent (lens distortion, rolling shutter, multipath, thermal drift, packet loss, hardware clock error). These are honest evidence labels, not a claim that the simulator is hardware-realistic.

Development readers accept ABI 1.0; formal packing and admission require 1.1 or newer.

## Compatibility

`assess_observation_abi_compatibility(producer, reader)` checks a reader against a producer: same major version, reader at least as new, and exact agreement on action timing, coordinate conventions, and stream semantics. An older reader, a major-version mismatch, or a semantic change is reported as incompatible. The result is a compatibility report only — it does not migrate bytes or grant admission.
