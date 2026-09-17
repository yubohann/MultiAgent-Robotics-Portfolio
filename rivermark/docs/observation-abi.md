# Observation ABI

The observation ABI is the field-level contract for a formal episode, defined in `schemas/observation_abi_v1.schema.json`. `rivermark_benchmark.abi` validates the same rules in pure Python, so a reader can reject a payload before decoding it.

A file identity covers bytes, while the ABI fixes five facts about meaning.

1. **Timing.** A command is written before the simulation step, state is updated after the step, synchronized sensors are read after the state update, and the frame is stored last. Action fields use `command_before_step` semantics.
2. **Conventions.** World coordinates are right-handed in `x_east_y_north_z_up` order, body coordinates use FLU, and camera optical coordinates use OpenCV right, down, and forward axes. Lengths are metres, angles are radians, and quaternions are `wxyz`.
3. **Field metadata.** Every field declares dtype, shape, units, frame, agent and timestamp fields, missing-value policy, valid range, compression, and time semantics.
4. **Calibration.** Intrinsics, distortion model, and the closed extrinsic equation `T_world_camera = T_world_body * T_body_camera` are recorded. A sensor marked `unavailable` keeps that declared status, and proxies are rejected.
5. **Identity binding.** A canonical JSON identity ties the ABI document to a manifest or release receipt. Changing a unit, shape, timing rule, or calibration changes the identity.

## Fidelity Labels

ABI 1.1 requires every stream to declare a `fidelity` label, one of `simulator_consistent`, `noise_modeled`, or `hardware_calibrated`, plus a list of error sources outside its model, such as lens distortion, rolling shutter, multipath, thermal drift, packet loss, and hardware clock error. Fidelity labels describe evidence provenance.

Development readers accept ABI 1.0. Formal packing and admission require 1.1 or newer.

## Compatibility

`assess_observation_abi_compatibility(producer, reader)` checks a reader against a producer, covering a matching major version, a reader at least as new, and exact agreement on action timing, coordinate conventions, and stream semantics. An older reader, a major-version mismatch, or a semantic change is reported as incompatible. A compatibility report describes the pairing and leaves bytes and admission decisions to their own tools.
