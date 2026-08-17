# Archived Modules

Modules moved here are not part of the core Rivermark benchmark pipeline. They are
kept for reference and easy recovery. They are **not installed** and not covered by
the CPU test suite.

## Why archived

These modules are development/demonstration utilities or external-reference math.
Keeping them in the active package added maintenance surface without serving the
core pipeline (capture → validate → admit → evaluate → release).

## Contents

| Module | What it was | Reason archived |
|---|---|---|
| `demo.py` | MP4 demo rendering (`Mp4Writer`) | Demonstration output, not pipeline |
| `clean_room_smoke.py` | Second-machine clean-clone reproduction smoke | P0-D not yet completed; entry point removed |
| `omnidrones_rate_controller.py` | OmniDrones rate-controller math reference | External reference, not Rivermark output |
| `external_sources.py` | Snapshot audit of external robotics ecosystems | Peripheral bookkeeping |
| `evaluator_service.py` | Local authenticated evaluator-service prototype | Explicitly marked undeployed prototype |
| `isaac_transfer_validate.py` | Independent validation of the SB3 control transfer | Development-only demonstration |

> Note: `isaac_transfer.py` was restored to the active package — it is imported by
> the core capture CLI and the T2 modules, so it is a live dependency, not an
> archived module.

Tests moved with them:

| Test | Covered |
|---|---|
| `test_clean_room_smoke.py` | clean-room smoke |
| `test_omnidrones_rate_controller.py` | OmniDrones reference |
| `test_external_sources.py` | external snapshot audit |
| `test_evaluator_service.py` | evaluator service prototype |
| `test_isaac_transfer_validate.py` | SB3 transfer validation |
| `test_video.py` | video transcoding + `Mp4Writer` |

## Restore

To restore a module, move the `.py` file back to `src/rivermark_benchmark/`, move
its test back to `tests/`, re-add the `[project.scripts]` entry point (if any), and
re-add the lazy export in `src/rivermark_benchmark/__init__.py`.
