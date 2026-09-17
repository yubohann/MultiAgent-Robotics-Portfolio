# Archived Modules

Modules in this directory live outside the core Rivermark benchmark pipeline. The
repository keeps them for reference and recovery, outside the installed package
and outside the CPU test suite.

## Why archived

These modules are development or demonstration utilities and external reference
math. The active package serves the core pipeline, capture, validate, admit,
score, and release, and keeps a smaller maintenance surface.

## Contents

| Module | What it was | Reason archived |
|---|---|---|
| `demo.py` | MP4 demo rendering (`Mp4Writer`) | Standalone demonstration output |
| `clean_room_smoke.py` | Second-machine clean-clone smoke | P0-D milestone open, entry point removed |
| `omnidrones_rate_controller.py` | OmniDrones rate-controller math reference | External reference material |
| `external_sources.py` | Snapshot audit of external robotics ecosystems | Peripheral bookkeeping |
| `evaluator_service.py` | Local authenticated scorer-service prototype | Local prototype, marked undeployed |
| `isaac_transfer_validate.py` | Independent validation of the SB3 control transfer | Development-only demonstration |

> Note. `isaac_transfer.py` lives in the active package. The core capture CLI and
> the T2 modules import it, so it is a live dependency.

Tests moved with them.

| Test | Covered |
|---|---|
| `test_clean_room_smoke.py` | clean-room smoke |
| `test_omnidrones_rate_controller.py` | OmniDrones reference |
| `test_external_sources.py` | external snapshot audit |
| `test_evaluator_service.py` | scorer service prototype |
| `test_isaac_transfer_validate.py` | SB3 transfer validation |
| `test_video.py` | video transcoding and `Mp4Writer` |

## Restore

To restore a module, move the `.py` file back to `src/rivermark_benchmark/`, move
its test back to `tests/`, re-add the `[project.scripts]` entry point where one
exists, and re-add the lazy export in `src/rivermark_benchmark/__init__.py`.
