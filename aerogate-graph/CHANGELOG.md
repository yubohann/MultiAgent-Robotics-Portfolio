# Changelog

All notable changes to AeroGate Graph are documented in this file.

## Unreleased

### Added

- Deterministic rollout reports with runtime provenance and an optional JSON evidence file.
- A locked core dependency environment, a reproducibility protocol, and CI artifact upload.
- A research overview, a reviewable evidence path, and an expanded architecture diagram.
- Public rollout reports now preserve final clearance, pair-separation, and formation-error
  diagnostics alongside reward and completion state.
- Normalize unbounded diagnostics to JSON `null` and reject non-finite evidence values.

### Changed

- Use the public rollout and reproducibility checks as the preferred lightweight validation path.
- Modernize package license metadata to remove a setuptools deprecation warning during builds.

## 0.2.0 - 2026-07-31

### Changed

- Rebuilt the repository as the complete AeroGate Graph research codebase.
- Added a runnable Python package, public CLI, English documentation, regression tests,
  and continuous integration.
- Consolidated source modules, assets, released checkpoints, and evaluation artifacts at the
  project root.

### Fixed

- Construct static gate-post collision maps from the default external gate layout.
- Include `step_count` in multi-agent runtime information for stable smoke-test summaries.
- Restore the Git LFS-managed gate and drone mesh assets from their BSD-3-Clause upstream
  and document their attribution.

### Migration

This is a repository-level breaking change. The project now uses the root-level module
layout documented in `README.md` and `docs/ARCHITECTURE.md`; use the `aerogate` package
as the supported public entry point.
