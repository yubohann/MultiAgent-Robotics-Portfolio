# Changelog

## 0.1.0 - 2026-07-31

- Rebuilt the former redacted portfolio snapshot as a complete source-available fraud ML engineering repository.
- Packaged the hybrid SplitGNN and Transformer training code under `src/fraud_ml_engineering`.
- Added reproducible path conventions, dependency profiles, documentation, tests, data exclusion rules, and third-party notices.
- Added a dependency-light validator, repository identity checks, package build verification, and a Python 3.10/3.12 GitHub Actions quality matrix.
- Refactored the CLI into explicit argument and API-boundary helpers, standardized project-owned documentation, and renamed generated artifact prefixes to `hybrid_fraudgraph`.
- Removed the prior redacted placeholder structure and excluded datasets, checkpoints, caches, and generated outputs from version control.
- Added dependency-light run manifests, an experiment catalog, and an auditable comparison-report schema for evidence-backed research workflows.
- Added installed-wheel verification to CI and contribution checks so the distributable package is exercised outside the source tree.
- Consolidated shared experiment-runner protocol utilities and replaced the legacy historical-best report generator with an explicit-input comparison tool that rejects mixed revisions and temporary artifacts.
