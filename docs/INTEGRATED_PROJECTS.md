# Integrated Project Sources

The projects below are maintained as self-contained snapshots inside this portfolio, so reviewers can inspect code, documentation, tests, and project-specific notices in place.

| Portfolio directory | Upstream repository | Imported revision | Notes |
|---|---|---|---|
| `aerial-soft-arm/` | [yubohann/AerialSoftArm](https://github.com/yubohann/AerialSoftArm) | `8a457c4` | Public sim-to-real snapshot: rigid-soft truss arm grasping in Isaac Lab plus MuJoCo sim2real tooling. Runtime `runs/` and third-party working trees stay local; the vendored Soft-Spiral subtree keeps its own MIT license. |
| `aerocity-bench/` | [yubohann/AeroCityBench](https://github.com/yubohann/AeroCityBench) | `6925461` | Public benchmark source snapshot. Local Isaac assets, scorer-private inputs, and generated evidence stay in local storage. |
| `hm3d-realised-qd/` | [yubohann/hm3d-realised-qd](https://github.com/yubohann/hm3d-realised-qd) | `475495a` | Public HM3D realised-QD source snapshot. HM3D assets, private scoring data, checkpoints, and runtime outputs stay local. |
| `robocon-handbook/` | [yubohann/robocon-handbook](https://github.com/yubohann/robocon-handbook) | `cd4bf40` | Public handbook snapshot: 37 chapters plus preface and master TOC on team engineering, with the Feishu sync tool under `tools/`. Internal working documents stay local. |

Each integrated directory keeps its own README, dependency metadata, source tree, tests, and release scope. The English `README.md` is the default GitHub display document, and `README.zh-CN.md` is the Chinese companion.
