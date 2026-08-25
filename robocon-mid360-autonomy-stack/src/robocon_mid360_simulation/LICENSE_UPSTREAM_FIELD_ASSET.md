# Upstream notice: public ROBOCON25 Gazebo field asset

The files under `models/` and `worlds/robocon25_candidate.world` were copied
from `fynngwu/gazebo_simulation`, commit `58eca9828ade74a7939c297b68baddf836d8554d`.
The repository declares the MIT License (Copyright (c) 2025 wufy). Its field
layout visually matches Figure 2 of the official ABU ROBOCON 2025 rulebook.
The following rulebook values were independently checked against
`references/official/ABU_ROBOCON_2025_Rulebook_20240814.pdf`, SHA-256
`cd57640abf2625f9983e9768cd8fad7bb87d7f716b51d1222a3e0caae9d30eb3`:

- 15 m by 8 m playing area, with a 0.10 m high and 0.05 m wide fence;
- 1.80 m by 1.05 m backboards, 2.43 m basket height, and a 0.450--0.459 m rim;
- Size 7 basketball mass and circumference. The copied model uses 0.60 kg and
  a 0.12 m radius, which is within those stated ranges.

The upstream collision rim height of 3.03 m was changed to 2.43 m, and both
backboard meshes were lowered by 0.60 m in this adapted copy. This establishes
a `rulebook_geometry_subset_verified` asset. The Rulebook does not provide a
numeric 3-point arc definition in the text, so zone classification remains a
separate validated geometry task.

The original MIT license text is preserved in the source repository under
`repos/2026_fynngwu_gazebo_simulation/LICENSE`.
