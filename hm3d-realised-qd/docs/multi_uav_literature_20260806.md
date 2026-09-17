# Multi-UAV Short Route, Backtracking, and Corridor Scheduling Literature Summary

## Conclusion

The 00803 low-efficiency trace has a clear cause in the candidate pool design.
The fix is to give the candidate pool the mature hierarchy of global access route,
local viewpoint, and executable trajectory, since corridor conflicts produced only
one-agent delays in place of a full serial schedule. Mature multi-UAV exploration
systems solve the same problem with:

1. A global or regional coverage route before local viewpoint refinement.
2. Viewpoints sampled in clear space near frontier clusters.
3. Task allocation that includes continuity, arrival time, observation overlap,
 communication risk, and switch cost.
4. Serialized departure, area switching, or role switching in bottleneck corridors.
5. Event-driven execution and lower replanning frequency.

## Verified Papers and Repositories

### RACER, T-RO 2023

Repo, https://github.com/Robotics-STAR-Lab/RACER

RACER decomposes space with hgrid, builds global MTSP and ACVRP coverage routes, and
refines local viewpoints. Paths are truncated to the current execution window and
dead ends are handled explicitly, keeping replanning frequency low.

### FUEL, RA-L 2021

DOI: 10.1109/LRA.2021.3051563

FUEL uses frontier clustering, global access routes, local viewpoints, and
B-spline trajectories. The paper treats low replanning frequency as important for
exploration speed, which supports long access routes over repeated short segments.

### C2-Explorer, IROS 2026

Repo, https://github.com/Robotics-STAR-Lab/C2-Explorer

C2-Explorer extends RACER, FUEL, and FALCON with task continuity and connectivity
constraints. Its public high-level code currently has a single-layer grid
limitation, and HM3D multi-layer support requires a migration step.

### FC-Planner, ICRA 2024

Repo, https://github.com/HKUST-Aerial-Robotics/FC-Planner

FC-Planner generates global coverage routes and viewpoints from skeletons and
emphasizes the global and local hierarchy. The reusable idea is to keep long routes in
front of the decision and refine only local viewpoints during execution.

### Fast Multi-Robot Decentralized Exploration of Forests, RA-L 2023

Repo, https://github.com/v4rl-ucy/fast_multi_robot_exploration

This work uses EXPLORER / GARBAGE_COLLECTOR role switching to break short-sighted
oscillation. We adopt the switching idea, since explicit role or area
switching is a useful behavior contract once legal forward tasks are exhausted.

### GVP-MREP, IROS 2024

Repo, https://github.com/NKU-MobFly-Robotics/GVP-MREP

GVP-MREP uses MR-DTG, local and global GVP switching, graph Voronoi allocation, and job
state. It shows that multi-agent allocation needs owner, distance, gain, remaining
time, and invalid-candidate deletion, replacing repeated greedy reassignment.

### VORL-EXPLORE, IROS 2026

Repo, https://github.com/21ning/VORL-EXPLORE

VORL-EXPLORE introduces execution fidelity into task allocation, directly targeting
oscillatory replanning caused by ignoring execution difficulty. This matches the
short horizontal-segment overshoot and frequent turning observed in 00803.

## Local Changes Applied

- build_public_candidate_pool now receives minimum_multi_agent_route_candidates=1.
- _traffic_reservation_variants now generates serial chains for three or more
 agents.
- The runner joint guard accepts transitive predecessor chains and direct
 pairs.
- _write_new and _write_new_json now use unique temporary files with cleanup.

## Remaining Risk

- A persistent global region route remains future work. Task reservation only
 partially expresses route continuity.
- frontier_3d scoring will incorporate VORL-EXPLORE execution fidelity in a later
 iteration.
- Serial corridor chains increase makespan. Formal results must report decision
 count, effective motion time, and serial waiting time together.

This round stays at short smoke scale for PhysX runs.
