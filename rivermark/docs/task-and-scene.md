# Search Task and Scene

## The Search Task

The active protocol, `citylite_t1_expert_coverage_v2`, collects **fixed-public-route expert coverage**, where eight agents fly a predefined route through the city while the benchmark records their sensors and actions. Search itself is the T2 track, a closed loop where a policy observes, acts, and is scored against hidden targets.

The information split is the point of the benchmark.

- public route and state are visible to policies
- target coordinates, target IDs, and scorer matching stay private
- credit follows private scorer matches between confirmations and hidden targets

### Target Placement

Targets are sampled by the scorer, separately from the public episode seed. Each target carries an anonymous slot label, for example `search_target_slot_002`. During sampling, a target must stay inside the camera frustum, clear of structural obstacles, and above a projected-area threshold for every probe in a counted witness window.

The current protocol activates **direct visibility only**. Native geometry scans realized 4 of 4 targets for both direct-visible route pairings and 0 of 4 for both partial-visible pairings, so partial visibility stays out of the counted conditions.

### Train and Validation Split

The protocol declares two cells.

| Split | Route family | Start | Target region | Visibility |
|---|---|---|---|---|
| Train | A | A | B | direct-visible |
| Validation | B | B | A | direct-visible |

The route families share zero waypoints and segments and intersect at five points, so this is a **same-layout condition holdout**. Spatially disjoint splits and cross-scene generalization remain future work.

## The City-Lite Scene

City-Lite is a task-focused composition of the high-fidelity Rivermark city, built around two approved roots, the city itself and the four task obstacles. The kept layers cover the road network, terrain, buildings, structural props, and the task obstacles.

The scene contract is immutable and identity-bound. The static composition records roughly 20,000 active prims and 276 used USD layers.

### Collision Model

The native collision audit counts 4,807 drivable-surface colliders and 4 task-obstacle colliders, with structural props relying on proxies. The capture runtime extracts conservative axis-aligned bounding boxes from structural geometry and creates one invisible static PhysX collision cube per box. Vehicles stay outside covered volumes, while mesh-accurate doorways, concavities, and overhangs and an impact-response canary mark the next milestones.

### Scale

The command volume spans roughly 92 m by 92 m horizontally and 5.25 m vertically, a single urban block. Cross-scene generalization requires a second independently contracted layout.
