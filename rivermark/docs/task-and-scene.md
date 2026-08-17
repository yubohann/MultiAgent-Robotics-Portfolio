# Search Task and Scene

## The search task

The active protocol, `citylite_t1_expert_coverage_v2`, collects **fixed-public-route expert coverage**: eight agents fly a predefined route through the city while the benchmark records their sensors and actions. Search itself is the T2 track: a closed loop where a policy observes, acts, and is scored against hidden targets.

The information boundary is the point of the benchmark:

- public route and state are visible to policies
- target coordinates, target IDs, and evaluator matching stay private
- a policy cannot earn credit by self-reporting a count; the evaluator privately matches its confirmations to the hidden targets

### Target placement

Targets are sampled by the evaluator, not by the public episode seed. Each target carries an anonymous slot label (for example `search_target_slot_002`). During sampling, a target must stay inside the camera frustum, clear of structural obstacles, and above a projected-area threshold for every probe in a counted witness window.

The current protocol activates **direct visibility only**: native geometry scans realized 4 of 4 targets for both direct-visible route pairings and 0 of 4 for both partial-visible pairings. Partial visibility is therefore unsupported rather than silently relabeled.

### Train/validation split

The protocol declares two cells:

| Split | Route family | Start | Target region | Visibility |
|---|---|---|---|---|
| Train | A | A | B | direct-visible |
| Validation | B | B | A | direct-visible |

The route families share no waypoint or segment but intersect at five points, so this is a **same-layout condition holdout**, not a spatially disjoint split or cross-scene generalization.

## The City-Lite scene

City-Lite is a task-focused composition of the high-fidelity Rivermark city. The runtime references only two approved roots: the city itself and the four task obstacles. It removes legacy mission drones, foliage, decoration, and anything that could dominate a sensor view, keeping the road network, terrain, buildings, structural props, and the task obstacles.

The scene contract is immutable and hash-bound. The static composition records roughly 20,000 active prims and 276 used USD layers, but these are composition facts, not claims about physics fidelity.

### Collision model

The native collision audit counts 4,807 drivable-surface colliders and 4 task-obstacle colliders; structural props have no mesh colliders. The capture runtime compensates by extracting conservative axis-aligned bounding boxes from structural geometry and creating one invisible static PhysX collision cube per box. This keeps vehicles out of covered volumes, but it is **conservative AABB collision**, not exact building-mesh collision. Doorways, concavities, and overhangs are not mesh-accurate yet, and an impact-response canary is still pending.

### Scale

The command volume spans roughly 92 m by 92 m horizontally and 5.25 m vertically — an urban block, not a city. There is one layout, so any generalization claim will require a second independently contracted layout.
