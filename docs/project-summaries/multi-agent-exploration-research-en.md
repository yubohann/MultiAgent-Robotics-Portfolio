# Multi-Robot Cooperative Exploration and Navigation Research

**Research Lead and Independent Developer** | **Oct. 2025 – Present**  
**Research focus:** Cooperative exploration and target search in partially observable 3D environments. The work combines local perception, spatial connectivity, and execution feedback to coordinate multiple robots.

**Technical stack:** Python; C++; NumPy; PyTorch; Gymnasium; PettingZoo; Habitat-Sim; Habitat-Lab; Isaac Sim; Isaac Lab; PhysX; Robot Operating System (ROS 2).  
**Methods:** POMDPs; uncertain map priors and local geometric beliefs; RGB-D semantic mapping; YOLO-World open-vocabulary detection and Mobile-SAM pixel segmentation; depth back-projection and local point clouds; LiDAR-IMU alignment; procedural 3D scene generation; frontier and connectivity-graph planning; ACVRP and LKH; A*, kinodynamic A*, FMM, and B-spline trajectories; virtual-structure formation; MARL; Graph-SAC and Graph-MASAC; constrained SAC; successor features; behavior cloning and DAgger; Quality-Diversity and MAP-Elites; communication, collision, and safety constraints.

## Selected Research Contributions

### 1. Graph-Based Multi-Robot Planning and Safety Control — AeroGate Graph

- Developed a variable-team planning environment that encodes robots, route nodes, and obstacles as graph observations, combining A* routing, virtual-structure formation, Graph-SAC, and Graph-MASAC for cooperative decisions.
- Implemented velocity and acceleration limits, dynamic gate and obstacle handling, segment-level collision checks, and safety filtering; recorded path length, obstacle clearance, formation error, and planning latency for controlled single-robot and multi-robot comparisons.

### 2. Procedural 3D Search Benchmark and Multi-Sensor Physics Simulation — AeroCityBench and Rivermark

- Designed procedural city layouts and a unified task interface for 3D poses, sensor observations, action segments, communication messages, and search events. Kept policy-observable inputs separate from evaluation-only information to support controlled comparisons across geometric and semantic search tasks.
- Built multi-robot 3D scenes in Isaac Sim and PhysX. Aligned RGB, depth, semantic labels, LiDAR, inertial measurements, and body states across timestamps, coordinate frames, and sensor extrinsics.
- Integrated occupancy and free-space estimation, frontier candidates, region assignment, communication-connectivity checks, and collision checking into online execution, enabling comparisons across classical planners, MARL, Quality-Diversity, and vision-language policies.

### 3. Execution-Feedback-Driven Quality-Diversity for Cooperative Exploration — HM3D-Derived 3D Environments and Realised-QD

- Built a Quality-Diversity archive from execution receipts, using behavioral descriptors for realized coverage, redundant search, spatial role allocation, communication status, and safety interventions.
- Developed candidate-action evaluation with successor features and constraint costs. The formulation accounts for exploration return, execution duration, and safety cost while retaining distinct executable strategies.

### 4. Indoor Semantic Target Search and Connectivity-Aware Geometric Exploration — HM3D, Co-NavGPT2, and C2

- Implemented and extended a two-robot RGB-D semantic-search pipeline in HM3D, combining YOLO-World and Mobile-SAM target cues with depth back-projection, shared top-down mapping, and frontier extraction.
- Extended the C2 geometric-exploration pipeline with connectivity-aware mapping, task allocation, A*, kinodynamic planning, B-spline smoothing, and local collision checking for online replanning. Designed RegionEvidence to connect semantic cues with geometric frontiers for region ranking, task allocation, and local execution.

**Experimental design:** Used paired comparisons with fixed scene splits, observation permissions, controllers, action budgets, and random seeds. Reported separate metrics for semantic search success, path efficiency, geometric coverage, and execution safety.
