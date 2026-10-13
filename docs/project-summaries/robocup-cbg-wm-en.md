# Uncertainty-Aware Counterfactual Belief-Graph World Model for Rule-Constrained Multi-Robot Tactics

**Project Lead and Core Developer** | **Mar. 2025 - Oct. 2025**  
**Research focus:** Developed a world-model planning system for a partially observable two-robot visual adversarial task with target occlusion, delayed observations, and movable obstacles.  
**Technical stack:** ROS 2 Jazzy; Isaac Lab; Python; PyTorch; Gymnasium; Stable-Baselines3; Nav2; slam_toolbox; robot_localization; OpenCV; AprilTag; RGB-D; LiDAR; IMU; Sim2Real interfaces.  
**Methods:** Multi-agent deep reinforcement learning; SAC Flow; model-based reinforcement learning; uncertainty-aware belief tokens; typed object-interaction graphs; probabilistic ensembles; paired counterfactual interventions; Flow-proposal CVaR MPC; rule-aware action shielding; centralized training and decentralized execution; residual expert policies.

- Built a ROS 2 and Isaac Lab two-robot adversarial environment with target ownership, range and line-of-sight gating, dwell-time task triggering, movable obstacles, protected-region blockers, and collision constraints.
- Encoded multimodal perception as 23 fixed-type belief tokens with 368 dimensions; occluded objects retain their latest belief with increased uncertainty. Built an equivariant typed interaction graph and a five-model ensemble that predicts object-state deltas, rewards, termination, visibility, presence, and four rule-risk channels.
- Designed paired box-removal and protected-region-blocker interventions to measure changes in state, reward, risk, and graph edges. Used Flow policy proposals with receding-horizon CVaR MPC to rank trajectories by return, rule risk, and ensemble disagreement before action shielding.
- Learned six high-level tactical controls for target selection, protected-region approach timing, blocking, recovery, task triggering, and risk preference under centralized training and decentralized execution, using a residual expert policy to preserve rule-aware priors.
- Integrated AprilTag perception, robot_localization, Nav2, and task-execution control through a ROS 2 deployment contract, then completed real-robot experiments in a two-robot 1v1 setting.
- Built a separate rule-level simulation benchmark with a staged 5v5-to-50v50 curriculum; conducted a 256-game evaluation and Isaac Lab tactical visualization.
