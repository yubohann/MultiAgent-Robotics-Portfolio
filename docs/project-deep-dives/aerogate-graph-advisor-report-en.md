# AeroGate Graph: Structured Multi-Agent Control for Dense and Dynamic Gate Traversal

**Advisor-facing research report**  
**Author:** Bohan Yu  
**Research area:** multi-agent robotics, graph reinforcement learning, motion planning, formation control, and safety-aware learning

## 1. Executive Summary

AeroGate Graph is a controlled research environment for studying how a variable-size team of aerial agents can traverse dense fields of static and moving gates while maintaining progress, formation geometry, and safe clearance. I designed the project as a complete research stack: a reproducible two-dimensional task contract, a graph-based state representation, route planning for a team reference point, virtual-structure formation control, graph-structured Soft Actor-Critic policies, expert-guided imitation, action-level safety intervention, and a measurement-oriented evaluation harness.

I separate responsibilities that end-to-end demonstrations often combine. A global planner proposes a route for the team reference point. A virtual structure converts that route into desired positions for individual agents. A graph encoder represents agents, formation targets, route context, and nearby gate geometry as relational state. The policy proposes velocity commands, while a safety layer checks each command against separation, obstacle, boundary, gate-channel, and motion constraints before state integration. This decomposition makes the system easier to analyze, extend, and evaluate through targeted experiments.

The current evidence includes single-agent dynamic-gate comparisons across gate densities, multi-agent static and dynamic scenarios, deterministic environment checks, replay validation, and integrity manifests for numerical artifacts. In the recorded 42-gate single-agent dynamic scenario, the mainline method completed all 10 evaluated seeds with 100% success and 0% collision. In the multi-agent results, static 60-gate, dynamic 36-gate, and dynamic 60-gate settings each achieved 100% success across three episodes. The evaluation also identifies a high-density stress boundary: the single-agent 60-gate dynamic configuration achieved 0% success and 60% collision across 10 seeds. I use this boundary as evidence for curriculum design, motion prediction, and robustness research.

The central contribution is an inspectable framework that connects relational representation, route-plus-formation coordination, safety intervention, and evidence collection under one task contract. The isolated benefit of each learning module is a clear follow-up question for a larger multi-seed ablation study.

## 2. Research Question and Hypothesis

### Research question

How can a team with a variable number of agents make sustained progress through dense, time-varying gate layouts while preserving a feasible formation and respecting geometric safety constraints?

This question has four coupled parts:

1. **Route choice:** Where should the team reference point move?
2. **Formation geometry:** Where should each agent be relative to that reference point?
3. **Relational state:** How should the policy represent agents, gates, route context, and their changing relationships?
4. **Risk handling:** How should a risky velocity command be corrected before it creates a collision or an unrecoverable formation state?

### Working hypothesis

A graph interface combined with route-guided virtual-structure targets can preserve useful relational information as team size and gate density change. Centralized value estimation can improve credit assignment during training, while decentralized graph-policy execution keeps the control interface local to each active agent. A velocity-space safety layer can reduce avoidable geometric failures and provide diagnostic evidence about where the learned policy benefits from intervention.

The hypothesis is deliberately mechanistic. It can be tested through controlled comparisons of graph versus non-graph state, route-plus-formation versus direct joint control, shielded versus unshielded action execution, and static versus moving-gate profiles.

## 3. Why Dense Dynamic Gates Form a Multi-Agent Systems Problem

A single vehicle traversing a wide, static opening is comparatively simple. The problem becomes substantially richer when gates are dense, gates move, and several vehicles must pass as a coherent team.

- **Spatial coupling:** The route that is safe for one vehicle can be infeasible for the entire formation.
- **Temporal coupling:** A moving gate changes the feasible passage geometry between observation and arrival.
- **Coordination coupling:** Each agent's command changes the clearance and future options of its teammates.
- **Team-level feasibility:** A locally attractive velocity must also preserve formation geometry, boundary clearance, and future gate access.
- **Credit assignment:** A team-level outcome must be related to individual actions and relational context.

These properties motivate a structured state and a layered controller. A flat vector can encode the same measurements, yet it leaves entity identity, relation type, and variable team structure implicit in the policy input. A graph makes those relations explicit and gives the experiment a natural unit for scaling team size and local context.

## 4. Task Formulation

### 4.1 State and action

The current research abstraction is fixed-height planar flight. For agent `i`, the state contains planar position `p_i = [x_i, y_i]`, planar velocity `v_i`, heading, and simulation time. The policy action is a desired planar velocity `u_i`.

The environment applies a bounded velocity-control model:

```text
v_i(t + dt) = clip(v_i(t) + clip(u_i(t) - v_i(t), -a_max*dt, a_max*dt), v_max)
p_i(t + dt) = p_i(t) + v_i(t + dt) * dt
```

The model is intentionally compact. It preserves the constraints most relevant to route, formation, and safety reasoning while keeping the task independent of any particular simulator or vehicle model. The resolved time step, speed limit, acceleration limit, radii, and gate dimensions are recorded with each experiment.

### 4.2 Gate and collision geometry

Gate posts are represented by circular obstacles in the planar task. Collision checks use both point clearance and the swept motion segment over a time step. Acceptance requires the entire swept motion to remain clear of a post.

For a point `p` and an obstacle with center `c` and collision radius `r_o`, the signed clearance for a vehicle radius `r_d` is conceptually:

```text
clearance(p) = ||p - c|| - r_o - r_d
```

Dynamic gates update their centers and velocities over time. The evaluator also records relative swept clearance, which measures whether the vehicle and moving post remain separated over the complete interval.

### 4.3 Team objective

The team seeks high route progress and successful gate completion while preserving clearance, pairwise separation, and formation quality. A representative episode objective is:

```text
J = w_p * progress
    - w_g * goal_distance
    - w_f * formation_error
    - w_c * collision_cost
    - w_s * separation_cost
    - w_o * obstacle_cost
    - w_u * action_smoothness_cost
```

The exact weights belong to the experiment configuration. Success is interpreted together with path length, flight time, minimum clearance, pair distance, formation-position error, and termination reason.

## 5. System Architecture

```text
Gate geometry and motion
          |
          v
Global route for the team reference point
          |
          v
Virtual-structure formation targets
          |
          v
Graph observation for agents, route, gates, and team context
          |
          v
Graph policy proposes velocity commands
          |
          v
Safety checks and velocity correction
          |
          v
Bounded dynamics and position integration
          |
          v
Progress, clearance, coordination, and termination evidence
```

Two features make the architecture useful for research. Each interface has a clear semantic contract covering geometry, route, formation, observation, action, and diagnostics. A new policy or safety method can then be compared while the underlying task record retains the same meaning.

## 6. Graph-Based State Representation

### 6.1 Node semantics

The policy receives a fixed-capacity graph with masks that distinguish active entities from padding. The graph contains the following conceptual node groups:

1. **Agent nodes:** relative position to the team reference point, velocity, error to the desired formation position, local clearance, progress, and team-size context.
2. **Formation-target nodes:** desired relative positions, heading context, target index, and route progress.
3. **Route look-ahead nodes:** upcoming waypoints, relative position, heading, order, and progress.
4. **Obstacle and dynamic-gate nodes:** relative position, radius, velocity, short-horizon predicted position, and clearance information.
5. **Guidance nodes:** optional route hints, target direction, risk or speed information, and confidence.
6. **Team node:** reference-point position, heading, aggregate clearance, and global progress.

This organization gives the graph a semantic vocabulary. A message from a teammate carries a different role from a message from a gate post, even when their distances are similar.

### 6.2 Relational connectivity

Active nodes receive self-connections, and pairwise edge weights decay with distance. A generic distance-weighted edge can be written as:

```text
A_ij = exp(-d_ij / sigma),  i != j
A_ii = 1
```

The graph is padded to a stable capacity, while node and action masks prevent inactive slots from entering message aggregation or producing control commands. This preserves one policy interface across different active team sizes. The current implementation keeps an ordered agent block for transparent debugging; future experiments can measure the effect of permutation-invariant pooling and explicit randomization of agent order.

### 6.3 Message passing

The encoder first maps each node feature through a multilayer perceptron, then performs masked graph message passing:

```text
H^(l+1) = phi(A * H^l * W_l + b_l)
```

Pooling over valid nodes produces a team-level embedding. The actor combines each agent's local embedding with this pooled context to produce one action for each active agent. This design retains local control authority while exposing the relational information needed for coordination.

## 7. Route Planning and Virtual-Structure Formation

### 7.1 Route planning

The global planner searches an obstacle-inflated two-dimensional grid using A* with eight-neighbor connectivity and a Euclidean heuristic. The grid path is converted to world coordinates and shortened with collision-free shortcuts where possible. The output is a sequence of waypoints for the team reference point, while individual vehicles follow the resulting formation targets.

This decomposition reduces the complexity of joint planning. The planner handles global geometric feasibility, while the formation layer distributes that route among team members.

### 7.2 Virtual-structure target generation

Let `p_ref` be the team reference point, `theta_ref` its route heading, and `d_i` the local formation offset for agent `i`. The desired world position is:

```text
p_i^desired = p_ref + R(theta_ref) * d_i
```

where `R(theta_ref)` is the two-dimensional rotation matrix. Formation presets adapt the number of rows and columns to team size, and the offsets remain centered around the reference point.

The decomposition is easy to inspect:

- route progress answers whether the team is moving toward the task goal;
- formation-position error answers whether individual agents track the team geometry;
- pairwise distance answers whether the team remains separated;
- gate clearance answers whether the formation fits the local passage.

### 7.3 Dynamic-gate timing

Passage feasibility depends on both current and predicted gate geometry. The relational state therefore includes gate velocity and a short-horizon predicted position. The desired formation can then be adjusted using an estimate of the gate geometry at the expected arrival time. This creates a direct research path toward time-parameterized planning and model-predictive formation shaping.

## 8. Graph-SAC and GraphMASAC Learning

### 8.1 Graph-structured Soft Actor-Critic

The single-agent path uses Soft Actor-Critic with a graph encoder. SAC maximizes expected return together with an entropy term:

```text
J(pi) = E[ sum_t gamma^t (r_t + alpha * H(pi(.|s_t))) ]
```

Entropy regularization supports exploration in narrow passages by reducing premature commitment to a single side. The graph encoder supplies relational state, while the SAC objective supplies continuous velocity control. In this project, graph structure and SAC are combined around an integrated task formulation and measurable coordination framework.

### 8.2 Centralized training and decentralized execution

The multi-agent path uses a shared graph actor and a centralized joint-action critic. During training, the critic receives pooled graph state, masked joint actions, and the active-agent mask. Twin Q estimates and target critics follow the standard SAC stabilization pattern. During execution, each active agent receives its own action from the shared actor.

The critic's global view addresses multi-agent credit assignment: it can evaluate a joint action in the context of the whole team. The actor's shared parameters and graph context support consistent behavior across agents and provide a controlled interface for team-size experiments.

### 8.3 Modular control interpretation

The actor can be interpreted as combining three behavior tendencies: route following, formation tracking, and local avoidance. Learned gating weights can combine these proposals according to the current relational context. This decomposition links each action to a behavioral mechanism and a measurable state while keeping the primary claim at the framework level.

## 9. Expert Guidance, Behavior Cloning, and DAgger

The learning pipeline has two stages: expert-guided initialization followed by reinforcement fine-tuning.

### 9.1 Behavior cloning

A geometric formation controller or teacher policy generates demonstrations containing graph observations, joint actions, progress, pair distance, formation error, clearance, and termination reason. The actor is first trained with a masked supervised action loss. Dataset selection can balance team sizes and retain episodes according to progress and geometric quality.

This warm start places the policy in a meaningful region of the action space before high-variance reinforcement updates begin.

### 9.2 DAgger-style correction

Expert-only trajectories emphasize nominal states. DAgger addresses the resulting distribution shift by rolling out the current policy, querying the teacher on policy-visited states, and appending corrective actions to the training set. The implementation prioritizes states near dynamic gates, low-clearance states, small pair distances, large formation errors, corridor events, and hard termination conditions.

The combined pipeline is:

```text
geometric demonstrations
        -> behavior-cloning warm start
        -> policy rollout
        -> teacher queries on risk states
        -> corrected dataset
        -> graph-SAC fine-tuning
```

This supports a direct comparison between learning from clean expert trajectories and learning from the states the policy actually visits.

## 10. Safety-Aware Action Execution

The policy proposes a velocity; a rule-based safety layer evaluates and adjusts that proposal before the dynamics integrate the state. The checks cover:

- gate-channel feasibility;
- pairwise separation between active agents;
- obstacle and gate-post clearance;
- environment boundaries;
- formation collapse indicators;
- braking behavior, while the dynamics enforce acceleration and speed limits.

When a command approaches a configured risk boundary, the layer can reduce or redirect the velocity. It also records activation, intervention magnitude, proximity measures, and the resulting termination reason. The shield is therefore both a safety mechanism and an experimental instrument.

A useful reference formulation is the minimum-change projection around a learned policy. If `U(s)` is the set of locally admissible velocity commands, the idealized projection is:

```text
u_safe = argmin_u ||u - u_policy||^2
         subject to u in U(s)
```

The implemented shield uses sequential geometric and rule-based velocity corrections rather than solving this optimization exactly. The formulation clarifies its design goal: preserve the policy command where possible while responding to separation, boundary, gate-channel, and obstacle risk. The research question is how much progress and clearance are gained for a given intervention budget, measured by success, collision rate, minimum clearance, intervention ratio, and path efficiency.

## 11. Experimental Protocol and Reproducibility

### 11.1 Scenario families

The project supports:

- single-agent static gates;
- single-agent moving-gate density sweeps;
- multi-agent static-gate formation traversal;
- multi-agent moving-gate scenarios;
- one-to-eight-agent team experiments, with an eight-agent dynamic-gate curriculum;
- classic planner comparisons and replay-oriented validation;
- optional Isaac Lab 3D replay for visualization and future sim-to-real work.

### 11.2 Metrics

Every policy result should be reported with its scenario, gate count, team size, and seed or episode count. The core metrics are:

| Metric | Interpretation |
| --- | --- |
| Success rate | fraction of episodes that complete the required traversal |
| Collision rate | fraction with gate, obstacle, boundary, or agent collision |
| Progress distance | distance advanced toward the task goal |
| Path length | geometric efficiency of the executed route |
| Flight time | temporal efficiency under the bounded dynamics |
| Minimum clearance | closest signed distance to a relevant obstacle or gate post |
| Pair distance | team separation and collision margin |
| Formation-position error | deviation from the desired virtual-structure geometry |
| Shield intervention | how often and how strongly the safety layer modifies actions |
| Termination reason | whether the episode ended by success, collision, timeout, or a classified formation event |

### 11.3 Reproducibility contract

The core environment can run independently on CPU, providing a lightweight foundation for training and visualization. Deterministic checks reset and roll out the same seed twice, then compare public diagnostics. Evaluation artifacts retain resolved configuration, runtime versions, seeds or episode counts, and numerical manifests with SHA256 verification. This separates three evidence levels:

1. **Environment evidence:** import, geometry, dynamics, API, and deterministic replay checks.
2. **Policy evidence:** learned-policy episodes and their recorded metrics.
3. **Visualization evidence:** replay or rendering outputs used to inspect trajectories.

Together, these levels identify whether a claim concerns the environment contract, learned-policy performance, or trajectory visualization, while keeping planar simulation evidence separate from physical-flight evidence.

## 12. Recorded Results

The following tables report recorded values, with every metric tied to its scenario and sample count.

### 12.1 Single-agent dynamic 42-gate comparison

| Method family | Seeds | Success | Collision | Timeout | Path length | Flight time | Minimum clearance | Moving-gate swept clearance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Graph-policy mainline | 10 | 100% | 0% | 0% | 83.388 m | 72.62 s | 0.052 m | 0.049 m |
| A* planner baseline | 10 | 10% | 80% | 10% | 23.240 m | 13.07 s | 0.277 m | 0.277 m |
| Theta* planner baseline | 10 | 0% | 90% | 10% | 21.175 m | 12.48 s | 0.261 m | 0.261 m |
| RRT* planner baseline | 10 | 0% | 0% | 100% | 0 m | 64.30 s | 2.808 m | 2.808 m |

The planner baselines provide reference points alongside the learned control stack. Their shorter recorded paths often reflect early termination before the gate field is traversed; combined with high collision or timeout rates, the results motivate bounded, feedback-based execution when moving gates are part of the task.

### 12.2 Multi-agent static and dynamic results

| Scenario | Gates | Episodes | Success | Collision | Path length | Flight time | Gate-post radius | Vehicle radius | Effective opening |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Static formation | 60 | 3 | 100% | 0% | 66.525 m | 35.267 s | 0.14 m | 0.12 m | 5.88 m |
| Dynamic formation | 36 | 3 | 100% | 0% | 67.087 m | 47.000 s | 0.24 m | 0.22 m | 3.88 m |
| Dynamic formation | 42 | 3 | 66.7% | 33.3% | 57.965 m | 35.367 s | 0.24 m | 0.22 m | 3.88 m |
| Dynamic formation | 54 | 3 | 66.7% | 33.3% | 60.309 m | 34.033 s | 0.24 m | 0.22 m | 3.88 m |
| Dynamic formation | 60 | 3 | 100% | 0% | 66.669 m | 38.033 s | 0.24 m | 0.22 m | 3.88 m |

These rows suggest that difficulty reflects gate count together with gate motion, arrival timing, local geometry, and the interaction between formation shape and passage width. This motivates reporting geometry and failure reasons together with success rate.

### 12.3 Stress-boundary evidence

| Single-agent dynamic setting | Seeds | Success | Collision | Interpretation |
| --- | ---: | ---: | ---: | --- |
| 48 gates | 10 | 0% | 50% | high-density operating boundary |
| 54 gates | 10 | 0% | 90% | severe density and motion stress |
| 60 gates | 10 | 0% | 60% | stress boundary for curriculum and robustness work |

These rows mark the current operating boundary for the recorded configuration. The next measurement should vary curriculum schedule, prediction horizon, gate-motion amplitude, and safety-intervention budget while holding the evaluation contract fixed.

## 13. Interpretation of the Evidence

### What the current evidence supports

- The task contract, geometry, graph interface, and diagnostics form a coherent executable framework.
- In the recorded 42-gate single-agent dynamic scenario, the mainline stack completed all evaluated seeds while the planner-only comparisons exhibited substantial collision or timeout rates.
- The retained multi-agent scenarios demonstrate successful formation traversal in selected static and dynamic settings, including dynamic 36- and 60-gate cases.
- The stress rows expose a measurable operating range and create a concrete agenda for curriculum and robustness experiments.
- The evaluation harness records quantities that connect mechanism to outcome: clearance, path length, time, formation error, pair distance, intervention, and terminal reason.

### What requires a larger study

The current artifact establishes the integrated framework. The next measurement is a component-wise attribution study. A paper-level evaluation should use paired multi-seed runs with fixed layouts and training budgets for:

1. graph observation versus a matched flat observation;
2. route-plus-formation targets versus direct joint control;
3. policy-only execution versus shielded execution;
4. behavior cloning only versus DAgger plus reinforcement fine-tuning;
5. current-state gate features versus predicted dynamic-gate features.

The stable interfaces and metrics make these comparisons straightforward to implement and interpret.

## 14. Personal Technical Contribution

My contribution to the project spans five layers.

### 14.1 Research formulation

I formulated dense gate traversal as a measurable coordination problem that combines route progress, formation geometry, dynamic obstacles, and safety constraints under one task contract. This creates a common basis for comparing planning, graph representation, learning, and intervention.

### 14.2 System architecture

I designed the route-reference-point and virtual-structure decomposition, the graph node semantics, the masking interface for variable team sizes, the dynamic-gate prediction features, and the action-level safety boundary. Each layer exposes diagnostics that explain its effect on the next layer.

### 14.3 Learning pipeline

I implemented the graph-structured SAC path, centralized joint-action value estimation, twin critics and target updates, masked actions, behavior-cloning warm starts, DAgger correction, failure-aware replay, and optional safety-critic regularization. These components share the same environment and evaluation record, which makes controlled follow-up experiments straightforward.

### 14.4 Engineering for research

I built a dependency-light planar core, deterministic scenario generation, vectorized training support, checkpoint and artifact metadata, replay validation, and test coverage for geometry, configurations, public interfaces, and single- and multi-agent rollouts. The result is a framework that can be inspected before training and audited after evaluation.

### 14.5 Scientific communication

I organized results by scenario, method family, sample count, and evidence level. This makes the project suitable for a research discussion: the reader can see what was implemented, what was measured, where the operating range lies, and which hypotheses remain to be tested.

## 15. Presentation Scope for Advisors and Interviewers

The document is intentionally selective about technical detail.

| Show in an interview or research discussion | Keep in a private engineering appendix |
| --- | --- |
| research question and hypothesis | local machine paths |
| task abstraction and equations | source filenames and directory trees |
| graph node semantics and masking rationale | raw tensor keys and private configuration names |
| route, formation, dynamic-gate, and shield responsibilities | checkpoint filenames and experiment scratch logs |
| scenario, sample count, metrics, and evidence boundary | implementation-only helper functions |
| personal decisions, tradeoffs, and measurable contribution | verbose dependency or import details |
| next experiments and ablation design | unverified visual claims or unlabeled screenshots |

This scope lets an advisor evaluate research maturity, theoretical understanding, and ownership through concepts that are independent of the repository's internal vocabulary.

## 16. Evidence Boundary and Research Extensions

The current core deliberately focuses on planning and coordination in a fixed-height planar setting with simulator-provided geometry. This is a strong abstraction for isolating the effect of relational policy and formation design. The Isaac Lab bridge supports three-dimensional replay and visualization, creating a natural starting point for future work. The bridge currently serves as a replay and transfer interface; claims about 3D control will follow a dedicated evaluation.

The most direct research extensions are:

1. **Perception-grounded graph state:** replace privileged geometry with LiDAR, camera, or fused local observations and model uncertainty explicitly.
2. **Time-parameterized planning:** predict gate arrival intervals and optimize the route and formation jointly in space-time.
3. **Adaptive formation morphing:** change row and column structure when the effective opening becomes narrower than the nominal formation span.
4. **Formal or learned safety sets:** compare geometric projection, control barrier functions, and learned safety critics under a common intervention budget.
5. **Communication-aware coordination:** add delay, bandwidth, packet loss, and local-only observations to the graph interface.
6. **Sim-to-real validation:** preserve the task contract while introducing six-degree-of-freedom dynamics, actuator limits, state-estimation latency, and physical gate experiments.
7. **Statistical evaluation:** run frozen multi-seed ablations and report confidence intervals for success, clearance, efficiency, and intervention.

These extensions follow directly from the current interfaces, so the project functions as a research program with a clear path beyond the initial demonstration.

## 17. Interview-Ready Summary

> AeroGate Graph is a fixed-height planar environment I built to study multi-agent coordination through dense fields of static and moving gates. The team reference point follows a global route, and a virtual-structure layer converts that route into desired positions for each agent. I represent agent state, formation targets, route look-ahead, nearby gates, and team context as a masked graph, giving the policy a stable interface across team sizes. A shared graph actor proposes planar velocities, while a centralized critic evaluates joint actions during training. Before state integration, a safety layer checks separation, boundaries, obstacles, gate channels, and motion limits and can correct risky commands. The environment records progress, path length, flight time, clearance, pair distance, formation error, intervention, and termination reason. I also built behavior-cloning and DAgger paths so expert guidance can focus on policy-visited risk states before graph-SAC fine-tuning. My main contribution is an implemented, inspectable framework that connects route planning, formation geometry, relational learning, safety intervention, and reproducible evidence in one task contract.

## 18. Selected Literature Foundations

The project uses established methods as design anchors and combines them around a specific multi-agent gate task.

| Literature line | Role in this project |
| --- | --- |
| Hart, Nilsson, and Raphael, A* | global reference-point route planning |
| Daniel, Nash, Koenig, and Felner, Theta* | any-angle planning comparison |
| Karaman and Frazzoli, RRT* | sampling-based planning comparison |
| Haarnoja et al., Soft Actor-Critic | maximum-entropy continuous control |
| Lowe et al., multi-agent actor-critic | centralized training and decentralized execution pattern |
| Graph convolutional reinforcement learning | relational message passing for policy state |
| Spatial graph networks for multi-robot exploration | graph representation of multi-robot context |
| Virtual-structure formation control | formation geometry around a reference trajectory |
| Alshiekh et al., safe reinforcement learning via shielding | action intervention around learned control |
| Ross, Gordon, and Bagnell, DAgger | correction on policy-visited states |

Together, these foundations support the framework's focus on relational state, structured coordination, and measurable safety in a concrete multi-agent gate problem.

## 19. Final Advisor Takeaway

AeroGate Graph demonstrates a research approach suited to multi-agent robotics: define a controlled problem, assign clear responsibilities across layers, expose the relational variables that matter, record the evidence needed to interpret a failure, and define a clear path to the next experiment.

The project connects four themes that are central to modern robotics research:

- structured representations for variable-size teams and changing environments;
- planning and formation geometry that remain interpretable;
- learning methods that combine expert guidance with reinforcement optimization;
- safety and reproducibility treated as part of the research method.

This foundation is ready for deeper work on perception-grounded graphs, dynamic space-time planning, formal safety, communication constraints, and physical validation. It gives an advisor a concrete starting point for discussing both immediate research questions and a longer-term multi-agent robotics agenda.

---

## 20. Extended Technical Appendix for Advisors

### 20.1 Controlled task and variable roles

The current environment is a fixed-height planar abstraction. Agent `i` has planar position `p_i`, velocity `v_i`, heading, and time; the policy proposes a desired velocity `u_i`. The environment applies an acceleration bound and a speed bound before integrating position. Every run records these parameters, gate motion, radii, seed, checkpoint identity, and termination reason.

### 20.2 Geometry and swept safety

For a post centered at `c`, with obstacle radius `r_o` and vehicle radius `r_d`, instantaneous clearance is `||p-c||_2-r_o-r_d`. The evaluator also checks the swept segment between consecutive states, and moving gates are propagated over the same interval. The safety layer records the active constraint, proposed action, executed action, correction norm, and outcome.

### 20.3 Route, formation, graph, and action

The global planner produces a route for the team reference point. A virtual structure converts the reference point and orientation into slot targets using `p_i^desired=p_ref+R d_i`. The graph encodes active agents, desired slots, route look-ahead, nearby gates, and relation features. A shared actor proposes local velocities, a centralized critic evaluates joint context during training, and execution remains local to each active agent.

This decomposition separates an unsuitable reference route, a route feasible for the reference point but infeasible for the full formation, and a locally unsafe learned action. The recorded metrics make these explanations distinguishable.

### 20.4 Variable team size and learning

Typed graph nodes and a mask provide a stable interface across team sizes. Validation should compare an equal-capacity flat policy, a graph with relation edges removed, and a centralized joint policy under matched seeds and budgets. Behavior cloning initializes the policy from expert states; DAgger adds labels on states visited by the current policy, including deviations and recovery situations.

### 20.5 Dynamic gates and safety boundary

Current gate features describe geometry at observation time. Arrival-time features propagate gate center and velocity to the predicted arrival time, so the policy reasons about the geometry it will encounter. Compare prediction residual, arrival clearance, waiting, replanning, and path length. The current shield is an action-level geometric intervention; comparison with a control-barrier-function or learned safety critic should use the same safe set and actuator limits.

### 20.6 Evidence and causal ablation matrix

The recorded 42-gate single-agent dynamic case contains 10 seeds with 100% success and 0% collision for the mainline method. The 60-gate dynamic stress case contains 10 seeds with 0% success and 60% collision. Selected multi-agent static-60, dynamic-36, and dynamic-60 settings each contain three recorded successful episodes. These results define an operating range and stress boundary; they do not isolate component-level causality.

| Ablation | Mechanism question | Required outputs |
| --- | --- | --- |
| graph vs matched flat state | does relational structure help? | success, clearance, formation error, held-out scale |
| route-plus-formation vs direct control | does decomposition simplify coordination? | path, time, pair distance, critic variance |
| shielded vs policy-only | does intervention prevent avoidable failures? | collision, intervention, correction norm |
| BC/DAgger/RL-only | does policy-visited supervision improve recovery? | sample efficiency, recovery, failure class |
| current vs predicted gate state | does temporal prediction improve feasibility? | arrival clearance, waiting, residual |

Every row should use paired layouts, frozen evaluation checkpoints, fixed training budgets, and a common termination taxonomy.

## 21. Advisor Takeaway

AeroGate Graph is a controlled coordination framework for a variable-size aerial team moving through dense, time-varying gates. Its central contribution is the inspectable chain from reference route to virtual-structure targets, relational graph state, learned local actions, and pre-integration safety checks. The current results establish the framework and its recorded operating range; a paired multi-seed ablation will identify the independent value of each mechanism.
