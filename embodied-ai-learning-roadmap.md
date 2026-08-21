# Embodied AI from LLMs to Robots

> A 12-week project-driven learning and research roadmap from LLM foundations to safe embodied systems.

**Audience:** Students and engineers with Python and basic machine-learning experience who want a practical route into LLM applications, multimodal learning, robot learning, embodied AI or AI systems engineering.

**Workload:** 12 weeks, 14--18 hours per week: 4 hours study, 7 hours implementation, 3 hours experiments and 1--4 hours review and writing.

**Final artifact:** A reproducible `Embodied Task Agent` that closes the loop from language task to knowledge and visual perception, sub-goal planning, world-model prediction, robot policy and safety-gated execution in simulation, with an experiment report, ablations and a publishable GitHub repository.

**Core stack:** PyTorch, Hugging Face, LeRobot and Gymnasium/MuJoCo.

**Extension stack:** ROS 2, Isaac Lab, LIBERO, ManiSkill, RoboTwin 2.0, OpenVLA, pi0/openpi and GR00T.

**Last checked:** 2026-08-19. Pin versions before running an experiment. Check model cards, dataset cards, repository licenses and hardware requirements rather than relying on a README claim.

**Chinese companion:** [简体中文路线](https://github.com/yubohann/embodied-ai-from-llms-to-robots/blob/main/README.zh-CN.md). The bilingual files share the same outline, assignment IDs and paper arXiv IDs.

---

## Navigation

- [Learning outcomes](#learning-outcomes)
- [Capstone project](#capstone-project-embodied-task-agent)
- [Industry capability map](#industry-capability-map)
- [Prerequisites and diagnostic](#prerequisites-and-diagnostic)
- [Compute and software tracks](#compute-and-software-tracks)
- [12-week curriculum](#12-week-curriculum)
- [Paper ladder](#paper-ladder)
- [2026 research index](#2026-research-index)
- [Course and resource map](#course-and-resource-map)
- [Evaluation protocol](#evaluation-protocol)
- [Engineering standards](#engineering-standards)
- [Graduation rubric](#graduation-rubric)
- [Research upgrade path](#research-upgrade-path)
- [Public repository checklist](#public-repository-checklist)

## Learning outcomes

By the end of the core route, you should be able to:

1. Implement a small Transformer in PyTorch and explain tokenization, attention, optimization, training curves and inference bottlenecks.
2. Build an evaluable LLM application with structured outputs, tool calls, hybrid retrieval, evidence citations, abstention and traceable logs.
3. Adapt or evaluate a vision-language model on images, video or document pages, reporting calibration, latency and failure categories.
4. Train an action-conditioned world model that predicts future state, reward and termination, then plan with CEM or MPC.
5. Run an ACT, Diffusion Policy or VLA checkpoint in LIBERO, ManiSkill, MuJoCo or Isaac Lab and report episode-level results.
6. Define ROS 2 and simulator interfaces with action limits, state freshness, human takeover and emergency-stop boundaries.
7. Run baselines, ablations, multiple seeds and failure analysis, and publish a model card, data card and paper-style report.
8. Deliver software that is installable, tested, reproducible, observable and reversible.

## Capstone project: Embodied Task Agent

### System shape

```text
Natural-language task / rules
              |
              v
Task planner: LLM + RAG -> typed sub-goals, constraints, stop conditions
              |
              v
Multimodal perception: VLM / image-text retrieval -> objects, regions, evidence, confidence
              |
              v
State estimation: vision, robot state, timestamps, localization quality
              |
              v
World model: (state, action) -> future state, reward, termination, uncertainty
              |
              v
Policy and planning: ACT / Diffusion Policy / VLA / CEM-MPC
              |
              v
Safety execution layer -> simulator -> human confirmation -> optional robot
```

### Choose one task track

All tracks share the same data, evaluation and safety contracts.

| Track | Example task | Observation | Action |
|---|---|---|---|
| Mobile robot | Reach a zone, identify a target and plan a safe route | RGB/depth, odometry, map or local grid, language | `cmd_vel`, waypoint or local goal |
| Manipulator | Find, grasp and place an object; request help on failure | RGB/depth, proprioception, end-effector pose, language | end-effector pose, joint target or action chunk |
| Software-only | Complete a task in MuJoCo, ManiSkill or LIBERO | simulator image, state and language | environment-defined continuous or discrete action |

### Unified interface

```yaml
observation:
  language: string
  images: list[image]
  state: vector
  timestamp: float
  frame_id: string
action:
  type: subgoal | velocity | pose | joint | action_chunk
  value: vector_or_json
safety:
  max_action_age_ms: 200
  max_velocity: TBD
  workspace: TBD
  abort_on_collision: true
```

The LLM is responsible for task-level planning and constraints. Low-level control, collision checking, emergency stop and hardware protection remain deterministic control-layer responsibilities.

### Milestones

| Milestone | Week | Acceptance evidence |
|---|---:|---|
| M1: model and evaluation foundations | W3 | small Transformer, inference service and structured evaluation set |
| M2: multimodal toolchain | W6 | image understanding, image-text retrieval and evidence return |
| M3: prediction and policy | W9 | world model, CEM/MPC and robot-policy baseline |
| M4: closed loop and release | W12 | simulator task, failure recovery, report, demo and repository |

## Industry capability map

Companies need people who can turn models into reliable systems, not only call a checkpoint. Each capability below has a portfolio-level proof.

| Capability | What to demonstrate | 12-week evidence |
|---|---|---|
| Python/PyTorch | models, datasets, training loops and tests | Transformer, world model and evaluation code |
| C++/ROS 2 | nodes, messages, QoS, TF, bags and performance limits | ROS 2 bridge or a C++ performance node (advanced) |
| Linux/GPU | CUDA, memory, profiling, containers and process control | p50/p95 latency, memory, throughput and failure logs |
| LLM applications | structured output, tools, RAG and model serving | task planner with citations and abstention |
| Multimodal learning | visual features, VLMs, annotation and calibration | 50--200 item image/page set and error taxonomy |
| Robot learning | frames, kinematics, imitation learning and policy evaluation | ACT/Diffusion/VLA episode results |
| World models/RL | dynamics prediction, planning, uncertainty and OOD | multi-step error, CEM/MPC and safety gate |
| Data engineering | versions, splits, quality checks, lineage and privacy | manifests, data card and replayable generation script |
| Production engineering | APIs, logs, monitoring, CI, rollback and budgets | `make test/eval/report`, GitHub Actions |
| Research | question, baseline, ablation, statistics and writing | 6--8 page technical report and auditable plots |
| Collaboration | README, design docs, issues, PRs and demo | repository front page, architecture diagram and changelog |

### Role alignment

| Target role | Interview evidence | Weeks |
|---|---|---|
| LLM/application engineer | structured output, RAG evaluation, tools, latency and cost trade-offs | W2--W4 |
| Multimodal/VLA engineer | image-action data, adaptation, benchmarks and failure replay | W5--W6, W10 |
| Robot-learning engineer | kinematics, ROS 2, simulation, imitation learning, action interfaces and safety | W7, W9--W11 |
| Research engineer | paper reproduction, falsifiable hypothesis, baseline, ablation and statistics | W1, W3, W8, W12 |
| ML platform/inference engineer | GPU profiling, containers, CI, logs, monitoring, versioning and rollback | W2, W11--W12 |

One project can support several roles, but a resume and interview should use one primary positioning label and evidence chain instead of listing model names.

## Prerequisites and diagnostic

### Required

- Python, NumPy, Git, Linux command line and basic software engineering.
- Linear algebra, probability, calculus, gradient descent and neural-network training.
- PyTorch tensors, `Dataset`/`DataLoader` and GPU debugging.

### Recommended

- Transformer, CNN, representation learning and reinforcement-learning basics.
- ROS 2 topics/services/actions, TF, bags and QoS.
- Robot kinematics, coordinate frames, PID and simulation.

### Pre-week-1 diagnostic

```text
Write an MLP in PyTorch and overfit a tiny dataset.
Explain softmax, cross-entropy, Adam and batch size.
Read JSONL and create train/validation/test splits.
Create an environment on Linux or WSL and run pytest.
Plot a loss/accuracy curve and explain an anomaly.
```

Use [PyTorch Tutorials](https://pytorch.org/tutorials/), [CS231n Python/NumPy review](https://cs231n.stanford.edu/) and the [Hugging Face LLM Course](https://huggingface.co/learn/llm-course) to close gaps without delaying the whole plan.

## Compute and software tracks

| Track | Conditions | Core work | Extension |
|---|---|---|---|
| A: light | CPU or less than 8 GB VRAM | small Transformer, RAG, SmolVLM inference, state world model, MuJoCo | short remote-GPU runs |
| B: standard | one 16--24 GB GPU | QLoRA, ACT/Diffusion, LIBERO, ManiSkill, light Isaac Lab | OpenVLA/OFT inference |
| C: research | 40 GB+ Linux/NVIDIA or cloud GPU | larger fine-tuning, OpenVLA, pi0/openpi, GR00T, RoboTwin | real-robot work |

Recommended environment: Python 3.11/3.12, PyTorch, `uv` or Conda, CUDA, Docker, GitHub Actions and TensorBoard/W&B or Trackio. Save `environment.lock`, GPU information, git commit and configuration for every run.

# 12-week curriculum

Each week has four fixed parts: **study, implement, measure, deliver**. Core tasks form the route; advanced tasks add research depth.

## W1: Transformer and language-model foundations

**Study:** [Stanford CS336](https://cs336.stanford.edu/), [CS224N](https://web.stanford.edu/class/cs224n/), the [Hugging Face LLM Course](https://huggingface.co/learn/llm-course), and the Transformer, Chinchilla and FlashAttention-2 papers.

**Implement:** A tokenizer, causal self-attention, Transformer block, training loop, checkpoint and sampler in PyTorch. Use public small text such as TinyStories; do not use private data.

**Measure:** Training loss/perplexity, parameter count, tokens/s, peak memory, sequence length and batch-size effects. Record one overfit and one underfit case.

**Deliver:** `tiny_lm/`, curves, configuration, tests and a one-page explanation from token to logits.

## W2: Training systems, inference serving and performance

**Study:** CS336 systems assignments, profiling and memory; [llama.cpp](https://github.com/ggerganov/llama.cpp), [Ollama](https://docs.ollama.com/), [vLLM](https://docs.vllm.ai/) and [Full Stack Deep Learning](https://fullstackdeeplearning.com/course/).

**Implement:** Wrap a local Qwen or small model as a service with streaming, JSON Schema, timeout, retry, request ID and health check. Add `bench_inference.py`.

**Measure:** Time to first token, generation throughput, p50/p95 latency, concurrency, peak memory, failure rate and CPU/GPU utilization. Compare two backends or configurations.

**Deliver:** locally runnable service, performance report, fault-injection test and deployment notes. Advanced: read FlashAttention-2 code or implement a simplified kernel comparison.

## W3: Instruction tuning, quantization and LLM evaluation

**Study:** the [HF LLM Course](https://huggingface.co/learn/llm-course), [TRL](https://huggingface.co/docs/trl/), [PEFT](https://huggingface.co/docs/peft/), [Generative AI with LLMs](https://www.deeplearning.ai/courses/generative-ai-with-llms/) and [Fine-tuning LLMs](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/).

**Implement:** Choose Q4/Q8/full-precision comparison for low memory, or QLoRA/SFT on 100--300 structured examples with 16 GB+.

**Measure:** Format validity, task accuracy, abstention quality, latency, throughput, memory, training stability and overfitting on a 50--80 item test set covering format, facts, reasoning, refusal, tools and long context.

**Deliver:** model card, quantization or LoRA configuration, JSONL evaluation set, error taxonomy and replayable commands.

## W4: RAG, tools and task planning

**Study:** [Full Stack LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/), [HF Agents Course](https://huggingface.co/learn/agents-course), [FAISS](https://faiss.ai/), [Ragas](https://docs.ragas.io/) and the [HF Cookbook](https://huggingface.co/learn/cookbook/).

**Implement:** Build BM25, dense and hybrid retrieval baselines over 5--10 public or de-identified documents. Add a tool registry and planner that emits typed sub-goals, constraints, evidence and stop conditions.

**Measure:** On 40--60 answerable, unanswerable, cross-document, conflicting-version, tool-call and long-context queries, report Recall@k, MRR/nDCG, citation support, answer accuracy, abstention and end-to-end latency.

**Deliver:** `rag_eval.py`, retrieval visualization, evidence return, tool-error recovery and a RAG report.

## W5: Computer vision and vision-language models

**Study:** [Stanford CS231n](https://cs231n.stanford.edu/), the [HF Computer Vision Course](https://huggingface.co/learn/computer-vision-course), and CLIP, BLIP-2, LLaVA and SigLIP.

**Implement:** Use SmolVLM2 or a similar small VLM on 50--200 public or simulated images. Produce structured objects, regions, occlusions, hazards and uncertainty, retaining the evidence image or region.

**Measure:** Class F1, region IoU/hit rate, schema validity, calibration, lighting/occlusion degradation and per-image latency. Compare a color-threshold, classical detector or CLIP retrieval baseline.

**Deliver:** visual tool, data card, error table and reproducible visualizations. Advanced: implement a small CLIP contrastive-learning experiment.

## W6: Multimodal retrieval and model adaptation

**Study:** [ColPali](https://huggingface.co/blog/manu/colpali), its [model page](https://huggingface.co/vidore/colpali), [SmolVLM2](https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct) or a Qwen vision model card, and the first sections of the [VLA survey](https://arxiv.org/abs/2505.04769).

**Implement:** Compare OCR+text retrieval with visual late-interaction retrieval over field maps, device screenshots, rule pages and fault photos. Optionally adapt a small VLM with LoRA; without a GPU, focus on prompts, calibration and post-processing.

**Measure:** Recall@5, evidence-page hit rate, answer support, citation error, query latency, pre/post adaptation generalization and memory.

**Deliver:** multimodal RAG tool, retrieval visualization, model/data license inventory and a stage report.

## W7: Robotics, data formats and imitation learning

**Study:** [Modern Robotics](https://modernrobotics.northwestern.edu/), the [HF Robotics Course](https://huggingface.co/learn/robotics-course), [LeRobot docs](https://huggingface.co/docs/lerobot), [ACT](https://arxiv.org/abs/2304.13705) and [Diffusion Policy](https://arxiv.org/abs/2303.04137).

**Implement:** Complete one imitation-learning baseline in MuJoCo, LIBERO, ManiSkill or a ROS 2 simulator. Define observation, action, timestamp, termination and success; handle frames, normalization and action chunks.

**Measure:** Episode success, collision/out-of-bounds rate, action error, control frequency, inference latency and random-seed variance.

**Deliver:** policy checkpoint, data manifest, 10--20 evaluation episodes, trajectory visualization and action-interface document. Advanced: write an `rclcpp` node or bag-replay tool.

## W8: World models and state prediction

**Study:** [World Models](https://arxiv.org/abs/1803.10122), [DreamerV3](https://arxiv.org/abs/2301.04104), [TD-MPC2](https://arxiv.org/abs/2310.16828), the [DreamerV3 implementation](https://github.com/danijar/dreamerv3) and [V-JEPA 2](https://arxiv.org/abs/2506.09985).

**Implement:** From robot or Gymnasium/MuJoCo trajectories, train `(observation_t, action_t) -> observation_{t+1}, reward_t, terminated_t`. Compare an MLP state model with latent dynamics. Split by complete trajectories rather than shuffling adjacent frames. Add an ensemble, bootstrap or prediction interval.

**Measure:** 1/5/10-step error, reward MAE, termination F1, long-rollout collapse, OOD error growth, interval coverage and inference latency.

**Deliver:** world-model code, trajectory generator, real/predicted plots, uncertainty report and `predict_future` tool.

## W9: Model-based planning, RL and simulation transfer

**Study:** [Berkeley CS285](https://rail.eecs.berkeley.edu/deeprlcourse/), the [HF Deep RL Course](https://huggingface.co/learn/deep-rl-course) as optional low-maintenance material, [MuJoCo](https://mujoco.readthedocs.io/), [ManiSkill](https://maniskill.ai/) and [Isaac Lab](https://isaac-sim.github.io/IsaacLab/).

**Implement:** Use the W8 model for random shooting or CEM/MPC. Add at least three of lighting, friction, action delay and observation noise. Compare a fixed rule, model-free policy and world-model planner.

**Measure:** Return, success, collision/out-of-bounds rate, sample efficiency, planning latency, worst percentile and uncertainty-triggered safety stops.

**Deliver:** planner, domain-randomization configuration, one ablation figure and a sim-to-real risk table. Advanced: reproduce the same task on a pinned Isaac Lab release.

## W10: VLA, robot data and standard benchmarks

**Study:** [Open X-Embodiment](https://robotics-transformer-x.github.io/), [Octo](https://github.com/octo-models/octo), [OpenVLA](https://github.com/openvla/openvla), [openpi](https://github.com/Physical-Intelligence/openpi), [LIBERO](https://libero-project.github.io/), [RoboMimic](https://robomimic.github.io/) and [RoboTwin 2.0](https://robotwin-platform.github.io/).

**Implement:** Run one existing checkpoint in simulation, train ACT/Diffusion as the controlled baseline and use VLA as the comparison. Record action representation, input size, control frequency, memory and failure modes.

**Measure:** At least 20 episodes with task success, subtask completion, latency, human intervention, action violations and scene variance. Never replace local results with a paper number.

**Deliver:** unified policy adapter, benchmark configuration, checkpoint inventory, failure replay and license record. GR00T is an extension: its code, gated dependency, GPU and Python/FFmpeg requirements must be recorded separately.

## W11: System integration, ROS 2 and safe execution

**Study:** [ROS 2 Humble documentation](https://docs.ros.org/en/humble/) and [Full Stack Deep Learning](https://fullstackdeeplearning.com/course/) for testing, deployment, monitoring and collaboration. Read the [Embodied Intelligence survey](https://arxiv.org/abs/2507.00917) and [VLA survey](https://arxiv.org/abs/2502.06851).

**Implement:** Connect the W4 planner, W6 visual tool, W8 world model and W10 policy in `BOOT -> OBSERVE -> PLAN -> EXECUTE -> VERIFY -> RECOVER/ABORT`. Add schemas, action limits, workspace/collision constraints, pose/state freshness, confidence gates, timeout, retry budget, pause, cancellation, human takeover, emergency stop and unified logs containing task ID, episode, model version, latency, state age and failure reason.

**Measure:** Run 30--50 simulated episodes while injecting stale state, target loss, model errors, planning timeout, collision and communication interruption. Verify the expected safe state for each fault.

**Deliver:** end-to-end state machine, fault-injection scripts, timing diagram, QoS/interface document and safety report. Real hardware is allowed only after emergency stop, limits, low-speed mode and human confirmation are verified.

## W12: Research sprint, reproduction and release

Choose one falsifiable question: does hybrid RAG improve citation support; does uncertainty gating reduce world-model-planning collisions; does LLM sub-goal planning beat a fixed script at acceptable latency; does domain randomization improve unseen-scene success; does quantization improve end-to-end latency at an acceptable success loss; or does state feedback improve LLM recovery?

Change one primary factor, run a baseline and ablation with at least three seeds when possible, and state any single-seed or approximate experiment. Publish a 6--8 page paper-style report with abstract, problem, related work, method, setup, results, failure analysis, limitations, license and future work. Add a two-minute demo, architecture diagram, result plots, `README.md`, `CITATION.cff`, `LICENSE`, `CONTRIBUTING.md`, model card, data card, experiment manifest, test report and next-stage issues.

## Paper ladder

### Core reading

| Topic | Paper | Reading question |
|---|---|---|
| Transformer | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | What did attention and positional encoding change? |
| Scaling | [Chinchilla](https://arxiv.org/abs/2203.15556) | How should parameters, data and compute be balanced? |
| Efficient training | [FlashAttention-2](https://arxiv.org/abs/2307.08691) | Where are the IO and memory bottlenecks? |
| Alignment | [InstructGPT](https://arxiv.org/abs/2203.02155) | How do SFT, preference data and RLHF relate? |
| Vision-language | [CLIP](https://arxiv.org/abs/2103.00020) | How does contrastive learning align images and text? |
| Multimodal | [BLIP-2](https://arxiv.org/abs/2301.12597) | How are a vision encoder, Q-Former and LLM connected? |
| Vision instruction | [LLaVA](https://arxiv.org/abs/2304.08485) | What do visual instruction data and projection buy, and what do they fail to measure? |
| World models | [World Models](https://arxiv.org/abs/1803.10122) | How do representation, memory and control fit together? |
| Latent control | [DreamerV3](https://arxiv.org/abs/2301.04104) | Why can imagined trajectories improve sample efficiency? |
| Model-based planning | [TD-MPC2](https://arxiv.org/abs/2310.16828) | How do latent dynamics and MPC interact? |
| Imitation | [ACT](https://arxiv.org/abs/2304.13705) | How does action chunking help long-horizon tasks? |
| Diffusion policy | [Diffusion Policy](https://arxiv.org/abs/2303.04137) | Why is an action distribution suitable for diffusion? |
| Generalist policy | [Octo](https://arxiv.org/abs/2405.12213) | How can cross-robot data train a general policy? |
| VLA | [OpenVLA](https://arxiv.org/abs/2406.09246) | How does vision-language knowledge transfer to action? |
| Flow-matching action | [pi0](https://arxiv.org/abs/2410.24164) | How do flow matching, action chunks and VLA fit together? |

### Frontier reading

Choose 3--5. Use the same one-page note for every paper: problem, hypothesis, method, data, baseline, metrics, ablations, failure boundary, code status, license and reproducibility.

| Direction | Resource | Status |
|---|---|---|
| Predictive representations | [V-JEPA 2](https://arxiv.org/abs/2506.09985) | 2025 preprint |
| VLA overview | [Vision-Language-Action Models: Concepts, Progress, Applications and Challenges](https://arxiv.org/abs/2505.04769) | 2025 survey preprint |
| Embodied overview | [Learning Embodied Intelligence from Physical Simulators and World Models](https://arxiv.org/abs/2507.00917) | 2025 survey preprint |
| Predictive policy improvement | [Inference-Time Enhancement of Generative Robot Policies via Predictive World Modeling](https://arxiv.org/abs/2502.00622) | 2025 preprint |
| Bimanual benchmark | [RoboTwin 2.0](https://arxiv.org/abs/2506.18088) | 2025 preprint/project |
| Embodied security | [Security of Foundation-Model-Powered Embodied Agents](https://arxiv.org/abs/2608.16843) | 2026 preprint |

## 2026 research index

The following entries were retrieved from the official arXiv API on 2026-08-19 and screened by title, abstract and topic. Every entry is a `preprint`. arXiv availability does not imply peer review, public code, public data or local reproducibility. Read the paper, supplements and repository before using a result as evidence.

### Surveys, evaluation, governance and safety

| Paper | Focus |
|---|---|
| [Vision-Based Tactile Intelligence for Robotics](https://arxiv.org/abs/2608.15490) | Survey of visual-tactile sensing, learning and manipulation |
| [Learning Physical Interaction](https://arxiv.org/abs/2608.07558) | Survey of tactile/force-aware robot learning |
| [Weights or Skills?](https://arxiv.org/abs/2608.01851) | From policy weights to composable robot skills |
| [How Should World Models Be Evaluated for Embodied Decision-Making?](https://arxiv.org/abs/2606.15032) | Decision-centric world-model evaluation |
| [Security of World-Model-Based Embodied AI](https://arxiv.org/abs/2607.28226) | Lifecycle threats and defenses for world-model systems |
| [Security of Foundation-Model-Powered Embodied Agents](https://arxiv.org/abs/2608.16843) | Attack surfaces, defenses and evaluation |
| [A Comprehensive Survey and Systematic Real-World Evaluation of Embodied Vision-and-Language Navigation](https://arxiv.org/abs/2607.09792) | Real-world embodied VLN evaluation |
| [Physical AI Governance](https://arxiv.org/abs/2607.22877) | Lifecycle governance for physical AI |
| [H2R-Bench](https://arxiv.org/abs/2608.13049) | Human-video to robot-manipulation generation benchmark |
| [HumanoidVLN](https://arxiv.org/abs/2608.12860) | Physics-grounded VLN benchmark across humanoids |
| [FlatLab](https://arxiv.org/abs/2608.14049) | Simulation benchmark for flat-object manipulation |
| [360CityArena](https://arxiv.org/abs/2608.08814) | Open-city embodied navigation benchmark |
| [Compiling and Benchmarking Task-State Horizons for Embodied Agents](https://arxiv.org/abs/2608.08036) | Task-state horizons and long-horizon evaluation |
| [WorldSimProbe](https://arxiv.org/abs/2608.09298) | Simulator-faithfulness diagnosis for action-conditioned world models |
| [Explore, Map, Remember, Decide](https://arxiv.org/abs/2608.08077) | Embodied VLMs in safety-critical scenarios |
| [How Should I Pick a Foundation Model for My Robot?](https://arxiv.org/abs/2608.06898) | Community evaluation framework for robot foundation models |
| [Agentic Harnesses](https://arxiv.org/abs/2608.09857) | LLM verification layers for robot autonomy |
| [Failing Gracefully](https://arxiv.org/abs/2608.05313) | Mitigating the impact of inevitable robot failures |
| [Toward Certified Functional Safety for Industrial Humanoid Robots](https://arxiv.org/abs/2608.02809) | Functional safety and the fail-passive gap |
| [CoCoNav](https://arxiv.org/abs/2608.07751) | Conformal control for safe navigation in crowds |
| [VLAGuard](https://arxiv.org/abs/2608.01028) | Evaluation and mitigation of physical attention hijacking |
| [Hijacking Robots with a Piece of Paper](https://arxiv.org/abs/2608.05715) | Physical prompt injection against VLM-controlled robots |
| [Structure-Aware Robust Fine-Tuning](https://arxiv.org/abs/2608.03231) | Defense against physical attention hijacking |
| [Bit-Flip Attacks on Vision-Language-Action Models](https://arxiv.org/abs/2608.15475) | Security weakness of action-decoding architectures |

### VLA, long-horizon tasks and policy learning

| Paper | Research entry point |
|---|---|
| [tau_0-VLA](https://arxiv.org/abs/2608.16885) | Hierarchical VLA with world-model-guided test-time computation |
| [HAF](https://arxiv.org/abs/2608.16837) | Hierarchical action flow and latent RL for humanoid loco-manipulation |
| [NebulaVLA](https://arxiv.org/abs/2608.16503) | Dual-frequency VLA with guide actions |
| [SparkVLA](https://arxiv.org/abs/2608.16172) | Stop-aware hierarchy and adaptive action chunks |
| [ViTaR](https://arxiv.org/abs/2608.15816) | Visuo-tactile residual adaptation |
| [Algorithm-Architecture Co-Design for Efficient VLA Inference](https://arxiv.org/abs/2608.15636) | Speculative inference and verification |
| [EcoVLA](https://arxiv.org/abs/2608.15502) | Device-edge co-inference under energy constraints |
| [StructRL](https://arxiv.org/abs/2608.15139) | Structured action-space exploration for flow VLAs |
| [Imagining Recovery](https://arxiv.org/abs/2608.14822) | Counterfactual recovery at inference time |
| [Reflex](https://arxiv.org/abs/2608.14379) | Fast predictive VLA for reaction-critical manipulation |
| [Evolve Vision-Language-Action Model into an Agent](https://arxiv.org/abs/2608.14047) | On-the-fly tool use in a VLA |
| [Decoding Task Progress from VLA Representations](https://arxiv.org/abs/2608.13474) | Reading task progress from learned representations |
| [Temporal GRPO](https://arxiv.org/abs/2608.13026) | Temporal credit assignment for VLA reinforcement learning |
| [StellaVLA](https://arxiv.org/abs/2608.11671) | In-context structured demonstrations |
| [VANE](https://arxiv.org/abs/2608.09448) | Test-time training via future visual prediction |
| [JEPA-WAM](https://arxiv.org/abs/2608.09381) | Joint-embedding world modeling for VLA policies |
| [GWM-VLA](https://arxiv.org/abs/2608.07619) | Geometry-aware latent world modeling |
| [Capek 0.5](https://arxiv.org/abs/2608.06756) | Execution-centric vision-language model |
| [Beyond Flat Policies](https://arxiv.org/abs/2608.05999) | Hierarchical post-training for embodied agents |
| [In-Context VLA](https://arxiv.org/abs/2608.05738) | Language via in-context post-training and tools |
| [SpaceVLA](https://arxiv.org/abs/2608.05730) | Spatially grounded grasp/place anchors |
| [BridgeVLA++](https://arxiv.org/abs/2608.05042) | Data-efficient 3D manipulation and memory |
| [Explicit Language Memory for Long-Horizon Planning](https://arxiv.org/abs/2608.04765) | Language memory for long-horizon planning |
| [GUARD](https://arxiv.org/abs/2608.04510) | Grounding uncertainty and risk detection |
| [SAFECAST](https://arxiv.org/abs/2608.04246) | Contrast-set training and calibration for failure detection |
| [Deltoris](https://arxiv.org/abs/2608.04428) | Bit-level sparsity and speculative real-time inference |
| [Track4Action](https://arxiv.org/abs/2608.03727) | Distilling world-centric 3D tracking into VLA |
| [DRIFT](https://arxiv.org/abs/2608.03207) | Adversarial attacks on flow-matching VLA denoising |
| [How Should Vision-Language-Action Models Use Proprioceptive State?](https://arxiv.org/abs/2608.03052) | Proprioceptive input and control quality |

### World models, planning, simulation, data and embodiment transfer

| Paper | Research entry point |
|---|---|
| [DreamX-Phi 1.0](https://arxiv.org/abs/2608.13489) | Action-conditioned video world model |
| [Ontology-Grounded World Models](https://arxiv.org/abs/2608.13901) | Failure diagnosis and closed-loop repair |
| [SLIM-0.5B](https://arxiv.org/abs/2608.09771) | Action-grounded predictive latents |
| [XEWorld](https://arxiv.org/abs/2608.05799) | World-model generalization to unseen embodiments |
| [Toward the Cognitive--Physical Limits of Embodied Intelligence](https://arxiv.org/abs/2608.10618) | World-model-centric autonomous racing agent |
| [GraphThink](https://arxiv.org/abs/2608.07905) | Graph-enhanced LLM planning for long-horizon tasks |
| [From Failures to Supervision](https://arxiv.org/abs/2608.00613) | Failure-driven supervision for robust planning |
| [Long-Horizon Embodied Decision-Making via Multimodal Memory Compression](https://arxiv.org/abs/2608.01456) | Memory compression for long-horizon decisions |
| [When Replanning Becomes the Bottleneck](https://arxiv.org/abs/2608.01428) | Budgeted replanning |
| [PACE: Adaptive Budget Allocation for Time-Efficient Embodied Planning](https://arxiv.org/abs/2608.03034) | Adaptive planning compute budgets |
| [GaussMemory](https://arxiv.org/abs/2608.14986) | Task-driven 3D Gaussian scene memory |
| [Remember Smarter](https://arxiv.org/abs/2608.15269) | Visual-history compression and experience space |
| [Revisiting Open-Loop Execution in Robotics](https://arxiv.org/abs/2608.15938) | Open-loop versus reactive execution |
| [Robo-Dopamine 2.0](https://arxiv.org/abs/2608.15680) | History-conditioned and OOD-aware process rewards |
| [PACE: Phase-Progress-Aware Credit](https://arxiv.org/abs/2608.15026) | Phase-progress credit for long-horizon manipulation |
| [SkillComposer](https://arxiv.org/abs/2608.14944) | Reusable skills for natural-language robot programming |
| [Scaling Manual-Grounded Appliance Manipulation](https://arxiv.org/abs/2608.15863) | Data synthesis and unified planning |
| [FloodReasonBench](https://arxiv.org/abs/2608.15410) | Edge embodied-VLM benchmark for flood response |
| [Discovering Diverse Planning Policies](https://arxiv.org/abs/2608.08523) | Quality-diversity optimization for multimodal planning |
| [TrustRoboReward](https://arxiv.org/abs/2608.08491) | Calibrated preference-ordered robot rewards |
| [Mimir](https://arxiv.org/abs/2608.04933) | Neuro-symbolic memory in interactive environments |
| [BWM](https://arxiv.org/abs/2607.29302) | Low-cost, high-fidelity world simulator |
| [AquaJEPA](https://arxiv.org/abs/2607.29393) | Action-conditioned multimodal JEPA for underwater dynamics |
| [Self-Evolving Learning for Embodied AI](https://arxiv.org/abs/2607.28251) | Criticality-model-guided self-evolving learning |
| [RoboBRIDGE](https://arxiv.org/abs/2607.27881) | Bridging policies to robust real-world robots |
| [Cross-Embodiment Transfer via Behavior-Aligned Representations](https://arxiv.org/abs/2607.27549) | Behavior-aligned cross-embodiment transfer |
| [Failure Detection for Surgical Robot Imitation Policies](https://arxiv.org/abs/2607.27511) | Flow-matching world model for failure detection |
| [From Passive Video to Editable Experience](https://arxiv.org/abs/2607.26903) | Physically grounded experience synthesis |
| [Counterfactual Action Sensitivity Coverage](https://arxiv.org/abs/2607.27261) | Counterfactual coverage for robust imitation |

Read in this order: survey and evaluation first, one method paper second, then a same-condition W8--W11 baseline. For each paper record the arXiv ID, version date, code/data status, license, hardware requirements and failure boundary. Do not turn a title, abstract or submission date into a claim of superiority. Re-run the search quarterly and remove withdrawn, duplicate or unlocatable records.

## Course and resource map

| Resource | Type | Where it fits |
|---|---|---|
| [Stanford CS336](https://cs336.stanford.edu/) | research-style course | W1--W3: tokenizer, Transformer, systems, data and alignment |
| [Stanford CS224N](https://web.stanford.edu/class/cs224n/) | NLP/LLM course | W1, W3: attention and representations |
| [Stanford CS231n](https://cs231n.stanford.edu/) | vision course | W5: training, transfer and visual projects |
| [Berkeley CS285](https://rail.eecs.berkeley.edu/deeprlcourse/) | deep-RL course | W9: imitation, model-based and offline RL |
| [Full Stack Deep Learning](https://fullstackdeeplearning.com/course/) | engineering course | W2, W4, W11: data, testing, deployment and monitoring |
| [FSDL LLM Bootcamp](https://fullstackdeeplearning.com/llm-bootcamp/) | LLM engineering course | W4: RAG, agents and LLMOps |
| [HF LLM Course](https://huggingface.co/learn/llm-course) | open course | W1, W3: Transformers, data, tuning and inference |
| [HF Computer Vision Course](https://huggingface.co/learn/computer-vision-course) | open course | W5--W6: ViT, multimodal, video, 3D and optimization |
| [HF Deep RL Course](https://huggingface.co/learn/deep-rl-course) | open course | W9 optional; page marked low-maintenance |
| [HF Robotics Course](https://huggingface.co/learn/robotics-course) | open course | W7, W10: classic robotics, LeRobot and learning |
| [Modern Robotics](https://modernrobotics.northwestern.edu/) | textbook and videos | W7: poses, kinematics, dynamics and control |
| [LeRobot](https://github.com/huggingface/lerobot) | open platform | W7, W10--W11: data, policies, simulation and evaluation |
| [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) | simulation/RL platform | W9 optional; use a pinned release |
| [MuJoCo](https://mujoco.readthedocs.io/) | physics simulator | W7--W9 light route |
| [ManiSkill](https://maniskill.ai/) | manipulation benchmark | W7--W10 selective use |
| [LIBERO](https://libero-project.github.io/) | manipulation benchmark | W10 standard evaluation |
| [RoboMimic](https://robomimic.github.io/) | imitation-learning toolkit | W7, W10 baseline |
| [Open X-Embodiment](https://robotics-transformer-x.github.io/) | cross-robot data | W10 data and generalization |
| [NVIDIA Training](https://www.nvidia.com/en-us/training/online/) | engineering training | CUDA, deployment and Isaac topics |
| [DeepLearning.AI](https://www.deeplearning.ai/) | provider courses | GenAI, agents and tuning as targeted supplements |

Run at most two main courses in parallel. Keep videos, assignments, code and project evidence as separate records.

## Evaluation protocol

### Run manifest

```yaml
run_id: emc-2026-001
git_commit: TBD
environment: synthetic | sim | bag_replay | real_hardware
os: TBD
python: TBD
torch: TBD
cuda: TBD
gpu: TBD
model_id: TBD
dataset_id: TBD
seed: 0
config: configs/example.yaml
metrics: reports/results.jsonl
limitations:
  - TBD
```

### Metrics

| Module | Primary metrics | Always report |
|---|---|---|
| LLM | task accuracy, format validity | p50/p95 latency, throughput, memory, refusal |
| RAG | Recall@k, MRR/nDCG, citation support | unanswerable queries, retrieval latency, citation errors |
| VLM | F1, IoU/hit rate, calibration | occlusion/lighting degradation, schema errors, latency |
| World model | 1/5/10-step error, reward MAE | OOD, interval coverage, rollout collapse, planning latency |
| Policy/VLA | episode success | collision, out-of-bounds, action violation, intervention, variance |
| System | end-to-end success | p95 latency, resources, recovery and state age |

Separate paper results, public benchmarks, simulation results and hardware results. Do not compare across evidence levels.

### Research quality gate

Write "improves" only when the result uses the same data, environment, budget and evaluation script as a reasonable baseline; splits, seeds, versions and configurations are traceable; at least one ablation explains the source of the change; failures and scope limits are reported; and resource-limited single-seed or approximate runs are explicitly labeled.

## Engineering standards

### Repository shape

```text
emc-12week/
  README.md  LICENSE  CITATION.cff  CONTRIBUTING.md  CHANGELOG.md
  pyproject.toml  environment.lock
  configs/  src/emc/  tests/  scripts/  experiments/  reports/
  docs/architecture.md  docs/interfaces.md  docs/safety.md  docs/deployment.md
  assets/architecture.svg  assets/demo.mp4  assets/plots/
  data/README.md  data/manifests/
  third_party/licenses.md  third_party/sources.md
```

### Quality gates

```text
make setup
make lint
make test
make smoke
make eval
make report
```

At minimum, unit-test schema, timeout, retry, data splitting and the safety wrapper. Integration-test one complete simulated episode. CI checks formatting, dependencies, licenses and sensitive files. Every release includes result JSONL and configuration.

### Production checklist

- Service: health check, request ID, timeout, retry, rate limit and version routing.
- Data: quality checks, version, lineage, deduplication, privacy and data card.
- Model: checkpoint, model card, quantization, memory and input/output contract.
- Runtime: logs, metrics, traces, resource watermarks, failure replay and rollback.
- Robot: timestamps, frames, QoS, action limits, emergency stop and human takeover.
- Collaboration: design docs, issues, PRs, changelog and reproducible README.

### Safety and licenses

Prefer Apache-2.0, MIT and BSD-compatible sources, but check code, weights, datasets, assets and commercial conditions separately. OpenVLA code is not equivalent to its base-model license. Record GR00T code, gated weights and dependencies separately. Keep GPL, unknown-license and unconfirmed code isolated. Never publish private maps, bags, tokens, serial numbers or competition rules.

## Graduation rubric

| Area | Points | Evidence |
|---|---:|---|
| Theory and papers | 15 | 12 core paper notes explaining assumptions and limits |
| Model implementation | 20 | tiny Transformer, VLM/retrieval tool and world model |
| Experiments and evaluation | 25 | baselines, metrics, ablations, failures and multiple seeds |
| Robotics and system | 20 | policy episodes, simulator/ROS 2 interface and safety tests |
| Engineering quality | 10 | tests, CI, locks, logs, manifest and licenses |
| Public communication | 10 | README, architecture diagram, two-minute demo and report |

Pass with at least 70 points, at least half of both the evaluation and robotics categories, and a capstone that runs a smoke test from a clean environment.

## Research upgrade path

After 12 weeks, choose one question and iterate as `hypothesis -> baseline -> ablation -> failure boundary -> next hypothesis`.

| Observation | Research question | First experiment |
|---|---|---|
| RAG has evidence but the task is still wrong | evidence selection, conflicts or abstention calibration | dense vs hybrid vs evidence reranking |
| VLM recognition is unstable | confidence, active re-observation or missing modalities | VLM vs VLM plus verifier |
| World-model rollouts diverge | uncertainty, short-horizon planning or action delay | ensemble gate vs no gate |
| Simulation success does not generalize | coverage, randomization or invariant representations | no randomization vs three-factor randomization |
| VLA latency is too high | action chunks, quantization, caching or frequency hierarchy | latency-success Pareto curve |
| LLM sub-goals are unreliable | state feedback and recovery | scripted planner vs LLM planner |

State which results are reproductions, engineering changes or possible new methods. Do not package integration work as algorithmic novelty without new data, a new baseline or new analysis.

## Public repository checklist

- [ ] The README first screen has the goal, demo image/video, quick start and results table.
- [ ] One command installs the project, runs a smoke test and executes the smallest evaluation.
- [ ] Public data or download scripts include license, hash and version.
- [ ] Figures are generated from scripts and JSONL, never hand-edited.
- [ ] The model card contains source, training data, limitations and use conditions.
- [ ] The data card describes splits, privacy, bias and deletion.
- [ ] `LICENSE`, `CITATION.cff` and third-party notices are complete.
- [ ] No token, private path, network address, map, bag, serial number or unauthorized weight is included.
- [ ] Issue templates request bug, reproduction and environment information.
- [ ] Release notes state known failures, compatible versions and rollback steps.

## Maintenance and synchronization

Review the main route quarterly: course pages, repository activity, versions, licenses, model cards, benchmarks and hardware requirements. A new paper enters the frontier index only after its code, data, license and local resource requirements are checked.

The two language files must be updated together. Keep section names, week IDs, milestone names, evaluation thresholds and arXiv IDs aligned. A paper may be added in one language only temporarily during editing; before committing, run a link and ID comparison and resolve the mismatch.

The goal is not to collect the largest list of models. It is to build transferable ability: understand principles, write implementations, create data, design evaluation, analyze failures, deploy safely and deliver results that another person can reproduce.
