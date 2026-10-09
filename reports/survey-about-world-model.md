# World Models: A Research Survey

## TL;DR

A world model is best understood not as one architecture but as a learned internal simulator: it compresses observations into a state, predicts how that state evolves (often conditional on actions), and supplies predictions for planning, control, or simulation. Early systems made this idea concrete by learning compact visual representations and stochastic latent dynamics; subsequent work diverged into actor–critic learning through imagined experience, search over task-relevant latent predictions, and generative models that render interactive futures. [1][2][3][4]

The central design choice is what the model must predict. Reconstructive and video-generative models can expose plausible future observations, while control-centric models may deliberately omit pixel reconstruction and focus on rewards, values, or latent transitions useful for decisions. Neither objective dominates universally: visual fidelity does not guarantee faithful action consequences, and a model adequate for a narrow controller may not function as a general simulator. [4][5][6][7]

Recent work extends these approaches to interactive video, physical robots, navigation, driving, and policy evaluation. Hugging Face-indexed work on DayDreamer, Navigation World Models, IRASim, and DrivingDojo illustrates the move from learned latent control toward action-conditioned visual simulation and broader interaction data. [8][9][10][11] Progress is real, but robust long-horizon consistency, causal and counterfactual reliability, action coverage, uncertainty calibration, real-time inference, and decision-relevant evaluation remain unresolved. [12][13][14][15]

## Background

The term “world model” has several related uses. In model-based reinforcement learning it typically means a learned representation and transition/reward model that can support imagined rollouts; in generative-video research it can mean an action-conditioned predictor that renders what an environment may look like after an intervention. Some models target explicit state, some latent predictive features, and some visual output. These are overlapping but not interchangeable goals. The distinction matters because claims of “world understanding” or “simulation” require evidence that predictions remain useful under changed actions and over the time horizons relevant to downstream tasks. [2][4][5][13]

A useful abstraction is an observation encoder, a latent state or belief, an action-conditioned transition, and one or more heads for observation, reward, value, or policy. The latent state may summarize history under partial observability, while stochastic transitions can represent multiple plausible futures. The learned model can then be used in at least two ways: planning explicitly over candidate actions, or training a policy/critic from imagined trajectories. [2][3]

The early “World Models” formulation separated a visual compressor, a recurrent dynamics model, and a compact controller. The dynamics model predicted a distribution over future latent states, and a controller trained in generated “dream” trajectories could transfer to the environment; the authors also warned that controllers may exploit simulator errors. [1] PlaNet developed a recurrent state-space model for pixel-based control and used model-predictive control (MPC) to search over action sequences in latent space, replanning after observations. Dreamer instead optimized actor and critic behavior using imagined latent trajectories, with a value estimate to account for rewards beyond its finite imagination horizon. [2][3] These contrasting lines—online search versus policy learning through latent imagination—continue to shape the field.

## Themes

### 1. Latent dynamics and decision-focused models

Latent world models trade direct prediction of high-dimensional observations for a compact state representation. PlaNet combines deterministic recurrent state with stochastic latent state, learns transition, reward, and observation components, and plans by MPC in the learned space. Its multi-step latent “overshooting” objective is intended to improve predictions over the horizons planning needs; the paper also identifies accumulated model error and overconfident out-of-distribution predictions as risks. [2]

Dreamer shifts computation from repeated action search at deployment to learning an actor and critic from imagined experience. Gradients through the learned dynamics optimize the actor, while the critic estimates returns beyond the rollout window. DreamerV3 broadens this family with a fixed configuration evaluated across more than 150 tasks and reports Minecraft diamond collection from scratch; its reported recipe includes normalization and transformed targets to handle variation in reward scales. These results show that latent imagination can scale across task families, but they do not imply that every imagined trajectory is a reliable physical simulation. [3][16]

MuZero makes a different compromise: its hidden dynamics need only predict quantities relevant to search—reward, policy, and value—not reconstruct the observation or recover the true environment state. It uses Monte Carlo Tree Search with the learned model and reports results across Atari and board games. This objective can allocate model capacity to decision-relevant predictions, but its deterministic dynamics are a limitation for genuinely stochastic settings. [4] TD-MPC2 likewise uses a decoder-free, control-centric latent model, predicting latent transitions, reward, terminal value, and a policy prior; it optimizes short candidate trajectories with MPC rather than generating future pixels. The authors report evaluation across 104 continuous-control tasks, but such a control-specific success criterion should not be read as general visual simulation capability. [5]

The broader lesson is objective alignment. Reconstructing observations can provide rich supervision, but visual error is not necessarily proportional to action-selection error. Conversely, a model trained only to support control may discard details needed for other tasks or human inspection. Model choice should follow the intended downstream use, and evaluations should state clearly what the model is asked to preserve. [4][5][13]

### 2. Interactive video and generative environments

Generative world models aim to make predicted futures visible and to condition them on user or agent actions. Genie learns a video tokenizer, autoregressive dynamics model, and latent actions from unlabeled Internet videos; it can infer controls without ground-truth action labels. The paper demonstrates promptable interactive environments built from unlabeled video and inferred latent actions, offering a route to controllable generation without ground-truth action labels. [17] DeepMind’s Genie 2 announcement describes an autoregressive latent-diffusion model conditioned on individual keyboard or mouse actions and reports playable environments generated from a single image; the announcement also characterizes the work as early-stage and notes consistency/generalization limitations. [18]

A key technical challenge is adapting video generators that were trained to continue or condition on clips, rather than to respond faithfully to sequences of interventions. Dynamic World Simulation introduces an action-trajectory-conditioned adaptation module and a motion-reinforced training loss intended to align actions with visual changes. AVID instead adapts a pretrained image-to-video model through an action-conditioned adapter, targeting the practical situation where action-labeled videos and access to model weights are limited. These approaches retain generative priors while explicitly strengthening action conditioning, but the quality of generated video alone cannot establish correctness of the underlying dynamics. [6][7]

Other systems emphasize interaction and longer context. GameGen-X tunes a pretrained video-generation system for control of open-world game video. VRAG identifies limited memory and compounding errors as coupled problems for autoregressive rollouts and proposes global-state conditioning with video retrieval. The interactive-generative-video survey organizes capability needs around generation, control, memory, dynamics, and intelligence, emphasizing that these dimensions jointly affect utility. Together, this work indicates that a visually convincing first few seconds are not enough: useful interactive models need persistent state, consistent response to actions, and temporal coherence over extended rollouts. [6][12][13]

### 3. Embodied robotics, navigation, and driving

Robotics tests whether learned predictions can connect visual observations to physical actions. DayDreamer applies latent-imagination learning on physical robots, reporting online learning across locomotion, manipulation, and navigation tasks without relying on a simulator. It illustrates a complementary route to sim-to-real transfer: learn on the real system, where data collection itself must be safe and efficient. [8]

Generative visual simulators aim to make action outcomes inspectable. IRASim conditions generated robot-arm videos on action trajectories and historical observations; its evaluation uses three real-robot datasets, and the paper reports that its model-based planning improved Push-T IoU from 0.637 to 0.961 in one experiment. The paper also reports a correlation between world-model and ground-truth-simulator policy evaluations. Navigation World Models predicts future observations from visual history and navigation actions, then uses candidate trajectories to reach or rank goals, including in unfamiliar environments. Both approaches treat action-conditioned video as more than passive forecasting by using it in planning or evaluation. [9][10]

Driving highlights the importance of data coverage and closed-loop interaction. DrivingDojo is designed to represent varied maneuvers, multi-agent interaction, and open-world events, and includes an action-instruction-following benchmark. UniSim takes a related but distinct approach, turning recorded sensor logs into a closed-loop neural simulator that can render new camera/LiDAR observations under altered actor configurations and vehicle behavior. The contrast is informative: scalable learned video data can provide maneuver diversity, while reconstructed scene simulators can support controlled counterfactual changes; neither automatically solves rare-event coverage or full physical fidelity. [11][19]

Across these domains, action conditioning is a prerequisite, not a guarantee. Labels can be sparse or noisy, and videos may underspecify forces, geometry, contact, or intent. The system must predict the consequences of an action that was not merely correlated with the observed next frame. Consequently, robot and driving evaluations should measure downstream decisions, interactions, and closed-loop behavior—not only visual similarity. [7][10][11][19]

### 4. Evaluation: prediction quality versus useful simulation

Evaluation should separate several claims: whether a model predicts observations, follows interventions, supports planning, and improves decisions in a real or trusted environment. A reconstruction or next-frame score primarily measures predictive similarity under the test distribution; it does not by itself establish counterfactual validity, long-horizon stability, policy-ranking accuracy, or safe policy optimization. The interactive-video survey and recent benchmarking work both emphasize evaluation beyond visual plausibility. [13][14][15]

WorldGym treats an action-conditioned video model as a proxy environment for policy evaluation, combining rollouts with a vision-language reward function. Its results report correlation between model-based and real-world success and preservation of relative policy rankings in tested comparisons, but also identify realistic object interaction as difficult and cite calibration concerns for out-of-distribution actions. This is evidence of potential utility in a specific evaluation pipeline, not proof that a generated world is interchangeable with the physical environment. [14]

WorldTest takes a different route: it separates reward-free exploration from tests in related environments and scores external behavior across prediction, planning, and changes to causal dynamics. Its results are a reminder that success on familiar-task reward or frame prediction may not transfer to varied inference tasks. Together, these approaches motivate multi-level evaluation: visual/structural diagnostics; action-following and counterfactual tests; full-horizon closed-loop rollouts; uncertainty and calibration checks; and direct measures of policy utility or ranking agreement. [15][14]

## Trends and open problems

1. **From passive prediction to intervention-faithful dynamics.** Video models are being adapted with action channels, control adapters, and trajectory conditioning. The hard question is whether generated changes reflect action consequences rather than visual correlations; action-labeled data, diverse interventions, and counterfactual tests are necessary to make this distinction. [6][7][10]

2. **Memory and long-horizon state consistency.** Autoregressive rollouts compound local errors, and longer context alone may not maintain persistent object or scene state. Retrieval and global-state conditioning are one response, while latent-state models and explicit simulator state provide alternatives. Strong benchmarks must test revisitation, extended rollouts, and scene dynamics, not just short clips. [2][12][13]

3. **Uncertainty, partial observability, and causal identification.** Stochastic latent states can represent ambiguity, but a model can still be confidently wrong when actions or environments depart from training data. Identifying causal effects from logged observations is difficult when actions are confounded or important state is hidden. Belief updating and uncertainty calibration therefore belong in both model design and evaluation. [1][2][14][15]

4. **Representation versus rendering.** A productive design space spans implicit video prediction, compact control-centric latent dynamics, and explicit geometry or scene-state simulation. Control-focused models can be efficient without reconstructing pixels, whereas generative models expose predictions and can support visual planning; hybrid systems may combine state, dynamics, and learned appearance. No single representation is optimal for all tasks. [4][5][6][19]

5. **Data coverage and real-time interaction.** Internet video provides scale but often lacks reliable action labels; robotics data can align actions and outcomes but is expensive and narrow; driving datasets must cover rare interactions. Generative quality and inference speed also trade off with responsive interaction. Work such as DrivingDojo's coverage-oriented dataset and IRASim's reported non-real-time limitation make these complementary bottlenecks explicit. [17][10][11]

6. **Evaluation aligned to intended use.** The field needs reproducible protocols that distinguish rendering quality from action fidelity and downstream decision quality, compare against trusted environments, expose uncertainty, and test distribution shift. Policy evaluation via a learned simulator is promising only if its reward model, action support, optimizer, and rollout horizon are separately scrutinized; otherwise, apparent gains may be artifacts of the surrounding pipeline. [13][14][15]

In sum, world models are advancing along two connected axes: increasingly capable latent simulators for efficient decision-making, and increasingly interactive generative models for visual prediction and simulation. The most important near-term criterion is not whether a model looks like a world, but whether its representations and predictions remain calibrated, intervention-sensitive, and useful for the decisions it is meant to support. [5][13][14]

## References
[1] World Models. arxiv. https://arxiv.org/abs/1803.10122 (2018-03-27)
[2] Learning Latent Dynamics for Planning from Pixels. arxiv. https://arxiv.org/abs/1811.04551 (2019 (ICML 2019))
[3] Dream to Control: Learning Behaviors by Latent Imagination. arxiv. https://arxiv.org/abs/1912.01603 (2019-12-03)
[4] Mastering Atari, Go, chess and shogi by planning with a learned model. web. https://www.nature.com/articles/s41586-020-03051-4 (2020-12-23)
[5] TD-MPC2: Scalable, Robust World Models for Continuous Control. web. https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57b6dcda32b293df7c2d5341f49-Abstract-Conference.html (2024-05-31)
[6] Pre-Trained Video Generative Models as World Simulators. arxiv. https://arxiv.org/abs/2502.07825 (2025-02-10)
[7] AVID: Adapting Video Diffusion Models to World Models. arxiv. https://arxiv.org/abs/2410.12822 (2024-11-24)
[8] DayDreamer: World Models for Physical Robot Learning. hf-search. https://huggingface.co/papers/2206.14176 (2022-06-27)
[9] Navigation World Models. hf-search. https://huggingface.co/papers/2412.03572 (2024-12-04)
[10] IRASim: Learning Interactive Real-Robot Action Simulators. hf-search. https://huggingface.co/papers/2406.14540 (2024-06-20)
[11] DrivingDojo Dataset: Advancing Interactive and Knowledge-Enriched Driving World Model. hf-search. https://huggingface.co/papers/2410.10738 (2024-10)
[12] VRAG: Learning World Models for Interactive Video Generation. arxiv. https://arxiv.org/abs/2505.21996 (2025-05)
[13] A Survey of Interactive Generative Video. arxiv. https://arxiv.org/abs/2504.21853 (2025-04-30)
[14] WorldGym: World Model as An Environment for Policy Evaluation. arxiv. https://arxiv.org/abs/2506.00613 (2025-06)
[15] Benchmarking World-Model Learning. web. https://arxiv.org/html/2510.19788v2 (N/A)
[16] Mastering diverse control tasks through world models. web. https://www.nature.com/articles/s41586-025-08744-2 (2025-04-02)
[17] Genie: Generative Interactive Environments. arxiv. https://arxiv.org/abs/2402.15391 (2024-02-23)
[18] Genie 2: A large-scale foundation world model. web. https://deepmind.google/blog/genie-2-a-large-scale-foundation-world-model/ (2024-12-04)
[19] UniSim: A Neural Closed-Loop Sensor Simulator. arxiv. https://arxiv.org/abs/2308.01898 (2023-08-03)
