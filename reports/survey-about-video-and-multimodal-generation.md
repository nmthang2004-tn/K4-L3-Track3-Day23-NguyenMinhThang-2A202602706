# Video and Multimodal Generation: A Research Survey

## TL;DR

Video generation has moved from adapting image-generation systems toward spatiotemporal latent models, with diffusion and flow-matching Transformers now central and autoregressive designs increasingly explored for streaming and long-form output. The central trade-off is between parallel, high-quality denoising and the low-latency, extendable generation of causal models; both still struggle with sustained motion and scene consistency over long horizons. [1][2]

Multimodal progress means more than text-to-video: current systems condition on images, references, and controls, while some generate synchronized audio or unify understanding and generation across modalities. These are distinct capabilities—systems that ingest video and return speech or text should not be mistaken for systems that generate video. [3][4][5][6]

The field’s next bottlenecks are temporal and physical coherence, controllable composition, efficient long-duration inference, representative and well-documented training data, and evaluation that measures multiple dimensions rather than relying on one score. [7][8][9][10]

## Background

A video generator must synthesize a sequence whose frames are individually plausible and mutually consistent. Relative to images, video adds time as a modeling dimension: subjects, objects, camera motion, and scene state must evolve in a coherent way. Text-to-video (T2V) starts from a text prompt; image-to-video (I2V) adds a visual starting point or reference; further conditioned tasks may provide camera paths, identity, pose, depth, audio, or trajectories. A text prompt alone can be too underspecified for fine-grained requirements. [1][11]

The dominant practical recipe is to compress video into a latent representation and generate in that space, rather than directly denoising every pixel. Video diffusion systems often extend image-generation backbones with 3D convolution or attention, or factor spatial and temporal processing to reduce the cost of full spatiotemporal computation. Latent compression reduces the representation size, but designing temporal modules remains critical: full, factorized, causal, and sparse-causal attention encode different trade-offs in cost and temporal interaction. [1]

The field has also broadened from a narrow text-conditioned synthesis task into a multimodal generation and interaction problem. Some systems use text, images, or references to condition a video generator; others combine modalities in one model, while separate multimodal assistants understand images, audio, and video and generate text or speech. This survey distinguishes those scopes and compares systems by representation, conditioning, temporal strategy, and the evidence used to assess them. [3][5][12][6]

## Themes

### 1. From latent diffusion to flow matching and causal generation

Diffusion models generate by iterative denoising. For video, the model must denoise spatiotemporal representations while preserving motion and cross-frame consistency. Surveyed designs include video U-Nets and Transformers, with temporal processing added through 3D operations or factorized spatial/temporal modules. Factorization can lower the cost of full 3D computation, while latent-space generation reduces the size of the object being denoised. These methods offer powerful parallel synthesis, but generating a long sequence at once can exceed available hardware capacity. [1]

Flow-based models offer a related but differently parameterized route. Goku, for example, uses rectified-flow Transformers and a shared 3D VAE latent space for joint image and video generation; its authors report 2B- and 8B-parameter configurations and use memory-oriented techniques for long multimodal token sequences. The contribution illustrates a move toward shared image/video representations, but a paper’s benchmark results should be understood as results under that paper’s specific setup, not as a general proof that flow models outperform diffusion. [13]

A second design axis is whether a model generates a complete sequence bidirectionally or emits it causally in chunks or frames. Bidirectional denoising can use context from the full sequence, but has latency and memory costs incompatible with immediate streaming. Autoregressive generation naturally extends a prefix and can produce output incrementally, but errors and visual drift can accumulate with each step. CausVid addresses this tension by distilling a pretrained bidirectional diffusion Transformer into a chunkwise causal diffusion model; the authors report a four-step student distilled from a 50-step teacher and use bidirectional attention within chunks with causal attention across them. [2]

Pyramidal Flow Matching takes another efficiency path: it uses a temporal pyramid to compress video history for autoregressive generation, and a single end-to-end flow-matching objective rather than independently optimized cascade stages. LTX-Video instead emphasizes a highly compressed latent representation and jointly optimizes its video VAE and denoising Transformer. These approaches differ in where they spend complexity—history representation, multi-scale synthesis, or token compression—but share the goal of reducing the cost of high-resolution spatiotemporal generation. Reported speed and resolution numbers are model- and hardware-specific and are not directly comparable without matched settings. [14][15]

### 2. Long-horizon generation, continuity, and interaction

Longer duration exposes weaknesses hidden by short clips: subjects may change appearance, motion may stall or repeat, and generated scenes may lose coherence. Chunkwise autoregression can extend duration but introduces sequential dependence and accumulated error; long videos also require memory and compute for conditioning on an expanding history. These concerns appear across video-generation surveys and motivate causal attention, compressed history, and streaming-oriented designs. [1][2][14]

LongLive represents the interactive direction: its Hugging Face Daily Papers entry describes frame-level autoregressive video diffusion with streaming prompt input, while the paper describes mechanisms for prompt changes and long-context inference. The reported speed and duration are author-reported results, not a universal performance guarantee. This design goal differs from simply producing a long static prompt-conditioned clip: users can alter the instruction during generation, making prompt transitions and stable ongoing scene state part of the problem. [16]

Long-horizon reliability is not solved by speed alone. Faster causal rollout can make interaction practical, but it still needs mechanisms to maintain dynamics, identity, and scene state. Training on long prefixes and distilling or supervising beyond short windows are promising research directions, but comparisons require shared durations, prompt protocols, hardware, and temporal-quality measures. Current papers report different clips, resolutions, sampling settings, and metrics, so headline FPS or duration claims should not be ranked as if they came from one controlled benchmark. [2][14][16]

### 3. Multimodal conditioning and unified models

A practical progression is to condition video generation on richer inputs. Sora’s technical report describes a diffusion Transformer over spacetime patches in latent codes, trained on images and videos with variable duration, resolution, and aspect ratio; it presents image and video inputs for animation, extension, and editing, while also acknowledging weaknesses in physical interactions and long-sample coherence. [17]

Image conditioning makes the first frame, reference identity, or visual composition available directly rather than requiring a text description to specify it all. HunyuanVideo-I2V documents an image-token pathway combined with video latents and text/caption information, and supports image-to-video generation. Runway Gen-4 emphasizes reference-guided consistency for subjects and locations across scenes. These are examples of product and model strategies for visual control; official announcements describe capabilities but do not provide a common independent benchmark for cross-system comparison. [3][18]

Audio expands the target from silent video to synchronized audiovisual output. Google DeepMind’s Veo 3 announcement describes video generation with audio, including ambient sound and dialogue, and highlights lip synchronization. This is a materially different generation target from adding a soundtrack after video synthesis: alignment between speech, sound, and visible events becomes an additional requirement. [4]

Unified multimodal systems span several architectures. Chameleon uses early fusion: image tokens and text tokens share a Transformer sequence, enabling interleaved text-and-image understanding and generation. This tight sharing contrasts with connector-based approaches that join a multimodal understanding model to a modality-specific generator; the latter can reuse pretrained components but risk an information bottleneck if only the connector is trained. A recent unification survey groups systems into autoregressive, diffusion, and hybrid families and emphasizes choices around tokenization, fusion, and modality-specific outputs. [5][12][19]

The term “multimodal generation” should not obscure task boundaries. OmniGen2 targets instruction-aligned image generation and editing, with separate text and image decoding paths. Qwen2.5-Omni accepts text, image, audio, and video, and generates text and natural speech; its time-aware multimodal input and streaming speech output make it relevant to multimodal interaction, but its report does not claim video generation. Keeping input and output modalities explicit prevents an understanding capability from being reported as a generation capability. [20][6]

### 4. Controllability: moving beyond a single prompt

A short text prompt is convenient but often cannot specify identity, motion, camera movement, layout, or timing precisely. The controllable-video survey categorizes conditions such as camera motion, depth, pose, identity, reference images, audio, and trajectories, and distinguishes single-condition, multi-condition, and more universal-control methods. This taxonomy frames controllability as a design space rather than one feature. [11]

Reference-based generation is one practical control family: the reference supplies appearance or scene information, while text describes desired changes or motion. Other conditions can specify structure or dynamics more directly. The trade-off is a more demanding interface and training problem: multiple signals must be reconciled, and evaluation must establish whether the output follows each instruction while remaining visually and temporally coherent. Existing taxonomies clarify what controls are being studied, but the available survey evidence does not establish one universally best control method or a single shared quantitative benchmark. [18][11]

### 5. Data, evaluation, and responsible evidence

Evaluation is multidimensional. VBench proposes 16 dimensions organized around video quality and condition consistency, with human preference annotations to examine alignment between benchmark scores and perception. Its analysis notes that a single metric such as FVD can compress away important strengths and weaknesses, and that results can trade temporal consistency against dynamic degree. A separate evaluation survey describes fragmented evaluation approaches spanning human-perception and instruction alignment, conventional metrics, and model-based judgments. Together these motivate reporting a profile of capabilities rather than one scalar score. [7][8]

Dataset composition affects what models learn and what benchmarks measure. VideoUFO introduces a million-scale, user-focused T2V dataset built around prompt topics, with brief and detailed captions; its authors report that topic coverage differs across existing datasets and that models vary by topic. The paper also notes limitations in inferring topic coverage from existing captions. This supports a broader lesson: scale alone is insufficient if the data and prompts underrepresent the uses on which systems will be judged. [21]

Provenance and licensing are also central. An audit of public text, speech, and video datasets reports that web-crawled, synthetic, and social-media data have become prominent, and identifies licensing and geographic/linguistic representation gaps. The report emphasizes that conclusions from public dataset documentation do not describe every private training corpus, but does show why dataset documentation, permission tracking, privacy, and representational coverage belong in the research agenda. [10]

Audio-video systems need evaluations that test cross-modal consistency, not only independent audio and video quality. A benchmark survey and recent benchmark proposals motivate measuring alignment and synchronization; however, similarity scores or multimodal-model judgments remain particular measurement choices and should be checked against human judgments and task-specific evidence. [4][8]

## Trends and open problems

1. **Temporal consistency is the core quality frontier.** Image-level realism does not guarantee persistent identity, plausible object interaction, or sustained motion. Future systems need better mechanisms for scene state and long-range dependencies, along with evaluations that expose drift over time rather than scoring isolated frames or short clips. [1][17][7]

2. **Efficiency and quality must be optimized together.** Latent compression, factorized attention, temporal pyramids, few-step distillation, and causal streaming each reduce a different cost. The open question is whether these techniques can support longer, higher-resolution output without trading away motion diversity or stability. Fair comparisons should disclose hardware, duration, resolution, sampling steps, and evaluation conditions. [1][2][14][15]

3. **Multimodal control needs compositional reliability.** Reference images, text, audio, and structural controls can make generation more useful, but combining them creates conflict and alignment problems. Models should be tested on whether requested attributes persist across time and whether audio and video agree, not just whether outputs appear plausible. [4][11]

4. **Unification remains an architectural and data challenge.** Early fusion, connectors, modality-specific decoders, and hybrid systems embody different compromises in shared representation, modularity, and information flow. Tokenization, cross-modal alignment, data construction, and evaluation remain open issues; results should state exactly which modalities are accepted and which are generated. [5][12][19][20][6]

5. **Benchmarks and data need broader coverage and transparency.** Disentangled metrics, human evaluation, topic-diverse prompts, and provenance documentation are complementary. No single benchmark currently resolves all questions about realism, alignment, dynamics, controllability, and safe or equitable performance, so evidence should combine metric suites, human assessment, and clear dataset/model disclosures. [7][8][10][21]

In summary, video and multimodal generation is becoming a systems field: representation learning, causal inference, controls, real-time interfaces, data governance, and evaluation all shape capability. Progress will be most meaningful when new systems demonstrate not just attractive short clips, but reliable temporal behavior, explicit modality alignment, controllability, and reproducible evaluation across diverse conditions. [1][11][7][9]

## References
[1] Video Diffusion Models: A Survey. arxiv. https://arxiv.org/abs/2405.03150 (2024-11-17)
[2] From Slow Bidirectional to Fast Autoregressive Video Diffusion Models. arxiv. https://arxiv.org/abs/2412.07772 (2025-09-23)
[3] HunyuanVideo-I2V: A Customizable Image-to-Video Model based on HunyuanVideo. web. https://github.com/Tencent/HunyuanVideo-I2V (2025-03-06)
[4] Fuel your creativity with new generative media models and tools. web. https://deepmind.google/blog/fuel-your-creativity-with-new-generative-media-models-and-tools/ (2025-05-20)
[5] Unified Multimodal Understanding and Generation Models: Advances, Challenges, and Opportunities. arxiv. https://arxiv.org/abs/2505.02567 (2025-05 (retrieved version v6))
[6] Qwen2.5-Omni Technical Report. hf-search. https://huggingface.co/papers/2503.20215 (2025-03-26)
[7] VBench: Comprehensive Benchmark Suite for Video Generative Models. arxiv. https://arxiv.org/abs/2311.17982 (2023-11-29)
[8] A Survey of AI-Generated Video Evaluation. arxiv. https://arxiv.org/abs/2410.19884 (2024-10 (day not verified))
[9] Bridging Text and Video Generation: A Survey. arxiv. https://arxiv.org/abs/2510.04999 (2025-10-06)
[10] Bridging the Data Provenance Gap Across Text, Speech and Video. web. https://arxiv.org/abs/2412.17847 (2025-02-19)
[11] Controllable Video Generation: A Survey. arxiv. https://arxiv.org/abs/2507.16869 (2026-01-19 (landing-page metadata retrieved))
[12] Multi-modal Generative AI: Multi-modal LLMs, Diffusions, and the Unification. web. https://arxiv.org/html/2409.14993v3 (2025-11-25 (version shown by retrieved page))
[13] Goku: Flow Based Video Generative Foundation Models. arxiv. https://arxiv.org/abs/2502.04896 (not stated in retrieved text)
[14] Pyramidal Flow Matching for Efficient Video Generative Modeling. arxiv. https://arxiv.org/abs/2410.05954 (2025-03-15 (retrieved version))
[15] LTX-Video: Realtime Video Latent Diffusion. arxiv. https://arxiv.org/abs/2501.00103 (2024-12-30)
[16] LongLive: Real-time Interactive Long Video Generation. hf-daily. https://huggingface.co/papers/2509.22622 (2025-09-26)
[17] Video generation models as world simulators. web. https://openai.com/index/video-generation-models-as-world-simulators/ (2024-02-15)
[18] Runway Gen-4: AI Video Generation with World Consistency. web. https://runwayml.com/research/introducing-runway-gen-4 (2025-03-31)
[19] Chameleon: Mixed-Modal Early-Fusion Foundation Models. arxiv. https://arxiv.org/html/2405.09818 (2024-05-16)
[20] OmniGen2: Towards Instruction-Aligned Multimodal Generation. arxiv. https://arxiv.org/abs/2506.18871 (2025-06-23)
[21] VideoUFO: A Million-Scale User-Focused Dataset for Text-to-Video Generation. web. https://proceedings.neurips.cc/paper_files/paper/2025/file/1e6057620ed314b0020b3a30284b0f83-Paper-Datasets_and_Benchmarks_Track.pdf (2025 (exact day not retrieved))
