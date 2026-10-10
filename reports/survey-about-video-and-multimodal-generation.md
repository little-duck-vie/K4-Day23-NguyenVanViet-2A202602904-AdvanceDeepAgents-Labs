# Survey on Video and Multimodal Generation

## TL;DR
- Video generation is moving toward unified multimodal systems that combine visual world modeling, interactive conditioning, and agentic controls [1][2][3].
- Evaluation is becoming more specialized, covering data-video understanding, prompt enhancement, and proactive safety rather than visual quality alone [4][5][6].
- Temporal consistency, controllability, safety, and efficient interactive generation remain the main unresolved challenges [7][8][9].

## Background
Video generation extends image synthesis across time, making consistency and causal evolution as important as per-frame quality. Multimodal systems add text, images, audio, actions, or structured data as conditions. WOVEN connects visual world modeling with multimodal LLMs, illustrating the growing overlap between understanding, prediction, and generation [1].

## Unified and interactive generation
LiveTalk studies real-time multimodal interactive video diffusion [2]. LynnReal-Omni targets native multimodal video generation for agentic visual workflows [3]. Both directions move beyond one-shot text-to-video generation toward systems that respond continuously to heterogeneous controls, but this makes latency and control fidelity first-class constraints.

## Evaluation and safety
DataVista evaluates multimodal models on data-video understanding [4], while work on unified evaluation of prompt enhancers examines an upstream component that can substantially affect generated results [6]. ConceptGuard adds proactive risk detection for text-and-image-to-video generation [5]. These efforts show that evaluation must cover comprehension, conditioning, and safety in addition to appearance.

## Self-correction and world consistency
Self-correction optimization addresses interleaved multimodal generation, where errors can propagate across alternating modalities [7]. Multi-agent egocentric world modeling emphasizes fine-grained embodied interaction [8], and OuroWorld targets diverse looping 3D cinemagraphs [9]. Collectively, these works expose the need for persistent spatial and temporal state rather than independent frame synthesis.

## Trends and open problems
The field is converging on interactive controls, richer world representations, task-specific evaluation, and safety mechanisms [5][2][8]. Open problems include long-horizon temporal drift, identity preservation, synchronization across modalities, controllable camera and object motion, efficient serving, provenance, and evaluation under adversarial prompts [5][6][9].

## References
[1] WOVEN: Weaving Visual World Modeling into Multimodal LLMs. arxiv. https://arxiv.org/abs/2610.12417 (2026-10-08)
[2] LiveTalk: Real-Time Multimodal Interactive Video Diffusion via Improved On-Policy Distillation. hf-search. https://huggingface.co/papers/2512.23576 (2025-12-29)
[3] LynnReal-Omni: Native multi-modal Video Generation for Agentic Visual Workflows. hf-search. https://huggingface.co/papers/2609.15863 (2026-09-14)
[4] DataVista: Diagnosing Multimodal LLMs on Data Video Understanding. arxiv. https://arxiv.org/abs/2610.11993 (2026-10-08)
[5] ConceptGuard: Proactive Safety in Text-and-Image-to-Video Generation through Multimodal Risk Detection. hf-search. https://huggingface.co/papers/2511.18780 (2025-11-24)
[6] Towards Unified Evaluation of Prompt Enhancers for Video Generation. arxiv. https://arxiv.org/abs/2610.11736 (2026-10-08)
[7] Self-correction Optimization for Interleaved Multimodal Generation. arxiv. https://arxiv.org/abs/2610.10400 (2026-10-07)
[8] Multi-Agent Egocentric World Model with Fine-Grained Embodied Interaction. hf-daily. https://huggingface.co/papers/2610.12299 (2026-10-08)
[9] OuroWorld: Bringing Any 3D World Alive as Diverse, Endlessly Looping 3D Cinemagraphs. hf-daily. https://huggingface.co/papers/2610.12461 (2026-10-08)
