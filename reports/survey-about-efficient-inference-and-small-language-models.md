# Survey on Efficient Inference and Small Language Models

## TL;DR
- Small-language-model design seeks useful capability under explicit memory, latency, and deployment constraints rather than maximizing parameter count [1][2].
- Efficiency must be evaluated together with robustness: a compact or fast model is not useful if its learned directions can be hijacked [3].
- The strongest evaluations should report end-to-end latency, memory, quality, and safety on the target hardware and workload [3][1].

## Background
Efficient inference concerns the compute, memory, and latency required to serve a trained model. Small language models approach the problem through compact architectures and carefully selected training data, while systems techniques additionally use quantization, pruning, caching, batching, and routing. A foundations-oriented treatment places inference alongside pre-training, prompting, alignment, and reasoning, reinforcing that these layers interact [2].

## Small-model case studies
The H2O-Danube3 technical report provides a concrete small-language-model case study [1]. Such reports are useful because model size alone does not determine deployment cost: tokenizer behavior, sequence length, numerical precision, batching strategy, and supported context all affect real throughput and memory use.

## Scaling and measurement
Small-model technical reports such as H2O-Danube3 make it possible to examine architecture and deployment choices together [1]. This highlights the difference between theoretical parameter reductions and realized system savings. Evaluations should therefore separate model-level efficiency from hardware-level measurements and compare at matched quality.

## Safety as an efficiency constraint
Work on removing hijacked directions from backdoored vision-language models demonstrates that deployment optimization cannot ignore model integrity [3]. Defensive processing adds cost, but omitting it can invalidate otherwise attractive efficiency results. This trade-off is particularly important for small models deployed at the edge, where monitoring and recovery resources are limited.

## Trends and open problems
Research is moving toward joint optimization across model architecture, representation, serving systems, and safety [3][1][2]. Open problems include reproducible energy measurement, realistic long-context workloads, quality degradation under aggressive quantization, hardware-specific tuning, secure model compression, and fair comparisons between local small models and routed access to larger models.

## References
[1] H2O-Danube3 Technical Report. hf-search. https://huggingface.co/papers/2407.09276 (2024-07-12)
[2] Foundations of Large Language Models. hf-daily. https://huggingface.co/papers/2501.09223 (2026-10-08)
[3] Purifying Backdoored Large Vision-Language Models by Removing Hijacked Directions. arxiv. https://arxiv.org/abs/2610.09941 (2026-10-07)
