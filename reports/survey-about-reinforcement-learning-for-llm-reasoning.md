# Survey on Reinforcement Learning for LLM Reasoning

## TL;DR
- Current work treats reasoning as a sequential decision problem: models must decide not only what to infer, but when additional reasoning is worth its cost [1][2].
- Algorithmic research spans federated group-relative optimization, multi-agent co-training, memory-aware learning, and selective rollout allocation [3][4][5][6].
- The main open problem is allocating reward and compute to examples that produce transferable reasoning improvements rather than narrow benchmark gains [7][8].

## Background
Reinforcement learning for LLM reasoning optimizes model behavior using outcome or process-derived signals. Unlike supervised imitation, the learner must explore candidate reasoning trajectories and assign credit to useful decisions. A sequential-decision characterization is therefore valuable for comparing methods with different assumptions about autonomy, supervision, and computational cost [2].

## When and how much to reason
Reasoning before every action can waste inference compute. Adaptive cross-turn estimation studies when an agent should generate new reasoning and when earlier reasoning remains sufficient [1]. Selective-rollout work applies the same resource-allocation principle during RL, concentrating training effort where an additional rollout is expected to pay off [6]. These approaches connect inference efficiency with training efficiency.

## Optimization and distributed learning
Group-relative policy optimization remains an active design space. Fed-GRPO studies how reward signals can support federated optimization [3], while SynCo uses multi-agent reinforcement learning and synthetic-data co-training for self-evolving LLMs [4]. MiMo-V2.6 represents the complementary scaling direction, expanding RL compute toward model self-improvement [9].

## Data selection, memory, and transfer
Memory-constrained RL makes resource limits part of the reasoning objective rather than an afterthought [5]. Competence-difficulty alignment questions uniform sampling and argues that training examples should match the learner's current capabilities [7]. Cross-domain evaluation is equally important because improvements confined to one task family may not constitute general reasoning progress [8].

## Trends and open problems
The recent literature increasingly combines adaptive computation, selective data or rollout allocation, and distributed learning [1][3][7][6]. Unresolved issues include reward misspecification, unstable credit assignment, contamination of evaluation sets, memory and inference costs, and whether gains transfer across domains [5][8]. Stronger evaluations should report both reasoning quality and the compute required to obtain it.

## References
[1] When Should Agents Think? Adaptive Reasoning via Cross-Turn Estimation. arxiv. https://arxiv.org/abs/2610.12061 (2026-10-08)
[2] A 3D Characterization Framework for Intelligent Sequential Decision Making. arxiv. https://arxiv.org/abs/2610.11696 (2026-10-08)
[3] Fed-GRPO: Reward-Signal-Driven Federated Group Relative Policy Optimization. arxiv. https://arxiv.org/abs/2610.11502 (2026-10-08)
[4] SynCo: Data Synthesis Co-Training for Self-Evolving LLMs via Multi-Agent Reinforcement Learning. arxiv. https://arxiv.org/abs/2610.11345 (2026-10-08)
[5] Reinforcement Learning for LLM Reasoning Under Memory Constraints. hf-search. https://huggingface.co/papers/2504.20834 (2025-04-29)
[6] Act Only When It Pays: Efficient Reinforcement Learning for LLM Reasoning via Selective Rollouts. hf-search. https://huggingface.co/papers/2506.02177 (2025-06-02)
[7] Rethinking the Sampling Criteria in Reinforcement Learning for LLM Reasoning: A Competence-Difficulty Alignment Perspective. hf-search. https://huggingface.co/papers/2505.17652 (2025-05-23)
[8] Revisiting Reinforcement Learning for LLM Reasoning from A Cross-Domain Perspective. hf-search. https://huggingface.co/papers/2506.14965 (2025-06-17)
[9] MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement. hf-daily. https://huggingface.co/papers/2610.11959 (2026-10-08)
