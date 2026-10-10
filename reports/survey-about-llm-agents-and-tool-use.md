# Survey on LLM Agents and Tool Use

## TL;DR
- Agent research is shifting from static answer generation toward monitored trajectories in which systems plan, call tools, observe results, and revise actions [1][2].
- Learning-based approaches aim to acquire tool-use policies from interaction, but reward hacking and unreliable execution remain central risks [3][4].
- Evaluation is expanding beyond conversational benchmarks to scientific workflows, optimization, and embodied multi-agent settings [5][6][7].

## Background
An LLM agent couples a language model with state, planning, tools, and an execution loop. Reviews of autonomous agents and tool users emphasize that capability depends on the full interaction architecture, not solely on the base model [2]. The resulting trajectories also create new requirements for observability and intervention.

## Monitoring and control
OnTrack studies real-time monitoring and intervention over agent trajectories [1]. This direction is important because a correct final answer can hide unsafe or inefficient intermediate behavior. Monitoring methods should therefore detect structural deviations early enough to intervene rather than auditing only completed runs.

## Learning to use tools
Tool-R0 explores self-evolving tool learning without an initial labeled dataset [4]. Such approaches reduce dependence on hand-authored demonstrations, but they make reward quality critical. The Reward Hacking Benchmark directly targets cases in which tool-using agents exploit evaluation or environment weaknesses instead of completing the intended task [3]. Together, these works show that autonomous learning and robust evaluation must develop jointly.

## Broader agent capabilities
DataSense-Bench moves toward evaluating AI-scientist workflows [5], while agentic black-box optimization benchmarks test iterative experimental decision-making [6]. Multi-agent egocentric world modeling adds embodied interaction and coordination to the evaluation landscape [7]. These settings expose failures that are difficult to observe in single-turn question answering.

## Trends and open problems
Recent work scales reinforcement learning toward agent self-improvement [8], but greater autonomy also increases the cost of mistakes. Open problems include secure tool permissions, prompt-injection resistance, reproducible environment state, calibrated stopping, trajectory-level attribution, and evaluation that cannot be gamed [1][8][3]. A mature agent stack will need strong monitoring and constrained execution alongside better planning and learning.

## References
[1] OnTrack: Real-Time Monitoring and Intervention in LLM Agent Trajectories via Streaming Structure-Aware Optimal Transport. arxiv. https://arxiv.org/abs/2610.12375 (2026-10-08)
[2] From Language to Action: A Review of Large Language Models as Autonomous Agents and Tool Users. hf-search. https://huggingface.co/papers/2508.17281 (2025-10-28)
[3] Reward Hacking Benchmark: Measuring Exploits in LLM Agents with Tool Use. hf-search. https://huggingface.co/papers/2605.02964 (2026-05-03)
[4] Tool-R0: Self-Evolving LLM Agents for Tool-Learning from Zero Data. hf-search. https://huggingface.co/papers/2602.21320 (2026-02-24)
[5] DataSense-Bench: The First Step Toward an AI Scientist. arxiv. https://arxiv.org/abs/2610.12190 (2026-10-08)
[6] A Closer Look at Agentic BBO: Benchmarking LLM Agents for Black-Box Optimization. arxiv. https://arxiv.org/abs/2610.12183 (2026-10-08)
[7] Multi-Agent Egocentric World Model with Fine-Grained Embodied Interaction. hf-daily. https://huggingface.co/papers/2610.12299 (2026-10-08)
[8] MiMo-V2.6: Scaling Reinforcement Learning Towards Self-Improvement. hf-daily. https://huggingface.co/papers/2610.11959 (2026-10-08)
