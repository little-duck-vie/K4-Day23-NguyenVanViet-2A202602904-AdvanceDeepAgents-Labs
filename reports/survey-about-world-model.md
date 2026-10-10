# Survey on World Models

## TL;DR
- World-model research is converging on models that predict action-conditioned outcomes rather than merely compressing observations [1][2].
- Embodied applications connect learned dynamics to driving and task planning, making decision usefulness as important as visual fidelity [3][4].
- Recent work explores preference optimization, probabilistic structure, and richer 3D/4D representations, while evaluation and sim-to-real transfer remain open [5][6][7].

## Background
A world model is an internal predictive representation used to reason about how an environment changes. Recent conceptual work asks what it means for a neural network to learn such a model, while tutorial work connects the idea to physical AI [2][4]. A broader survey literature also distinguishes understanding a world from merely predicting its next observable state [8].

## Action-conditioned prediction
The central technical requirement is action faithfulness: changing an action should change the predicted consequence in a useful and calibrated way. DreamTrue explicitly targets action-faithful robot prediction and counterfactual post-training [1]. Work on probabilistic structure integration represents a complementary direction, using structured uncertainty rather than treating every latent transition as equally certain [6]. Together, these directions suggest that useful world models need both controllable dynamics and uncertainty awareness.

## Embodied planning and simulation
World models become operational when their predictions improve planning. DriveDreamer grounds learned simulation in autonomous-driving data [3], whereas work on Dual Preference Optimization studies world modeling as a component of embodied task planning [5]. These approaches differ in application and training objective, but both evaluate the model through downstream decisions rather than reconstruction alone.

## Representation scale and coverage
Scaling video data does not automatically guarantee that a model learns every capability required for interaction; the study of 30,000 hours of egocentric video highlights this limitation directly [9]. In parallel, surveys of 3D and 4D world modeling broaden the representation problem beyond flat video prediction toward persistent geometry and time-varying environments [7].

## Trends and open problems
The field is moving toward action-aware training, structured uncertainty, embodied evaluation, and spatially richer representations [1][5][6][7]. Open problems include causal generalization under novel actions, long-horizon error accumulation, reliable uncertainty estimates, comparable benchmarks, and transfer from simulation to physical systems. The continuing debate over understanding versus prediction indicates that benchmark success alone does not settle whether a model has captured reusable world structure [2][8].

## References
[1] DreamTrue: Action-Faithful Robot World Model with Counterfactual Post-Training. arxiv. https://arxiv.org/abs/2610.12468 (2026-10-08)
[2] What Does it Mean for a Neural Network to Learn a World Model?. arxiv. https://arxiv.org/abs/2507.21513 (2025-07-29)
[3] DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving. hf-search. https://huggingface.co/papers/2309.09777 (2023-09-18)
[4] A Tutorial on World Models and Physical AI. arxiv. https://arxiv.org/abs/2606.12783 (2026-06-11)
[5] World Modeling Makes a Better Planner: Dual Preference Optimization for Embodied Task Planning. hf-search. https://huggingface.co/papers/2503.10480 (2025-03-13)
[6] World Modeling with Probabilistic Structure Integration. hf-search. https://huggingface.co/papers/2509.09737 (2025-09-10)
[7] 3D and 4D World Modeling: A Survey. hf-search. https://huggingface.co/papers/2509.07996 (2025-09-04)
[8] Understanding World or Predicting Future? A Comprehensive Survey of World Models. web. https://dl.acm.org/doi/10.1145/3746449 (2025-09-09)
[9] What 30,000 Hours of Ego-centric Video Does Not Teach. arxiv. https://arxiv.org/abs/2610.12464 (2026-10-08)
