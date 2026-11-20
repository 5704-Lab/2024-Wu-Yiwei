# Ten direct semantic-gradient moves

Executed 2026-10-11. Four cached real source latents. Frozen SD3 VAE and SigLIP2 target-text logit; normalized state gradient recomputed after each move, L2 step0.25, ten moves. No generative velocity, noising, training, masks or inversion. This is not a fixed-initial-direction linear traversal.

    .venv/bin/python experiments/clean_semantic_gradient_walk/run.py

--step-norm selects the constant latent L2 step; --device selects GPU. The comparison includes source RGB, source VAE and cached original FlowEdit at CFG3.5/13.5 and CFG1/1. Each pair has10 updated512 PNGs and11 latent states, with per-step score and gradient diagnostics. All40 score increments were positive; no clear task transformation was observed at this small movement budget.

See [report](../../docs/干净x0沿语义梯度移动10次_简单实验_2026-10-11.md). Optimized matching scores are not independent editing-success evidence.
