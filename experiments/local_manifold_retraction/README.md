# Local manifold retraction experiment

2026-10-09. Frozen SD3 Medium, no inversion, masks, training or finetuning. Geometry uses cached empty-text predictions without CFG amplification. Main generation retains original FlowEdit CFG 3.5/13.5 and its seed0 CPU noise sequence.

## Reproduce

    .venv/bin/python experiments/local_manifold_retraction/run_toy.py
    .venv/bin/python experiments/local_manifold_retraction/run_real.py --stage local
    .venv/bin/python experiments/local_manifold_retraction/run_real.py --stage full
    .venv/bin/python experiments/local_manifold_retraction/run_scale_refinement.py
    .venv/bin/python experiments/local_manifold_retraction/run_refined_real.py --stage full
    .venv/bin/python experiments/local_manifold_retraction/evaluate_endpoints.py
    .venv/bin/python experiments/local_manifold_retraction/evaluate_endpoints.py --refined
    .venv/bin/python experiments/local_manifold_retraction/summarize.py

Protocol and code hashes are in outputs/protocol_*.json. Independent auxiliary scale refinement runs on GPU1; generation on GPU0. Refined editing writes to outputs/refined, with separate extension registration and wrapper hash. Both original and refined protocol/model-code hashes are retained.

Original parameters: effective noise sigmas 0.025/0.05; refined exploratory pilot 0.00625/0.0125. One antithetic pair, common noise across scales and variants. SINGLE is one-scale mean retraction; DUAL uses quadratic two-scale extrapolation. Both solve the same Euclidean projection subproblem with relative correction radius 0.05 times the current baseline increment norm at every active step. Actual increment is not globally shrunk.

Original cohort: four development pairs, eight corrected trajectories. Refined pilot: bear and crown-cat, four corrected trajectories. Each has 33 real integration steps. Bear original FlowEdit is independently rerun in both cohorts and compared to retained baseline. Inputs come from experiments/stage1_full/inputs; baseline state snapshots from experiments/siglip_state_response/baseline_states. Source images and VAE reconstruction are separately shown.

All image-edit panels include original FlowEdit. Full512 PNGs and final latent/step diagnostics are retained. SigLIP is used only for auxiliary endpoint evaluation, with the same PIL processing path for baseline and variants; it is not an optimization reward or true success metric. Whole-image MAE is not non-edit-region fidelity.

The toy check archives initial SLSQP status failures in toy_initial_solver_status.json; its numerically close solutions were independently rechecked with trust-constr before proceeding. No raw negative outcomes are deleted. The two-mode midpoint is a structural counterexample, not an implementation failure.

Scope: manifold retraction proxy and development image diagnostic. No recovered true manifold, posterior-mode avoidance, heldout quality improvement or speedup is claimed.

Design: [staged protocol](../../docs/局部流形双尺度回缩_分阶段实验设计_2026-10-09.md).

Results: [full report](../../docs/局部流形双尺度回缩_解析与图像编辑实验结果_2026-10-09.md). run_real_executed_v1.py preserves the exact executed code hash; run_real.py now corrects metadata serialization and timer placement. Stored trajectory tensors are unchanged, with TorchVersion metadata canonicalized to strings for weights-only loading. Stage timers from v1 are labeled as starting after model load.
