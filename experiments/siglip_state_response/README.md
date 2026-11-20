# SigLIP clean-state response precheck

Executed 2026-10-09. Frozen SD3 Medium VAE and local SigLIP2 Base patch16 256; native tensor bilinear antialias preprocessing. No training, inversion, or spatial mask. Four development pairs, two original FlowEdit states per pair; not heldout evaluation.

Run with workspace .venv/bin/python:

- run_analytic.py: independent toy derivatives, constrained solve and integration identities.
- prepare_baseline_states.py: same-noise original FlowEdit continuation, exact final-state reproduction.
- run_precheck.py --pair bear__00 --step 33 --hvp --state-file experiments/siglip_state_response/baseline_states/bear__00.pt: actual gradient/HVP diagnostics. Repeat for four pairs and steps 33,47.
- Add --dtype float64 --split for the bear33 precision audit; VAE on GPU0, SigLIP on GPU1.
- run_direction_audit.py: independent FP64 finite differences of gradient-direction alignment and clipping active sets.
- run_matched_controls.py: same-radius G versus D on eight states.
- render_gate_comparisons.py: local previews including original FlowEdit final; these are not full corrected trajectories.

Outputs and logs retain all failure trials. run_precheck_v1.py and v2.py archive earlier diagnostic versions; current run_precheck.py includes direct log-sigmoid gradient-factor check. Per-point JSON records source code SHA256, model constants, text truncation, precision and preprocessing. Input states originate in experiments/stage1_full; baseline_states contains reused and resumed snapshots. Models/data/environments remain unchanged.

Interpretation: local differential identities supported; seven states show preliminary common ascent, one rejected; nuisance sensitivity unresolved. Numerical margins and independent quality evaluation remain required. No speed or full-edit success claim.

See [report](../../docs/SigLIP干净状态响应_首轮数值与语义预检结果_2026-10-09.md).

## Adaptive total-increment revision

The new mandatory every-step proximal experiment has completed: see [protocol and artifacts](adaptive_prox/README.md) and [results](../../docs/SigLIP自适应近端约束_局部与全轨迹实验结果_2026-10-09.md). Local proxy gains did not translate to task completion at the registered full lambda1. All 396 corrected steps executed with nonzero updates.
