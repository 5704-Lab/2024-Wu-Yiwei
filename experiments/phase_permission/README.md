# Phase permission: empirical validation

Registered 2026-10-23 before inspecting new intervention results.

Goal: test whether bounded appearance mechanisms preserve necessary phase changes
while rejecting identifiable geometric changes, and whether this helps real FlowEdit.
No training, inversion, extra LLM, reference image, or edit-region mask. Frozen SD3
Medium and existing SigLIP2 Base. Original source/target prompts are unchanged.

Stages:
1. Actual source images with known appearance and geometry interventions, including
   off-family texture changes. Compare global, smooth and richer appearance families.
2. Real cached FlowEdit states at steps 17, 33 and 47 on four images. Check decoder
   derivatives, bounded mechanism fitting, finite accepted updates and task retention.
3. If numerics are valid, run actual source-anchored corrections on complete trajectories;
   compare original FlowEdit, phase anchoring without appearance permission, and
   mechanism-permitted phase anchoring. Bear/owl are appearance tasks; bus/crown
   are explicitly outside the pure-appearance assumption.

Primary assumptions and falsification: if valid appearance changes cannot be separated
from geometric perturbations, or local optimization only improves its own residual
while blocking editing, this is negative evidence. An improved phase loss or SigLIP
score is not sufficient evidence of image-editing success. Preserve failed attempts.

Settings: 512 RGB, FP32, 50 scheduler steps / 33 active steps, n_min=0, n_avg=1,
source CFG=3.5, target CFG=13.5, seed=0 shared CPU-generated noises. Source reference
is VAE reconstruction. GPU 0 is shared with unrelated jobs; do not terminate them;
timings are not dedicated speed benchmarks.

Observations: zero-DC complex oriented Gaussian frequency filters, frequencies
1/16, 1/8, 1/4 cycles/pixel at 128 pixels, four orientations, three fixed color
projections. Reflect padding; fixed sampling lattice; smooth phase embedding with
source-fixed epsilon and source confidence weights (not semantic editing masks).
Appearance: per-RGB-channel log gain and offset on fixed cosine basis (1x1, 2x2,
3x3), bounded coefficients and cumulative gain/offset. Explicitly no arbitrary RGB
residual or free motion in the appearance family. All code/results remain local.

## Completed results

See the [Chinese report](../../docs/相位许可理论_真实实验结果与适用边界_2026-10-23.md),
[full image grid](outputs/all_comparison.png), and [completion audit](outputs/completion_audit.json).

Completed: 216 known-intervention cases, 24 out-of-family photometric counterexamples,
36 real local candidates, 16 full trajectories / 528 active steps, 12 real conditional
response decompositions, 20 independent reduced-QP checks, and endpoint fit sensitivity.
All original FlowEdit states are reproduced exactly. The permitted-phase residual fell
17–32%, but the main pose/view changes remained. No overall editing-quality benefit or
stable advantage over ordinary soft phase anchoring has been established.

Implementation is a restricted 3D Gauss–Newton search, not a full-dimensional robust
cone solver. Unknown residuals are soft-penalized; identifiable geometry is diagnosed
separately, not used as an exclusive controller. See registration for all assumptions.
SigLIP and phase loss are proxies, not semantic ground truth. First numerical fitting
attempts are preserved under `attempt1/`; final source snapshots are in `outputs/code_snapshot/`.

Run from the repository root. GPU stages should be sequential; CPU stages can run separately.

```bash
.venv/bin/python experiments/phase_permission/validate_solver.py
.venv/bin/python experiments/phase_permission/mechanism.py
.venv/bin/python experiments/phase_permission/stress.py
.venv/bin/python experiments/phase_permission/real.py --stage local
.venv/bin/python experiments/phase_permission/real.py --stage full
.venv/bin/python experiments/phase_permission/response.py
.venv/bin/python experiments/phase_permission/polish.py
.venv/bin/python experiments/phase_permission/analyze.py
.venv/bin/python experiments/phase_permission/audit.py
```

Local/full stages skip completed result files. Delete nothing to rerun: copy the
experiment directory or explicitly configure a new output root. Resuming a checkpoint
must use the same source code/configuration; do not mix modified code with saved states.
