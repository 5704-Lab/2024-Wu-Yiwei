# Conditional/null joint-control experiment, 2026-10-25

Theory: [paired conditional/null response and minimum necessary changes](../../docs/条件与空文本反事实_源锚定最小编辑的联合约束_2026-10-25.md).

Report: [actual results and adaptive changes](../../docs/条件空文本联合约束_实验结果与自适应调整_2026-10-25.md).
Completed **18 full trajectories / 594 active updates**, including baselines and
negative controls. None establishes joint semantic editing success. Six actual
source-tube trajectories have independent finite-state verification; proxy
constraint success is distinct from target completion and source-heading quality.

Initial protocol, recorded before observing this experiment's results:

- Frozen local SD3 Medium, original VAE and evaluation-only SigLIP2. FP32,
  TF32 disabled; 512 pixels; original CFG 3.5/13.5; 33 active updates.
- Paired fixed-state probes on bear, owl, van and cat, at steps 17/25/33/47.
  Preserve the original four-query batch, add two cross queries separately.
  Check baseline compatibility, six-query decomposition and source calibration.
- Full seed-0 development trajectories on bear and owl: BASE, existing CF
  correction, CF without empty-field feedback (NO_NULL), and JOINT.
- JOINT uses matrix-free Gauss--Newton source geometry restoration and a
  halfspace retaining 50% of the positive conditional-velocity response.
  Six CG iterations, correction radius 1.5 times proposal norm, finite decoded
  geometry/merit acceptance. It uses no SigLIP feedback or image re-encoding.
- This is a local approximation, not an implementation or proof of a full
  reachability value function. The source geometry descriptor and text velocity
  remain proxies; the conditional halfspace is not validated semantic necessity.
- Inspect actual target colour, source heading/identity, background and texture.
  Report global text scores, source phase/geometry and evaluation-only top-64-row
  background errors separately. No single score is an editing-success criterion.
- If the first version fails, keep all outputs, register changes under fresh
  tags, compare the changed mechanism at the same states, then rerun full edits.
  Additional seeds or sources are subsequent checks, not untouched test data.

Run GPU jobs sequentially; other users/processes share both GPUs.

Numerical adjustment after the first fixed-state probes: `v1_probe` normalized
near-zero first-step interaction/state roundoff into unit directions. Its
first-step direction-rank results are invalid. Decomposition and baseline checks and all
finite image interventions remain retained. Before rerunning rank diagnostics,
`v2_probe` excludes directions below 0.1% of the original proposal norm. This
threshold is a numerical diagnostic rule, not a semantic permission threshold.

After v1's geometry proxy decreased despite obvious heading errors, a new source
observer is calibrated in `patches.py`. It uses the existing frozen SigLIP patch
features, never a global text score. Bounded nuisance subspaces and tolerances
come only from known source photometries. Evaluate ranks 0/2/4/6/8; before
examining calibration results, select the smallest rank >=2 for which both fixed
mirror/shift controls exceed the maximum held-out legal photometry excess by a
factor of five on both sources. Then check the actual wrong-heading outputs;
if they are not separated from legal controls, do not treat it as a useful
additional source observer. This is development calibration, not a held-out
semantic benchmark or an automatic object/attribute binding solution.

The photometry positives mean "geometry-preserving", not "semantically permitted
everywhere by the edit instruction": global recolouring can change background.
Also, global SigLIP evaluation and the patch observer share the same encoder,
so their agreement is not independent semantic validation. Actual images and
source comparisons remain necessary.

The rank rule selected **rank 2**, with known-geometry/held-out-photometry excess
ratios 389 (bear) and 548 (owl). Actual wrong-heading v1 outputs are also well
outside this photometric tolerance. `v2_patch` therefore runs PATCH against
PATCH_RAW on both seed-0 cases. PATCH_RAW penalizes raw source patch deviations;
PATCH removes the calibrated bounded colour nuisance and adds a fixed-source
photometric tolerance. Both retain 100% of the positive original CF tangent
response, rather than v1's 50% (which was inactive in bear's 33 updates).
These are two simultaneous changes relative to v1; the PATCH/PATCH_RAW contrast
isolates the source observer, and should not be attributed to retention alone.

After bear's completed PATCH still turned and its cumulative source loss reached
0.0429, `v3_tube` tests an actual finite cumulative source budget. The energy
budget is 0.5 * 5 * maximum legal rank-2 photometry excess, fixed from calibration.
If the corrected proposal is outside, eight finite bisections shrink the whole
step from the previously accepted state. Task floor misses are explicitly logged;
this source-first fallback is not claimed to satisfy both constraints. No tube
radius is chosen by maximizing the eventual image score.

At bear's v3 step 31 the guard reached its budget and subsequent steps held,
while the protected conditional response remained positive. `v4_tube_relaxed`
therefore changes only retention from 1 to 0: it requires nonnegative conditional
tangent response, rather than preserving all of the raw CF response. This tests
whether protecting the mixed text field is itself preventing source restoration.
The source observer, budget, solver and trust radius stay fixed. Completion is
still assessed separately; relaxing a proxy floor is not a success by definition.

The same-state solver probe at bear step 31 found that CG 6 -> 24 barely changed
the source loss (.000995 -> .000988), but 10x source strength with CG 24 entered
the budget (.000593 < .000733) while retaining positive task response. Thus the
earlier hold is not evidence of local infeasibility. `v5_adaptive_bear` and
`v5_adaptive_owl` test the identical adaptive policy on separate GPUs: use the
original correction first, retry 10x source strength / CG 24 only if outside,
select the candidate with lower actual source loss, and keep the same finite
tube fallback. Retention is 1; all attempts and floor misses are logged. This
is a finite two-candidate search, not an exact nonlinear QCQP solver.

The initial heavy jobs ran sequentially. After observing approximately 20 GiB
peak process memory on GPU 0 and stable 19 GiB occupied on GPU 1, the tube job
uses GPU 1 with a 50% PyTorch per-process memory cap. This permits one job per
device while preserving headroom; no unrelated process is stopped.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/null_joint_20261025/run.py --stage probe --pairs bear__00 brown_owl__00 bus__00 cat_crown__00 --tag reproduce_probe
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/null_joint_20261025/run.py --stage full --tag reproduce_v1
```

## Artifacts and reproduction

Existing tags are immutable: use fresh tags with current code. Each historical
driver is stored under its output tag as `snapshot_*_seed0.py`, with a matching
hash in the protocol. To replay one exactly, copy it beside `run.py` under a new
filename (so relative imports still resolve) and choose a fresh output tag.
`patches.py` is the shared source observer; its protocol hashes are retained.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/null_joint_20261025/patches.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/null_joint_20261025/run.py --stage full --modes PATCH_RAW PATCH --retention 1 --tag reproduce_v2
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/null_joint_20261025/run.py --stage full --modes PATCH_TUBE --retention 1 --tag reproduce_v3
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/null_joint_20261025/run.py --stage full --modes PATCH_TUBE --retention 0 --tag reproduce_v4
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/null_joint_20261025/run.py --stage full --modes PATCH_ADAPT --retention 1 --tag reproduce_v5
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/null_joint_20261025/check_observer.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/null_joint_20261025/solver_probe.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/null_joint_20261025/response_audit.py --tag v2_patch --modes PATCH_RAW PATCH --compare-raw
.venv/bin/python experiments/null_joint_20261025/analyze.py --tags v1_full v2_patch v3_tube v4_tube_relaxed v5_adaptive_bear v5_adaptive_owl
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/null_joint_20261025/verify_tubes.py --tags v3_tube v4_tube_relaxed v5_adaptive_bear v5_adaptive_owl
```

- [Uniform endpoints](outputs/analysis/endpoints.json) and [CSV](outputs/analysis/endpoints.csv).
- [Trajectory audit](outputs/analysis/audit.json),
  [source tube recomputation](outputs/analysis/tube_recomputation.json),
  [historical code and input provenance](outputs/analysis/reproducibility.json).
- [Bear summary](outputs/analysis/bear__00_summary.png),
  [owl summary](outputs/analysis/brown_owl__00_summary.png).
- [Patch observer calibration](outputs/patch_calibration/results.json),
  [derivatives](outputs/observer_derivatives.json),
  [same-state solver probe](outputs/solver_probe/bear__00/results.json),
  [paired finite conditional work](outputs/v2_patch_response_paired_seed0/results.json).

Protocol, snapshots, progress files, logs, intermediate PNGs and full latent
trajectories preserve negative results. Timing excludes some startup/calibration
and GPUs are shared, so no acceleration or exclusive-runtime claim is made.
