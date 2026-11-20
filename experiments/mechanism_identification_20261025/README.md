# Mechanism coverage and identifiability, 2026-10-25

This experiment follows the [research order](../../docs/源图文本状态速度联合约束与共同编辑机制_经典及前沿调研与物理建模_2026-10-25.md).

Protocol registered before results:

1. Generate 16 deterministic two-object toy scenes with explicit material,
   heading and location factors. Scenes 0--3 are development, 4--7 calibration,
   and 8--15 held-out within this renderer. Each target recolours the left bird
   only; source pose, right object and backdrop remain fixed by construction.
   Identical-material pairs deliberately test colour-only binding ambiguity.
2. Evaluate the existing frozen SigLIP2, VAE, source patch observer, geometry and
   image-filter phase. Compare known legal endpoints with wrong-object,
   both-object, changed-heading, shifted and duplicated-object counterexamples.
   Original source photometries are geometry controls, NOT legal local edits.
3. Fit source-global affine, source-coordinate photometric, and source-feature
   graph photometric families to each image offline. All use shared image
   coefficients, with finite bounded convex least-squares fits. Ground-truth
   target access here measures family coverage only, not an automatic editor.
4. Compare source fixed-position patch observations against a global shared
   photometric nuisance and dense source--candidate correspondence diagnostics.
   No correspondence weights are used to multiply editing updates.
5. Cross model-proxy versus oracle readouts and restricted mechanism candidates
   versus a bank additionally containing the known legal endpoint. The latter
   is explicitly an oracle upper bound, not a deployment candidate or a claim
   that original FlowEdit directions have been tested on the toy scenes.
6. Inspect natural-image source controls and retained SD3 failures separately.
   If coverage/observation gates fail, diagnose and adjust these first; do not
   use more full trajectories to substitute for these missing interfaces.

Predeclared calibration: source-relative text gain floor = half the median
positive-endpoint gain on calibration scenes, provided that median is positive;
otherwise the proxy fails calibration and its selection is reported as such.
Source budgets come from the original global-photometry controls. Also report
a local-witness-calibrated budget, using only scenes 4--7, to test the effect of
covering actual legal edits. All selection results retain failed constraints.
No target-completion score or constraint metric alone establishes joint success.

Models remain frozen. No new LLM, segmentation model, external image reference,
training or editing mask is used by the model-proxy branch. Renderer factors and
reference pixels are confined to labelled controls, fitting diagnostics and the
oracle branch. Synthetic results are not natural-image editing validation.

GPU 0 only; other GPU processes are left untouched. FP32, TF32 off. Tags are
immutable; store code snapshots, hashes, source records and all failures.

## Completed results and adaptive decisions

[Chinese research report](../../docs/共同编辑机制_覆盖可辨识性与潜空间可执行性_实验探索_2026-10-25.md).
48 unique renderer sources / 384 unique variant images, with repeated historical
copies excluded from those counts. Four natural-image sources and retained SD3
failures are controls, not newly successful edits. No new complete SD3 trajectory.

- V1: original global-photo source budgets accept only 2/8 legal test edits.
  Correcting calibration accepts 8/8, but wrong-object edits pass too.
- [V2 adaptation](ADAPTATION_V2.md): common positive material gains, matched-count
  graph comparison, assignment and flip readouts. Approximate material coverage
  12/16; semantic candidate selection remains unreliable.
- [V3 numerical audit](ADAPTATION_V3.md): native-pixel bounded convex fitting
  fixes the downsampling/product mismatch and incomplete solve. Same-family
  coverage reaches 16/16 inspected and 8/8 new scenes. Four real VAE checks find
  finite near-legal updates, with strict zero identity lifts. The full chord has
  much larger linearization error than small finite-difference derivative error.
- [V4](ADAPTATION_V4.md): explain candidate appearance by a common mechanism,
  then measure remaining signals. Four-caption right-preservation tests do not
  solve binding. Oracle-fitted candidate inclusion can give a misleading 8/8;
  the direct-bank audit is 7/8 with a wrong-object selection.
- [V5](ADAPTATION_V5.md): three shared additive coefficients close a natural
  global-white coverage gap. New 8-scene joint-complex observations accept all
  legal edits and reject all turn/shift/duplicate controls, but admit wrong-object
  and both-object edits. RGB alone rejects two legal edits; phase alone misses
  four turns. Existing calibrated patch controls also detect the geometry cases.
  This is not proof of phase-specific editing benefit. Direct-bank mechanism
  selection remains 3/8 versus 5/8 for patch and 8/8 oracle.

The first V5 launch failed before model/protocol initialization due to a module
name collision with an older `joint_permission/binding.py`. Import order was
fixed; the failed log and successful retry log are retained. No result was
overwritten or selected based on that import failure.

## Artifacts

- [V1 analysis](outputs/v1/analysis.json), [V2 analysis](outputs/v2/analysis.json).
- [V3 exact coverage / VAE audit](outputs/v3/refine_results.json).
- [V4 observations](outputs/v4/analysis.json),
  [direct bank, adaptive audit](outputs/v4/direct_bank_analysis.json).
- [V5 observations and selections](outputs/v5/analysis.json),
  [natural-image controls](outputs/v5/natural.json).
- [Provenance / numerical audit](outputs/analysis/audit.json),
  [PNG figure](outputs/analysis/observation_comparison.png),
  [SVG figure](outputs/analysis/observation_comparison.svg).
- [Actual VAE decoded comparison](outputs/v3/scenes/scene35/vae_comparison.png),
  [natural bear explanation](outputs/v5/natural/bear__00.png).

## Reproduction

Historical outputs are immutable. Each output folder retains driver snapshots;
datasets retain their matching generation snapshot and exact image hashes. For
an exact historical replay, restore the matching driver/dependency snapshots to
a separate sibling experiment directory under `experiments/`, and run there.
Do not place the copy outside this repository: helper imports rely on ROOT.
The current default material fitter reproduces the old family; the global
offset is enabled explicitly only for V5.

Commands used (GPU 0 only, run sequentially; existing tags intentionally refuse
overwrite):

```bash
.venv/bin/python experiments/mechanism_identification_20261025/dataset.py --tag v1
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=4 .venv/bin/python -u experiments/mechanism_identification_20261025/probe.py --tag v1
.venv/bin/python experiments/mechanism_identification_20261025/analyze.py --tag v1
.venv/bin/python experiments/mechanism_identification_20261025/dataset.py --tag v2 --count 32 --fresh-heldout-start 16
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=4 .venv/bin/python -u experiments/mechanism_identification_20261025/probe.py --tag v2 --extended
.venv/bin/python experiments/mechanism_identification_20261025/analyze.py --tag v2
.venv/bin/python experiments/mechanism_identification_20261025/dataset.py --tag v3 --count 40 --fresh-heldout-start 32
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=4 .venv/bin/python -u experiments/mechanism_identification_20261025/refine.py --tag v3
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=4 .venv/bin/python -u experiments/mechanism_identification_20261025/binding.py
.venv/bin/python experiments/mechanism_identification_20261025/dataset.py --tag v5 --count 48 --fresh-heldout-start 40
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=4 .venv/bin/python -u experiments/mechanism_identification_20261025/offset.py
OPENBLAS_NUM_THREADS=4 .venv/bin/python experiments/mechanism_identification_20261025/audit.py
```

V1 uses its retained original driver; the current probe additionally records
assignment diagnostics. Small batching roundoff between versions is not an
editing effect. Adaptive conjunction and direct-bank JSONs are diagnostic
analyses of existing rows; they are explicitly not new independent test runs.
