# Joint text/state permission experiments

2026-10-23. Goal: test actual binding evidence before allowing text/state-dependent
constraints, then investigate source-anchored local geometry and a larger latent
search space. Existing weights only, no extra LLM, inversion, reference input,
semantic editing mask or model training. Preserve failed attempts.

Stage 1: natural-image mirror/vertical-flip pairs with visually verified relation
labels, several equivalent text contrasts, and an antisymmetric readout. This is
a limited relation test, not a general natural-language compiler or VLM benchmark.

Stage 2: contrast-invariant local gradient tensors, multiscale observation and
tail-sensitive aggregation. Known photometric/translation interventions must be
measured before inference. No attention-derived editing regions.

Stage 3: real latent local and trajectory corrections, with actual task-factor
and finite geometry checks, compare restricted search and matrix-free CG where
appropriate. Full task quality, visual source pose and compute are reported;
proxy improvements alone cannot establish success.

Adaptive branches (all prior outputs retained):

- Raw SigLIP relation tests failed; compare actual SD3 shared-noise conditional
  flow-matching residuals, with three times and two fixed random probes.
- Unnormalized tensor geometry confused spatial illumination with motion;
  preserve `geometry_attempt1.py` and its calibration, then normalize local light.
- Run actual latent CG corrections with two SigLIP task factors or finite RF factors.
- If soft geometry still fails, test a finite source-coordinate appearance family:
  `y(theta)=x+E(T_theta(D(x)))-E(D(x))`, with bounded smooth RGB gains.
- Plain nearest-proposal appearance projection under-edits on the first bear;
  add an explicit in-family fur/feather attribute objective and compare RF gating.

The last branches assume a known pure-colour task. They do not claim general
free-language compilation. Optical geometry and language-model proxies remain
fallible, and hard membership of an encoded family does not guarantee exact
decoded-pixel geometry.

## Results

Full report: [自由文本联合许可：实验结果与自适应调整](../../docs/自由文本联合许可_实验结果与自适应调整_2026-10-23.md).
Uniform endpoints: [JSON](outputs/analysis/endpoints.json), [CSV](outputs/analysis/endpoints.csv).
Numerical checks, complete-run counts and interrupted pilots: [audit.json](outputs/analysis/audit.json).

| Branch | Complete trajectories | Finding |
|---|---:|---|
| `control/full`, SIG and RF | 4 | Geometry improves, bear/owl pose still changes |
| `manifold`, PROJECT and PROJECT_RF | 4 | Source pose retained, substantial under-editing |
| `manifold_semantic/SEMANTIC`, bear | 1 | Fixed semantic penalty still under-edits |
| `constrained_fast` | 2 | Attribute progresses but blue bear / green owl |
| `routing`, P_ONLY and P_SOURCE_V | 8 | Unwanted identity/view changes remain |
| `achromatic`, exposure/desaturation | 2 | Avoids blue/green exploit, under-edits |
| `achromatic_strong` | 2 | Background changes; owl head overexposes |
| `achromatic_neutral`, owl | 1 | Pose retained, grey haze and incomplete white feathers |

Total: **24 full trajectories / 792 active updates**, plus 8 local candidates,
15 relation contrasts for each of two scorers, and 56 known calibration controls
evaluated before/after normalization. Two interrupted pilots have 7 and 18
updates respectively and are excluded from the full-run total.

No method in this development batch establishes reliable general free-text
permission. Known black/white attributes are declared manually. Strong progress
uses margin 3.5 / increment .16 after .5 / .08 under-edits; these are exploratory
parameters, not universal completion thresholds. Neutral mixing tests a convex
mixture towards neutral white rather than exposure. `body_probe.py` shows why
wording diversity in the same scorer is not independent semantic evidence.

## Reproduction

Run from the repository root with existing local weights. SD3 Medium FP32,
512 pixels, seed 0, CFG 3.5 / 13.5, shared source/target noises, scheduler indices
17–49 (33 active updates). Encodings and scheduler reuse `stage1_full/inputs`;
the original baseline is the previously verified `flowedit_frequency_trajectory`.

```bash
.venv/bin/python experiments/joint_permission/binding.py
.venv/bin/python experiments/joint_permission/rf_binding.py
.venv/bin/python experiments/joint_permission/geometry_calibrate.py
.venv/bin/python experiments/joint_permission/control.py --stage local
.venv/bin/python experiments/joint_permission/control.py --stage full
.venv/bin/python experiments/joint_permission/manifold.py
.venv/bin/python experiments/joint_permission/constrained_fast.py
.venv/bin/python experiments/joint_permission/routing.py
.venv/bin/python experiments/joint_permission/achromatic.py
.venv/bin/python experiments/joint_permission/achromatic.py --tag achromatic_strong --goal-margin 3.5 --beta .16
.venv/bin/python experiments/joint_permission/achromatic.py --pairs brown_owl__00 --tag achromatic_neutral --goal-margin 3.5 --beta .16 --family neutralmix
.venv/bin/python experiments/joint_permission/analyze.py
.venv/bin/python experiments/joint_permission/body_probe.py
```

Complete trajectories are skipped on rerun. Keep archived interrupted
`manifold_semantic/SEMANTIC_RF` and slow `constrained` pilots unchanged; stop
reasons are in `semantic_stop.json` and `constrained_stop.json`. The unnormalized
geometry version and original calibration are preserved separately.

`analyze.py` checks finite values, source states, 34 states / 33 step indices,
endpoint equality, routing source errors, final achromatic-family membership
and actual finite task acceptance. Its top-64-row RGB metric is a post hoc
background diagnostic on bear/owl, never used as an inference mask or loss.
Wall times exclude model loading and include saving/probing; shared GPU load
means they are not a controlled performance benchmark. No held-out quality or
multiseed claim is made.

The first extended audit failed because it used different TF32 / attention
settings from the experiments (about .001 maximum latent reconstruction error).
`analyze_backend_mismatch.log` preserves that failure. With the same math attention
backend and TF32 disabled, all five achromatic-family final latents reconstruct
with **zero maximum error** from archived source and parameters, and the complete
audit passes. This confirms implementation membership, not semantic correctness.
