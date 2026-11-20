# Real relation-constraint experiments, 2026-10-24

Report: [相位联合约束：真实实验与瓶颈定位](../../docs/相位联合约束_真实实验与瓶颈定位_2026-10-24.md).

**46 complete trajectories / 1,518 active updates**, including baselines and
negative controls; these are not 46 independent quality successes. Also retained:
3 actual local corrections, 3 solve-accuracy controls at the same real state,
2 mechanism-only pilots, 40 known image interventions evaluated under four
observation variants (160 rows), and source-reference / semantic diagnostics.

Frozen SD3 Medium + original VAE + SigLIP2, FP32, 512 pixels, CFG 3.5 / 13.5,
33 active steps (17–49), shared CPU-generated noises. Most runs use seed 0;
the caption experiment uses seeds 0 and 1. No extra pretrained weights, LLM,
external reference, segmentation or attention editing mask. Photometric fields,
source-guided filters and estimated geometric fields are explicitly bounded or
scoped appearance hypotheses, not general free-text permission.

## Completed trajectory groups

| Directory | Full trajectories | Result |
|---|---:|---|
| `v2/full` | 4 | Phase, shared amplitude, proxy-fact and velocity-only ablations; bear still turns |
| `v3/full` | 2 | Exact BASE plus hard source tube; tube holds but under-edits |
| `material_full_chroma` | 2 | Pose stays, texture/background errors and owl under-editing |
| `conditions` | 16 | 4 caption variants × 2 images × 2 seeds; limited bear heading effect, owl fails |
| `guided_full` | 4 | Historical source-anchored image lift; under-editing |
| `guided_paired` | 4 | Paired-lift controls; under-editing remains |
| `transport_full` | 6 | Historical source-anchored image lift; severe artifacts |
| `transport_roundtrip_control` | 2 | Same bad lift without any image warp also corrupts images |
| `transport_paired` | 6 | Zero intervention exactly reproduces BASE; warp variants still change pose / blur |

`material_pilot` has **no native FlowEdit velocity** and is never counted as a
complete FlowEdit trajectory. Offline guided/registration panels are also not
inference results. All images are development cases; no general benchmark claim.

## Numerical findings and artifacts

- [Uniform endpoints](outputs/analysis/endpoints.json), [CSV](outputs/analysis/endpoints.csv),
  [46-trajectory audit](outputs/analysis/audit.json).
- [Bear comparison](outputs/analysis/bear__00_comparison.png),
  [owl comparison](outputs/analysis/brown_owl__00_comparison.png),
  [all bear outputs](outputs/analysis/bear__00_all_methods.png),
  [all owl outputs](outputs/analysis/brown_owl__00_all_methods.png).
- The actual hard tube stays below radius .15 at all 33 updates; 18 updates are
  holds and 15 miss the proposed positive proxy-progress floor. This is a failure
  of joint task completion, not a successful edit hidden by a strict metric.
- Adding amplitude raises the conditional geometry response Gram minimum
  eigenvalue by a median factor 1.409 on four sources. Sharing parameters adds
  only a median factor 1.00177 over the independent model in these controls;
  global semantic proxies add only 1.00036. These weighted response Grams are
  not calibrated Fisher information or proof of pose observability.
- At the same real state, 4 / 16 / 32 CG iterations reduce linear residual from
  .213 to below .001, but endpoint geometry changes very little.
- A fixed-source re-encoding compensation is **not** a zero-intervention identity
  for an arbitrary current state. The corrected lift is
  `raw + E(corrected_image) - E(raw_image)`. Both no-warp trajectories then match
  all original FlowEdit states exactly, maximum error 0. This corrects the new
  image-intervention route; it does not invalidate a declared fixed-source
  appearance-family parameterization.

The `SAFE` name only means finite acceptance by the specified geometric proxy;
it does not guarantee semantic success, pose preservation or lack of blur.
Transport scripts now default to paired encoding and `SOURCE_NORM_SAFE`.
Guided-filter scripts also default to paired encoding. Historical failed lifts
remain available through explicit flags and retain all original outputs.

## Reproduction

Use existing local weights and inputs from the repository root. Completed runs
are skipped; preserve failures when making further variants. `v1` and `v2`
contain code snapshots; the `v3` snapshot was reconstructed after a CLI-only
change and verified against the hash recorded at experiment launch.

```bash
.venv/bin/python experiments/relation_tube_20261024/calibrate.py
.venv/bin/python experiments/relation_tube_20261024/run.py --stage local --pairs bear__00 --steps 33 --modes PHASE SHARED SHARED_FACT --tag local_current
.venv/bin/python experiments/relation_tube_20261024/run.py --stage full --pairs bear__00 --modes PHASE SHARED SHARED_FACT VELOCITY --tag v2
.venv/bin/python experiments/relation_tube_20261024/run.py --stage full --pairs bear__00 --modes BASE TUBE --tube-radius .15 --tag v3
.venv/bin/python experiments/relation_tube_20261024/solver_probe.py
.venv/bin/python experiments/relation_tube_20261024/roundtrip.py
.venv/bin/python experiments/relation_tube_20261024/material.py
.venv/bin/python experiments/relation_tube_20261024/tone_probe.py
.venv/bin/python experiments/relation_tube_20261024/material.py --full --chroma-only
.venv/bin/python experiments/relation_tube_20261024/views.py
.venv/bin/python experiments/relation_tube_20261024/conditions.py
.venv/bin/python experiments/relation_tube_20261024/pose_probe.py
.venv/bin/python experiments/relation_tube_20261024/guided.py
.venv/bin/python experiments/relation_tube_20261024/guided.py --full --anchored
.venv/bin/python experiments/relation_tube_20261024/guided.py --full --paired
.venv/bin/python experiments/relation_tube_20261024/transport.py --offline
.venv/bin/python experiments/relation_tube_20261024/transport.py --lift anchored --tag transport_full --modes SOURCE_NORM STEP_NORM SOURCE_RAW
.venv/bin/python experiments/relation_tube_20261024/transport.py --lift anchored --tag transport_roundtrip_control --modes NO_WARP
.venv/bin/python experiments/relation_tube_20261024/transport.py --lift paired --tag transport_paired --modes NO_WARP SOURCE_NORM SOURCE_NORM_SAFE
.venv/bin/python experiments/relation_tube_20261024/analyze.py
```

Run memory-heavy GPU jobs sequentially. One concurrent roundtrip-calibration
startup ran out of memory; its log is `roundtrip_oom_attempt.log`, and the
successful rerun is `roundtrip.log`. It is not counted as a method result.

The geometry annotation appended in `conditions.py` is manually verified for
these source images. No automatic pose extraction was implemented. Cropped-view
readouts and SD3 residual pose evidence failed diagnostic checks and were not
used as accepted semantic constraints. Top-64-row background metrics are
evaluation-only strips, never inference masks.
