# Multi-source FlowEdit increment SVD and spatial spectra

24 distinct development source images, 3 paired noise seeds, frozen FLUX.1-dev at 512px. Main report: [Chinese design and findings](../../docs/FlowEdit增量SVD与空间谱_24源多种子及在线干预_2026-11-17.md).

The registered main experiment contains 72 baseline trajectories and 84 extra state-dependent online interventions on 12 predeclared sources (seed0). Every run has 24 active steps with two separate Transformer calls each: 156 runs, 7488 planned calls. All source/target guidance settings are 1.5/5.5; no editing masks, training or inversion. Unlike frozen-increment replay, the online arms re-query velocities at their modified states.

`prepare.py` registers the protocol before inference and prepares the same frozen VAE and CLIP/T5 conditions for all 24 cached task IDs. It records token lengths and truncation counts. `run.py` executes two independent GPU shards, saves original intended updates, transformed proposals, actual BF16 states and rounding measurements. The top-4 projection uses FP32 eigendecomposition of the 16-channel Gram matrix and was checked against direct SVD.

`measure.py` analyzes both intended and effective increments in FP64. It evaluates all sources with local decoder interventions at steps 1/12/24, fixed-increment replay, equal-energy controls, random channel projection, softened low-pass, translation and orthogonal channel sign flip. All sources receive a step-24 amplitude scan. This process can wait for each source's inference to complete.

`score.py` adds frozen SigLIP2 source/target short-text contrast as a non-optimized auxiliary probe, not a semantic success certificate. `summarize.py` aggregates within seed, then equally over seeds and sources, with descriptive source bootstrap intervals and complete visual contact sheets. `auxiliary.py` separately checks 8 existing SD3 trajectories (4 sources, two CFG settings), layout dependence and seed-subspace overlap.

Commands, from repository root:

```bash
.venv/bin/python experiments/flowedit_delta_spectrum_multi_20261117/prepare.py
.venv/bin/python experiments/flowedit_delta_spectrum_multi_20261117/run.py --worker 0
.venv/bin/python experiments/flowedit_delta_spectrum_multi_20261117/run.py --worker 1
.venv/bin/python experiments/flowedit_delta_spectrum_multi_20261117/measure.py --worker 0 --wait
.venv/bin/python experiments/flowedit_delta_spectrum_multi_20261117/measure.py --worker 1 --wait
.venv/bin/python experiments/flowedit_delta_spectrum_multi_20261117/score.py --wait
.venv/bin/python experiments/flowedit_delta_spectrum_multi_20261117/summarize.py
.venv/bin/python experiments/flowedit_delta_spectrum_multi_20261117/auxiliary.py
.venv/bin/python experiments/flowedit_delta_spectrum_multi_20261117/finalize.py
```

The two run workers may execute concurrently, preferably loading the second after the first model is on its GPU to limit CPU peak memory. Existing prepared studies are rejected; inference resumes only jobs with completed markers. Summary plots are derived artifacts and may be regenerated. Do not change `run.py` during inference. For a new protocol, use a separate versioned output location in a copy of the preparation and runner scripts.

All artifacts are under `outputs/study_v1/`: `protocol.json`, input hashes, per-worker weight round-trip audits, `runs/<pair>/seedN/<arm>/trajectory.pt`, per-source `measurements/`, auxiliary `scores/`, and source-level `aggregate/`. PNGs are clipped for display. Numerical local/replay metrics accumulate unclipped FP32 decoder outputs in FP64, and frequencies use cycles per latent pixel, not RGB pixel.

The complete cohort has executed: 156 trajectories, 7488 Transformer calls, 1476 measurement VAE decodes plus 24 preparation VAE decodes and 24 encodes. All 24 baseline replay images exactly reproduce their paired baseline endpoint. The authoritative completion counts and numerical audit are in [verification.json](outputs/study_v1/aggregate/verification.json). Maximum decomposition-identity relative error is approximately 1.49e-14; the Laplacian/gradient-energy identity has error below 2.36e-16.

Main visual pages: [online1](outputs/study_v1/aggregate/online_page1.png), [online2](outputs/study_v1/aggregate/online_page2.png), [online3](outputs/study_v1/aggregate/online_page3.png). All 24 sources also have replay pages and three-seed pages. [Source trajectories](outputs/study_v1/aggregate/source_trajectories.png) and [summary.json](outputs/study_v1/aggregate/summary.json) contain the principal statistics. This cohort is development data; no held-out generalization or independent human-rated success rate is claimed.

Process provenance is preserved in [process_receipts.json](outputs/study_v1/aggregate/process_receipts.json). The tool reported exit 143 when collecting the two long-running inference workers and scorer. Both inference logs end with `WORKER_COMPLETE`; the scoring completion artifact contains all 180 images. All registered output artifacts and numerical audits were verified after collection. The cause of the process exit status is not established; a clean OS process exit is not claimed.
