# FlowEdit increment SVD and spatial spectrum

Experiment design plus a completed offline FLUX pilot, 2026-11-17. See the [Chinese report](../../docs/FlowEdit单步增量的SVD与空间谱_实验设计及缓存预实验_2026-11-17.md) for derivations, visual observations, and the planned online controls.

The input is the existing 1024px bear comparison (`sota_inversion_20261105/outputs/comparison_v1/results.pt`). Both FlowEdit guidance configurations are analyzed: 24 active increments each. Unpack native FLUX `(1,4096,64)` to `(1,16,128,128)` before the spatial-by-channel SVD. Differences are formed in FP64 from the cached BF16 states; the unrounded model velocity is not available here.

`outputs/pilot_v2` is the authoritative pilot. It contains 48-step numerical analyses, 13-arm local interventions at active steps 1/12/24 for the author guidance setting, and frozen-increment endpoint replay. The run used 56 FP32 VAE decodes, 0 Transformer calls, and approximately 36.4 seconds. Full replay exactly reproduces the previously saved clipped baseline RGB. No state-dependent velocity was re-evaluated, so this is **not an online modified FlowEdit evaluation**.

From the repository root, using a fresh output directory:

```bash
.venv/bin/python experiments/flowedit_delta_spectrum_20261117/analyze.py \
  --output experiments/flowedit_delta_spectrum_20261117/outputs/pilot_v3
```

`--statistics-only` skips the VAE. `--device cuda:1` selects the other GPU. Existing output directories are rejected. No original inputs, outputs, model weights, or prompts are overwritten.

Useful artifacts:

- [Full visual comparison](outputs/pilot_v2/flowedit_author/frozen_replay.png).
- [Same spectra, different spatial update](outputs/pilot_v2/flowedit_author/spectral_blind_spot.png).
- [Local step 24](outputs/pilot_v2/flowedit_author/local_step_24.png).
- [Trajectory statistics](outputs/pilot_v2/trajectory_statistics.png) and [rank-frequency table](outputs/pilot_v2/joint_rank_frequency.png).
- `protocol.json`: inputs, hashes, layouts, methods, frequency units, actual call count.
- Per configuration: `steps.json`, `steps.csv`, `audits.json`, `decompositions.npz`.
- `flowedit_author/visual_metrics.json`: local and replay float-RGB metrics, decoder non-additivity, baseline reproduction error.
- Individual 1024px decoded images, local comparison boards, and the spatial SVD factors.

FFT powers sum over channels and frequencies to latent squared norm. Band thresholds are radii .0625 and .25 **cycles per latent pixel**, with DC included in low. The SVD energy is uncentered; centered covariance is computed separately to verify its exact equivalence to centered SVD. `joint_rank_frequency` sums to both the SVD energy fractions and the frequency-band fractions.

Intervention controls include rank 1/4/8, tail after rank 4, energy-matched scalar scaling, rank 4 restored to full norm, centered rank 4 with DC preserved, DC-only, three Fourier bands, and circular horizontal shift by 16 latent pixels. The shift preserves singular values and per-channel Fourier powers while changing spatial alignment. Its visual failure is an identifiability counterexample, not an editing proposal.

Numerical assertions verify reconstruction, Parseval, covariance/SVD equivalence, orthogonal tail energy, equal-energy control, joint energy marginals, translation invariance, and telescoping replay. Maximum relative algebra error across 48 increments: approximately 8.03e-15. RGB metrics use FP64 accumulation on unclipped FP32 decoder output; PNG is clipped for display. Heatmaps share a scale within each board, with that board's maximum printed above them.

The preliminary `pilot_v1` is retained for provenance. It used another 56 VAE calls; its FP32 cosine accumulation could slightly exceed 1. `pilot_v2` uses FP64 accumulation and adds decoder additivity measurements. Use v2 for numerical interpretation. No online experiment or held-out image evaluation has been executed.
