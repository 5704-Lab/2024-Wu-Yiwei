# Original FlowEdit trajectory and RGB change spectra

Executed 2026-10-10 on four dataset pairs. Original SD3 Medium FlowEdit, no corrective intervention. T=50, n_max=33, seed0, original CFG3.5/13.5. Steps0–16 are explicitly idle; steps17–49 are active. Final latent matches retained baseline exactly in all four cases.

Run from workspace root:

    .venv/bin/python experiments/flowedit_frequency_trajectory/run.py
    .venv/bin/python experiments/flowedit_frequency_trajectory/export_spectra.py
    .venv/bin/python experiments/flowedit_frequency_trajectory/analyze.py
    .venv/bin/python experiments/flowedit_frequency_trajectory/render_conditions.py

run.py supports --pairs followed by cached pair IDs and --seed. It reuses inputs from experiments/stage1_full/inputs; no dataset or weights changes.

Outputs: aligned50-step overview, active33-step overview, five10-step larger panels, source and step spectra, shared colorbar, absolute and step-size-normalized temporal graphs. Every sample is one horizontal group with images above aligned spectra. Source RGB and VAE reconstruction are separately shown.

Per pair: frames/ contains512 RGB previews, spectra/ full-size spectra, spectra.npz numerical power arrays, metrics.json per-step energy/Parseval audit, trajectory.pt initial and50-step latent states. Step powers come from unquantized signed decoded RGB increments; RGB mean power with orthonormal FFT divided by pixel count. Color scale is shared, no per-image normalization. Fourier boundary leakage and DC are retained.

Interpretation and links: [report](../../docs/FlowEdit全时间步轨迹与变化频谱_脚本与观察_2026-10-10.md).

Condition annotations include exact source/target captions, requested semantic change and per-step active/idle ribbons. Conditions are fixed at active steps; no text interpolation is implied. render_conditions.py redraws panels and curves from saved artifacts without regeneration.

## Equal CFG1 rerun

Use run.py --source-cfg 1 --target-cfg 1 --output-dir experiments/flowedit_frequency_trajectory/outputs_cfg1. Pass the same --output-dir to export_spectra.py, analyze.py and render_conditions.py. compare_cfg.py creates the original-versus-CFG1 comparison. CFG captions read the stored protocol; original outputs are preserved. Endpoint distance to the old cache is an intentional different-CFG distance, not a replication error. See [CFG1 report](../../docs/FlowEdit_CFG均为1_全轨迹频谱对照_2026-10-10.md).
