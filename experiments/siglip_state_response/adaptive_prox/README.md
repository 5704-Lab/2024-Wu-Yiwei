# Adaptive response proximal experiment

Date: 2026-10-09. Four previously used development sources. Frozen SD3 Medium, VAE and SigLIP2 Base; no inversion, training, finetuning or spatial editing masks.

## Registered before execution

See protocol_local.json and protocol_full.json. Fixed latent coordinate basis, omega range [0.25,4], power 2, response RMS floor 1e-8, gamma=0.1 times baseline increment norm, V_floor=0, V_cap=10000. Local lambda grid [0.25,1,4]; full trajectory lambda=1 selected before results, no full-result parameter tuning.

G: baseline increment plus semantic drive. ISO: total-increment proximal shrinkage with mean adaptive weight. ADAPT: response-weighted proximal shrinkage. SHUFFLE: permuted weight coordinates, local only. Norm control rescales all local total increments to ADAPT's increment norm at lambda1, and does not solve the original QP.

All methods share budget rules; actual increment/correction norms and runtimes differ. The full experiment is not a same-total-time comparison. Original FlowEdit images appear in every generated comparison panel.

## Commands from workspace root

    .venv/bin/python experiments/siglip_state_response/run_adaptive_prox.py --stage local
    .venv/bin/python experiments/siglip_state_response/run_adaptive_prox.py --stage full
    .venv/bin/python experiments/siglip_state_response/run_adaptive_norm_control.py
    .venv/bin/python experiments/siglip_state_response/summarize_adaptive_prox.py

The norm control used GPU1; generation used GPU0. Environment recorded in environment.json. Code SHA256 is in each stage protocol. The full cohort has four pairs times three corrected variants times 33 actual integration steps. Each trajectory artifact stores all step diagnostics and the final latent. BASE bear is additionally re-generated for exact cached-baseline verification.

Analytic tests compare the closed-form/radius solution to independent SciPy SLSQP and exercise zero gradient, zero state increment, integration signs and query identity. Input cache is experiments/stage1_full/inputs, original trajectory states originate in experiments/siglip_state_response/baseline_states. Seed0 CPU noise sequence and original CFG 3.5/13.5 are retained.

Optimized SigLIP logits are diagnostics, not independent task success or source preservation. Local-state gains must not be treated as full-trajectory quality. Report qualitative task evidence separately. No heldout or reduced-step experiment in this cohort.
