# Paired representation and incomplete-description validation

[Execution registration](../../docs/配对相关性与描述不完整_实验执行登记_2026-10-13.md).

Frozen SD3 Medium and SigLIP2 Base. Only original RGB, original source text, target text and text versions formed by deletion are inputs. No added LLM alignment, reference images, editing masks, training, fine tuning or inversion.

Completed work:

- Source RGB and 34 cached latent states per pair, perturbation diagnostics and previous method endpoints.
- 36 local paired-residual proximal candidates at 12 actual states.
- 36 complete image trajectories across 4 pairs and 9 description/guidance methods.
- Post hoc derivative, paired-loss equivalence and all-position patch-relation checks.

The local paired-residual method has not been advanced to complete trajectory correction: its instance observability is not established. The complete trajectories are FlowEdit description/CFG controls and the explicitly labeled counterfactual velocity decomposition, not a full validation of paired-residual proximal editing.

Run from the workspace root, sequentially for GPU memory:

    .venv/bin/python experiments/paired_relation_validation/observe.py
    .venv/bin/python experiments/paired_relation_validation/local.py
    .venv/bin/python experiments/paired_relation_validation/encode_conditions.py
    .venv/bin/python experiments/paired_relation_validation/condition_runs.py
    .venv/bin/python experiments/paired_relation_validation/derivative_audit.py
    .venv/bin/python experiments/paired_relation_validation/analyze.py

To add only missing endpoint readouts, use `derivative_audit.py --endpoints-only`.

The initial five-item counterfactual implementation exceeded its numerical gate on one prompt pair. It is preserved in `outputs/counterfactual_batch5_attempt/` and `condition_runs_batch5_attempt.py`; final comparisons use two four-item batches for all eight counterfactual trajectories. All original baseline latents are exactly reproduced, and final counterfactual batch-compatibility errors are zero. GPU timing was under shared load and is not a dedicated speed benchmark.

Every comparison includes original FlowEdit. Global distances and matching logits are proxies; they do not define exact edit success, unchanged-region accuracy, density modes or originality.
