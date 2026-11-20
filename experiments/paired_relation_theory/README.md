# Paired relation mathematics audit

[Current theory and revised strategy](../../docs/源图源文本与编辑状态目标文本_配对相关性与逐步修正_2026-10-12.md).

Run from the workspace root:

    .venv/bin/python experiments/paired_relation_theory/check_math.py

Forty synthetic FP64 nonlinear feature cases check the condition-frame rotation, free target strength, state derivatives, feature/latent cross-coupling, constrained proximal solution, active budget, feedback identity, and finite retraction. The state finite difference independently checks derivatives; text embeddings are never perturbed.

This is an analytical audit. No pretrained model or new real-image editing experiment is executed, and no semantic, identity-preservation, speed or originality result follows from these checks.
