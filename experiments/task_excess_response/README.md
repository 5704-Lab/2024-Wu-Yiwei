# Clean-state task excess response experiment

Protocol: [registered design](../../docs/无终点冻结逐步y修正_真实图像实验设计_2026-10-12.md).

Frozen SD3 Medium and SigLIP2 Base. Inputs are original FlowEdit source RGB, source/target text and cached encodings. No inversion, training, spatial editing masks, contour references or baseline endpoint constraint.

The source Gauss–Newton metric is approximated by a four-probe diagonal. This is a declared local approximation; projection uses that same positive metric. Calibrated responses are step-local and are not semantic truth.

Run from workspace root:

    .venv/bin/python experiments/task_excess_response/run.py --stage calibration
    .venv/bin/python experiments/task_excess_response/run.py --stage local
    .venv/bin/python experiments/task_excess_response/run.py --stage full
    .venv/bin/python experiments/task_excess_response/analyze.py

Every image comparison includes original FlowEdit. Numeric task logits and source descriptors must be read together with the images.

Completed: calibration, 60 local trials, 24 full trajectories and a post hoc BASE-state derivative/model audit (`diagnose.py`). [Results and definition audit](../../docs/无终点冻结逐步y修正_真实图像实验结果与定义审查_2026-10-12.md). Source descriptors improved, but pose, viewpoint and identity shifts were not visibly resolved. Contrast progress can accompany an absolute target-score decrease. This experiment is retained as evidence for revising the strategy, not as validation of the new paired-relation method.
