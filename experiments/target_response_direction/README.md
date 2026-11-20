# Clean-state target-response direction precheck

Executed 2026-10-09. Frozen SD3 Medium; no training, inversion, spatial mask, or new complete edited-image comparison.

Report: [results and gradient-definition audit](../../docs/FlowEdit干净状态响应方向_首轮预检结果与梯度定义审查_2026-10-09.md).

## Commands

    .venv/bin/python experiments/target_response_direction/run_precheck.py --analytic-only
    .venv/bin/python experiments/target_response_direction/run_analytic_edges.py
    .venv/bin/python experiments/target_response_direction/run_precheck.py --device cuda:1 --dtype float32 --checkpoint --pair bear__00
    .venv/bin/python experiments/target_response_direction/run_precheck.py --device cuda:0 --dtype float64 --checkpoint --shard --pair bear__00
    .venv/bin/python experiments/target_response_direction/run_precheck.py --device cuda:0 --dtype float64 --checkpoint --shard --pair cat_crown__00 --scales .01,.001,.0001
    .venv/bin/python experiments/target_response_direction/run_definition_audit.py --device cuda:0
    .venv/bin/python experiments/target_response_direction/summarize.py

Inputs reuse retained experiments/stage1_full/inputs/ and original FlowEdit snapshots under experiments/stage1_full/outputs/. Cached z is the current clean editing state; zE is the original next candidate. All main responses use these clean tensors at auxiliary time zero.

The two-GPU FP64 path needs both GPUs. The initial single-GPU FP64 OOM and initial device-map errors are preserved in logs and failure JSON. Earlier script versions are kept to match source hashes in previous results.

Four FP32 states: bear__00, brown_owl__00, bus__00, cat_crown__00, seed0, step33. FP64 checks cover bear and cat crown only.

The passed_numeric JSON field means the internal mixed-chain identity check only, not the complete stage-1 gate. The armijo_without_numerical_margin field means a diagnostic calculation, not a deployed accepted correction.

The registered two proposals were radius factors 1 and 0.5. Smaller factors are failure diagnostics; they were not retroactively accepted as the registered method. No complete corrected trajectory or heldout experiment ran.
