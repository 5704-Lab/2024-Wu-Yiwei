# Failure-directed follow-up, 2026-10-23

Completed: six full trajectories / 198 active corrections and 96 dense-observer
calibration cases. [Report](../../docs/相位许可失败原因对照_修正半径频率任务与稠密几何_2026-10-23.md),
[images](outputs/all_comparison.png), [audit](outputs/audit.json).
The three changes did not visibly restore the source heading. Local contrast
normalization reduced false flow on appearance-only interventions, but this new
observer has not been applied as a controller or validated against real pose truth.

Previous experiments lowered phase residual while retaining pose errors. This run
tests three sequential changes on bear and owl: correction radius .2 -> .6;
emphasize lowest frequency (higher-band amplitude weights .25); remove the task
floor. Existing inputs/seed/CFG/noises and source reference are unchanged.

This is an exploratory two-image causal comparison, not independent validation.
The reduced three-direction search space remains unchanged. LOW_NO_TASK is a
diagnostic for semantic tradeoffs, not an accepted deployable method.

`solver.py` is a preserved copy of the preceding real solver with an explicit
disable-task switch. Old results are not overwritten. Run from workspace root:

```
.venv/bin/python experiments/phase_followup/run.py
.venv/bin/python experiments/phase_followup/dense_probe.py
.venv/bin/python experiments/phase_followup/analyze.py
```
