# 2026-10-06 条件响应数值修订

R1：[固定协议](protocol.json)、[汇总](outputs/summary.json)。检查 26 诊断状态、52 FP32 分支与 6 FP64 配对分支。R1 完整执行，三个状态、四分支的能量恒等式误差超阈值，判定未通过。旧 2026-10-05 结果保留。

R2：[精度退回协议](../stage1_adaptive/protocol.json)、[汇总](../stage1_adaptive/outputs/summary.json)。复用 R1 的 23 个通过状态；三个失败状态整个局部步骤重新使用 FP64，提交 FP32 后复核。R2 在这组诊断状态通过，不是独立数据泛化证据。

统一计算函数位于 `src/correctflow/conditional_response.py`。源码采用单样本数学 SDPA 和可恢复的两级标量缩放，条件度量 M=I 不变；不使用跨文本差分、不更新权重。

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/stage1_revision/run.py --device cuda:1 --precision fp32
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/stage1_revision/run.py --device cuda:0 --precision fp64
.venv/bin/python experiments/stage1_revision/summarize.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/stage1_adaptive/run.py
.venv/bin/python experiments/stage1_adaptive/summarize.py
```

首次 FP64 任务读取了尚未完成的 `.pt`，发生一次执行错误；修复为等待 JSON 完成标记和原子文件提交。错误日志、修复前源码、重试日志保留；重试补齐一个缺失状态，没有删除不利数值结果。

配套：[理论讨论](../../docs/条件响应数值修订_尺度分离与校正稳定性_2026-10-06.md)、[单独实验设计](../../docs/条件响应数值修订与机制先导实验设计_2026-10-06.md)。
