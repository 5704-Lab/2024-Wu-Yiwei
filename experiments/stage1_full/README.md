# 阶段 1：SD3 Medium 完整 FP32 计算审查

目的：核对各自文本条件的自动微分、低秩 QP 和实际状态舍入。此阶段不评价语义编辑质量，不更新模型参数，不访问留出集。

## 固定范围

- 开发集 24 张源图，每图固定首个编辑对；种子 0、1。
- 原 FlowEdit SD3：T=50，n_max=33，n_min=0，n_avg=1，源 CFG=3.5，目标 CFG=13.5。
- 分辨率 512；模型、VAE 和文本编码 FP32；网络参数冻结。
- 三个检查点取实际访问噪声值的经验 80%、50%、20% 分位对应最近点：步 23、33、43。
- 使用官方 `calc_v_sd3` 和 SD3 时间定义 `timestep/1000`；主轨迹始终按原 FlowEdit 更新，不应用实验校正。
- 源、目标的序列及 pooled 嵌入分别微分。`u_i=stopgrad(-h*v_i)`，`g_i=J_i^T*u_i`，`r_i=J_i*g_i`。不使用两个文本的嵌入差。
- 每例在 CPU 独立生成并冻结 33 个高斯噪声，导数与诊断使用另一个随机数发生器。相同种子跨源图使用共同随机数。

## 预设数值门槛

对偶残差及官方批量速度与单分支求导接口的相对差均为 1e-4；QP KKT 和 FP32 舍入后约束残差为 1e-7；重复求导相对误差为 1e-6。全部失败保留，不以补充诊断替换原门槛。

除 144 状态、288 分支外，另做：

- 16 源图、种子 0、三个点的 48 个单步同文本控制；不称为另外执行了完整恒等轨迹。
- 前 4 源图高噪声点的 8 次分支重复求导。
- 前 8 源图高噪声点的状态 Jacobian 对称性诊断；它不参与主方法或通过门槛。
- 低秩实现额外在 24 个 17 维问题核对稠密参考，涵盖共线、零响应和不同球半径；最大位移差 6.1082e-16。

QP 权重是 `I + Q_s + (I-Q_a)`，alpha=0.3、rho=1，位移半径为 `0.1*||Delta_0||`。这些值仅用于计算检查，尚未做图像质量调参。若目标响应缺失，相应权重置零；若条件对比代理为零，新增校正退回零。

## 执行

在仓库根目录使用本地 `.venv`：

```bash
.venv/bin/python experiments/stage1_full/prepare_inputs.py --device cuda:0
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/stage1_full/run_audit.py --device cuda:0 --seed 0
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/stage1_full/run_audit.py --device cuda:1 --seed 1
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/stage1_full/summarize.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/stage1_full/analyze_failures.py
```

本轮种子 0 完成后，GPU 0 通过 `run_audit_shard.py --seed 1 --case-start 21 --case-stop 24` 接手种子 1 最后三个尚未启动的样例；原种子 1 任务随后按完成记录跳过它们。分配不改变条件、种子或计算式。

已有 `complete.json` 的样例可恢复；更改协议或代码后的重新实验应使用新输出目录，不能覆盖已有结果并沿用旧结论。`outputs/summary.json` 的 `complete` 和 `passed` 分开报告。

数值异常诊断：`check_attention_backends.py` 对固定异常状态比较注意力后端；全量完成后 `diagnose_fp64.py` 选最大对偶、最大速度差及下一个最大对偶的至多三个不重复分支做 FP64 参考。它们不改变全量 FP32 的通过判定。

## 输出

`inputs/manifest.json` 保留编码哈希与时间日程。`outputs/` 按源图和种子保存分支指标、QP 指标、控制、诊断和必要状态；总表保留缺失与失败。对应的完整报告位于 `docs/`。
