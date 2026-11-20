# FlowEdit 双方向理论：真实 FLUX 验证

日期：2026-11-06。配套[理论与证明](../../docs/FlowEdit双方向深度研究_源耦合证书与干净域编辑_定理证明及闭环_2026-11-06.md)。

2026-11-09 [噪声—速度适配补充](../../docs/随机噪声与条件速度适配_动态源耦合与局部提升_2026-11-09.md)：运行 `.venv/bin/python experiments/flowedit_theory_real_20261106/check_noise_adaptation.py`，仅从旧源路径缓存计算实际速度局部提升的[几何误差](noise_adaptation_20261109/query_geometry.json)。不调用模型，不改旧实验或清晰端实际编辑状态。

后续 CPU 复分析：[实验归因与理论修订](../../docs/FlowEdit双方向研究_实验归因与理论修订_2026-11-06.md)。运行 `.venv/bin/python experiments/flowedit_theory_real_20261106/analyze_results.py`，在独立的 `analysis_v1/` 保存[数值与输入哈希](analysis_v1/analysis.json)、[归因图](analysis_v1/research_diagnosis.png)。不改原始 `outputs/run_v1/`、不重新调用模型；六项代数／解析一致性检查通过。

## 实验身份与运行顺序

同一组本地 FLUX.1-dev 权重，同一真实熊图和原完整源／目标文本。将原图缩放到 512 后重新做 FP32 VAE 编码；重新运行 512 的对照，原 1024 实验不混入数值比较。输入文字条件沿用已核查缓存，模型参数冻结，不使用编辑掩码或源特征注入。

```bash
.venv/bin/python experiments/flowedit_theory_real_20261106/prepare.py
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u experiments/flowedit_theory_real_20261106/run.py
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -u experiments/flowedit_theory_real_20261106/refine_gradient.py
.venv/bin/python experiments/flowedit_theory_real_20261106/evaluate.py
```

脚本拒绝覆盖已登记／启动的版本。协议先于 Transformer 推理写入 [protocol.json](outputs/run_v1/protocol.json)。原始日志为 [run_v1.log](run_v1.log)。

## 预先固定的核对

1. 1160 张量布局往返、严格加载，调用适配器与原生前向逐元素核对。
2. BF16 后接同权重 FP32，检查径向有限标量的完整梯度与有限差分。FP32 权重分布于两张 GPU，逐块 checkpoint；无完整 Jacobian 存储。
3. 固定 σ=.5、两个中点求积节点、同一标量，分别用完整输入梯度和便宜场方向提案；最多 8 步，最多四档缩步。
4. 源潜变量总位移 RMS≤.35。实际候选通过标量增益与源距离门后才提交。该观测完全可计算，不将其称为姿态或背景语义保证。
5. 以相同 FP32 算法运行两种 FlowEdit guidance 配置，保留不同预算；不按源保持指标单独宣称质量优势。
6. 源路径使用 Euler 2／4／8 步、Heun 16／32 步参照，检查 Hermite 插值缺陷、路径误差和编辑传播。
7. 八个随机查询样本用于诊断速度弦残差与方向方差；不把这些样本当成总体无偏性或 ODE 证书。
8. PSNR、SSIM、固定背景窗口和既有 SigLIP2 离线评分只在推理结束后运行。

## 精确投影版本的理论对应

本轮先验证最低配的总潜变量位移约束。令 D 为维数，S 为归一化径向标量，g=∇S，约束 K={z:||z−x||²/D≤R²}。候选

\[
y'=\Pi_K(y+\eta Dg)
\]

是严格凸问题 min_{z∈K} ||z−y||²/(2D)−ηgᵀ(z−y) 的唯一解。由凸最优性对 z=y，有 gᵀd≥||d||²/(ηD)。因此它保留主报告的变分不等式，且球投影精确满足实际保持集合；不需要给简单球约束估计曲率或运行大规模通用 QCQP。目标推进仍通过实际有限值门检查。便宜场提案不自动获得完整梯度的优化结论。

所有“实际值”在这里均指记录的浮点测量；没有自动获得真实函数评价的统一舍入误差证书。

## 结果与范围

主实验与单独登记的梯度补充均已完成。完整结论见[实际验证与失败分析](../../docs/FlowEdit双方向理论_真实FLUX实验验证与失败分析_2026-11-06.md)。

协议／代码 SHA、输入身份、接受状态与独立终点评价等 20 项产物审计全部通过，见 [verification.json](outputs/run_v1/verification.json)；代码快照和 64 个产物的哈希见 [artifact_sha256.json](outputs/run_v1/artifact_sha256.json)。这些审计不认证语义质量。

- [主对照图](outputs/run_v1/comparison.png)：完整梯度编辑不足，便宜方向提案产生彩色伪影；当前方案未通过实际编辑有效性验证。
- [源耦合对照](outputs/run_v1/source_coupling.png)：Euler2→Euler8 的源路径误差 .0656→.0183，编辑终点误差 .0127→.00840；三个短区间编辑仍主要保留棕熊。
- [有限步审计](outputs/run_v1/finite_gate_audit.json)：完整梯度接受 2 步，便宜提案接受 4 步，全部提交状态满足代理增益和总位移预算；没有把停滞报告为任务完成。
- [梯度补充](outputs/run_v1/gradient_refinement_v1/result.json)：原登记尺度未通过；在 5e−6／1e−6 的 RMS 扰动上，相对误差约 1.1%。原始失败保留，主轨迹不变。
- [图像及独立任务代理](outputs/run_v1/image_metrics.json)：源 PSNR、背景窗口与完整提示分差同时报告，不能用编辑不足的高 PSNR 宣称收益。

主实验 534 次封装前向＋1 次原生一致性前向、16 次 VJP；补充 70 次前向、12 次 VJP。checkpoint 重计算、加载、解码与评分开销另计，不作同 NFE 速度比较。

离线评价的首两次分别因 tokenizer 缺少最大长度和路径括号错误退出，已修正并完整重跑；`evaluate_v1.log`、`evaluate_v1_fixed.log` 保留失败，`evaluate_v1_final.log` 是成功日志。未改变生成轨迹。

低维 34 项数学核对重新运行并全部通过，副本见 [analytic_recheck.json](outputs/run_v1/analytic_recheck.json)，原始目录为[解析核对](../flowedit_closed_theory_20261106/README.md)。源路径用数值参照，网络 Lipschitz 常数仅取样诊断；本轮不能把其误差界当成全域已认证常数。
