"""Write a self-contained execution report from preserved numerical results."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'experiments/stage1_full/outputs'

def fmt(value):return f'{value:.6g}'
def main():
 s=json.loads((OUT/'summary.json').read_text());assert s['complete']
 diag_path=OUT/'fp64_outlier_diagnostics.json';diag=json.loads(diag_path.read_text())['results'] if diag_path.exists() else []
 b=s['branch_metrics'];q=s['qp_metrics'];failure=s['failures']['branches']
 dual_bad=sum(r['dual_residual']>1e-4 for r in failure);primal_bad=sum(r['baseline_primal_error']>1e-4 for r in failure)
 qp_ok=144-len(s['failures']['qp']);branch_ok=288-len(failure)
 text=rf'''# SD3 Medium 阶段 1 完整计算审查与异常诊断

日期：2026-10-05。对应方案：[分阶段实验设计](生成式无反演编辑_分阶段实验设计_2026-10-05.md)。

## 1. 结论

完整执行了 **24 张开发集源图 × 两个种子 × 三个噪声位置**，得到 **48 条基线轨迹、144 个状态、288 次分支导数审查**。全部预定样例保留，未访问机制集或留出集。全量完成标志为 `complete=true`；既定数值门槛判定为 **`passed={str(s['passed']).lower()}`**。

- 分支综合通过 {branch_ok}/288；对偶残差超阈值 {dual_bad} 次，速度一致性超阈值 {primal_bad} 次。这两项可能在同一分支重叠。
- QP、FP32 舍入后约束通过 {qp_ok}/144。
- 48 个同文本单步控制通过；8 次重复求导通过。
- 所有已完成轨迹的基线未来噪声保持不变：{str(s['all_noise_streams_unchanged']).lower()}。
- 此结果没有证明编辑语义更准确、非编辑区域一致性更强或可以减少步数。阶段 2 的语义机制门槛尚未运行。

## 2. 固定实验条件

使用已下载的原始 **SD3 Medium**，不是 SD3.5。权重与官方仓库 SHA 匹配，HF revision 为 `ea42f8cef0f178587cf766dc8129abd379c90671`。数据来自原 FlowEdit，固定 commit 为 `2620a9364f0f9c21368c36266e6200ff250273ec`；每张开发源图固定首个编辑对，不按结果选择目标。

分辨率 512；模型、编码器、潜变量、速度和导数使用 FP32。网络参数冻结，不训练、不微调、不反演、不使用编辑掩码。源图用 VAE 后验众数编码。基线配置为 T=50、n_max=33、n_min=0、n_avg=1，源 CFG=3.5、目标 CFG=13.5。

按实际访问噪声值的经验 80%、50%、20% 分位选择最近日程点：第 23、33、43 步，实际 t 约为 0.773268、0.594801、0.300147。时间定义采用官方 SD3 的 `network_timestep/1000`，没有把噪声分位直接当成网络时间。

官方 `calc_v_sd3` 以批量 4 计算源/目标 CFG；单分支自动微分使用 PyTorch 数学 SDPA，以支持前向模式 JVP。二者的前向数值差单独审查。随机噪声按每图种子 0、1 在 CPU 生成并冻结；不同图的同种子使用共同随机数。这复用了官方的噪声分布及更新式，不保证与其默认 GPU 随机数发生器逐元素相同。

阶段 1 始终继续原始 FlowEdit 基线，没有将计算的校正实际注入轨迹。

## 3. 计算对象

对源分支和目标分支分别计算

$$
J_i=D_{{e_i}}v_i,\qquad
u_i=\operatorname{{stopgrad}}(-h v_i),\qquad
g_i=J_i^\top u_i,\qquad r_i=J_i g_i.
$$

序列嵌入和 pooled 嵌入是两个独立可微输入块。源点仅对源条件求导，目标点仅对目标条件求导。不计算源、目标嵌入的差；没有通过给文本添加有限扰动来近似导数。随机文本方向仅用于 JVP/VJP 数学对偶审查，没有提交给编辑轨迹。

实际有限精度核对采用

$$
\epsilon_{{dual}}=
\frac{{|\langle J_i^\top u_i,d_e\rangle-\langle u_i,J_i d_e\rangle|}}
{{1+|\langle J_i^\top u_i,d_e\rangle|+|\langle u_i,J_i d_e\rangle|}},
$$

预设门槛为 1e-4；速度一致性同样为 1e-4。随机输出探针用于检查 $u^\top JJ^\top u$ 的非负性。

单步条件对比代理在共同候选点 q_E、共同 CFG=1 计算。它仍是冻结网络的条件对比向量，不等同准确语义奖励梯度。QP 使用 alpha=0.3、rho=1、lambda_s=lambda_u=1、半径 $0.1\|\Delta_0\|$；这些系数没有做图像质量调参。

低秩求解只存储两个响应方向及至多 2×2 的矩阵。24 个小维度退化与非退化问题上，它与原稠密求解器的最大位移差为 6.1082e-16。真实潜空间没有显式构造 65536×65536 矩阵。

## 4. 全量指标

| 检查 | 中位数 | 最大值 | 门槛/解释 |
|---|---:|---:|---|
| 对偶残差 | {fmt(b['dual_residual']['median'])} | {fmt(b['dual_residual']['max'])} | 1e-4 |
| 批量官方速度与单分支速度相对差 | {fmt(b['baseline_primal_error']['median'])} | {fmt(b['baseline_primal_error']['max'])} | 1e-4 |
| QP KKT 驻点残差 | {fmt(q['stationarity']['median'])} | {fmt(q['stationarity']['max'])} | 1e-7 |
| QP 原始可行性残差 | {fmt(q['primal']['median'])} | {fmt(q['primal']['max'])} | 1e-7 |
| QP 互补残差 | {fmt(q['complementarity']['median'])} | {fmt(q['complementarity']['max'])} | 1e-7 |
| FP32 实际位移约束残差 | {fmt(q['rounded_primal']['median'])} | {fmt(q['rounded_primal']['max'])} | 1e-7 |
| 实际位移与 FP64 QP 解的范数差 | {fmt(q['rounding_error_norm']['median'])} | {fmt(q['rounding_error_norm']['max'])} | 描述舍入幅度 |

随机 PSD 内积最小值为 {fmt(b['random_response_psd_inner']['min'])}；没有发现非有限的速度、梯度或响应。权重最小特征值的全量下界为 {fmt(q['W_min_eigenvalue']['min'])}。源距离与 QP 目标函数均没有超出预设允许量。

导数审查自身峰值 GPU 分配为 {b['max_gpu_memory_bytes']['max']/1024**3:.3f} GiB；不包含其他 GPU 作业或完整文本编码阶段。两张 GPU 分配种子与样例；种子 0 结束后，GPU 0 接手种子 1 末尾三个尚未启动的样例，原任务随后跳过已有完成记录。耗时包含与既有 GPU 作业共享资源，不能当作公平效率基准。

## 5. 全部失败分支

| 源图 | 种子 | 步 | 分支 | 对偶残差 | 速度相对差 |
|---|---:|---:|---|---:|---:|
'''
 for r in failure:
  text+=f"| {r['pair_id']} | {r['seed']} | {r['step']} | {r['branch']} | {fmt(r['dual_residual'])} | {fmt(r['baseline_primal_error'])} |\n"
 text+=r'''
失败原始记录完整保留于总表和对应样例目录；补充诊断不会将失败重标为通过。

## 6. 数值异常诊断

### 6.1 点积抵消

对已失败分支补充计算 $\|g\|\|d_e\|/\max(|g^\top d_e|,|u^\top Jd_e|)$。比值很大时，原分母会使较小的绝对舍入差成为较大的标量相对残差。这个诊断说明需要关注测试的数值条件，但本身不能证明梯度正确；原门槛保持不变。

### 6.2 注意力与批量前向

固定核对 flowers_draw、seed=1、step=43：统一为数学 SDPA 后，目标分支速度相对差仍为约 1.1036e-4。仅切换注意力后端没有消除该异常；批量与单样本矩阵运算的浮点差异需要继续核对，不能全部归因于注意力后端。

### 6.3 FP64 定点参考

'''
 if diag:
  text+='对全量异常中至多三个不重复分支做高精度参考，选择最大对偶残差、最大速度差及下一个最大对偶的分支。权重数值不训练，使用相同缓存文本和潜变量；探针固定为原 FP32 探针。FP64 审查不替代全量 FP32 门槛。\n\n'
  text+='| 样例/种子/步/分支 | FP32 对偶 | FP64 对偶 | 响应方向余弦 | FP64 批量速度相对差 |\n|---|---:|---:|---:|---:|\n'
  for r in diag:
   text+=f"| {r['pair_id']}/{r['seed']}/{r['step']}/{r['branch']} | {fmt(r['fp32_dual_residual'])} | {fmt(r['fp64_dual_residual'])} | {fmt(r['response_cosine_fp32_fp64'])} | {fmt(r['fp64_batched_isolated_primal_error'])} |\n"
  min_cos=min(r['response_cosine_fp32_fp64'] for r in diag)
  max_response_error=max(r['response_relative_error_fp32_fp64'] for r in diag)
  max_dual64=max(r['fp64_dual_residual'] for r in diag)
  max_primal64=max(r['fp64_batched_isolated_primal_error'] for r in diag)
  text+=f'\n三例 FP64 对偶残差最大为 {fmt(max_dual64)}，批量与单分支速度相对差最大为 {fmt(max_primal64)}。FP32/FP64 响应方向余弦最低为 {min_cos:.9f}；响应向量相对误差最高为 {max_response_error:.3%}。这些定点结果支持将已审查异常优先解释为点积抵消和浮点计算差异。方向保持较好不代表幅值完全稳定，也不能把三个分支的结论推广到全部未做 FP64 对照的状态。\n'
 else:text+='尚无已完成的 FP64 定点参考，不推断异常原因。\n'
 text+=r'''
## 7. 控制与代理解释的边界

48 个单步同文本控制覆盖 16 张源图、种子 0、三个检查点。相同文本及相同 CFG 的单步变化为零；不同 CFG 下允许原基线漂移，共同 CFG 的条件对比为零，因此新增校正退回零。这里没有另外执行完整的同文本恒等轨迹或解码 VAE 重构评价。

8 次重复求导的速度、两块文本梯度及传播响应均一致。这是同精度可重复性，不能证明语义预测有效。

8 个状态的 Jacobian 反对称诊断只用于解释条件对比代理，不参加主方法。非零结果支持继续按“网络代理工作量”描述；两个方向的测试不能证明场的全部性质，也未以重复精度噪声构造显著性检验。

## 8. 阶段决策与下一步

当前数值实现和低秩求解可以运行；严格 FP32 综合门槛未全部通过，因此没有启动阶段 2 的后续语义预测评价，也没有据此宣布完整编辑方法有效。

下一步应先固定速度计算接口的批量形状与求导后端，测量同精度执行噪声；用独立的计算审查协议处理抵消敏感的对偶统计量。若增加范数归一化残差，必须同时保留原指标，并在新运行前固定阈值，不能把本轮失败结果追认通过。随后再对机制集开展正负潜变量干预、冻结未来噪声的原始 FlowEdit 续推与独立图文代理评价。

QP 单步靠近源潜变量的数学保证依然不能替代准确的非编辑区域一致性，也不能证明目标语义未被抑制。是否值得支付 VJP/JVP 的成本，需要阶段 2–3 的方向预测与各向同性对照证据。

## 9. 复现与证据

- [实验执行说明](../experiments/stage1_full/README.md)
- [全量汇总](../experiments/stage1_full/outputs/summary.json)
- [分支指标](../experiments/stage1_full/outputs/branches.jsonl)
- [QP 指标](../experiments/stage1_full/outputs/qp.jsonl)
- [抵消诊断](../experiments/stage1_full/outputs/failure_cancellation_diagnostics.json)
- [注意力/批量诊断](../experiments/stage1_full/outputs/attention_backend_diagnostics.json)
- [FP64 定点诊断](../experiments/stage1_full/outputs/fp64_outlier_diagnostics.json)
- [官方实现来源记录](../vendor/flowedit/provenance.json)

缓存、模型、输入和执行源码的哈希均可从对应 manifest 与 summary 核对。原有低精度失败和单图预检保留，未用本次结果覆盖。
'''
 p=ROOT/'docs/SD3Medium阶段1完整计算审查与异常诊断_2026-10-05.md';p.write_text(text)
 print(p)

if __name__=='__main__':main()
