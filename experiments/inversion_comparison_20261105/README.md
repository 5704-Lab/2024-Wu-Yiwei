# FlowEdit 与直接 ODE 反演编辑：单张真实图（2026-11-05）

任务：原 FlowEdit 基准熊图，源提示中的 brown bear 替换为 black bear，其余场景描述保持。原 SD3 Medium、VAE、缓存文本编码，FP32、512²，原 shift=3 的 50 点 Euler 网格。无训练、空间更新掩码、特征注入或反演优化。

对比方法：

| 方法 | 起点与流程 | 编辑相关单分支计算次数 |
|---|---|---|
| FlowEdit | n_max=33, n_min=0, n_avg=1；每步源图共享噪声构造源／目标查询，积分速度差 | 33×4=132 |
| 普通部分 ODE 反演 | 源条件从 0 正向积分至 σ=0.85014296，再用目标条件反向走 33 步 | 33×2+33×2=132 |
| 普通完整 ODE 反演 | 源条件从 0 正向积分至 1，再用目标条件反向走 50 步 | 50×2+50×2=200 |

源 CFG=3.5，目标 CFG=13.5；所有方法一致。普通反演使用左端点显式 Euler，是标准 ODE 反演基线，不是专门的 RF-Inversion、精确可逆求解器或带注意力控制的反演编辑。

另运行两条同源提示、CFG=3.5 的反向重建，测量反演往返误差。图像→噪声与噪声→图像的显式 Euler 不保证精确互逆。源提示／CFG 一致重建控制与目标编辑的 CFG 不同，不将两者的差全部归因于文本替换。

`run.py` 新执行所有方法，FlowEdit 重放验证旧种子 0 的每步噪声哈希和最终潜变量。反演是确定性的，不声称其与 FlowEdit 具有相同“随机噪声种子”。完整反演的前 33 步正向前缀即部分反演状态，可精确复用，不使用目标信息。

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u experiments/inversion_comparison_20261105/run.py --device cuda:0 --tag fresh_tag
.venv/bin/python experiments/inversion_comparison_20261105/summarize.py --tag fresh_tag
```

已有输出目录拒绝覆盖。

主要产物位于 `outputs/bear_v1/`：

- `comparison.png`：源图、FlowEdit、部分反演编辑、完整反演编辑。
- `comparison_matched.png`：主对比，源图、FlowEdit、相同范围的普通 Euler 反演编辑。
- `head_detail.png`：同一源坐标裁剪，未对齐、不进入模型更新。
- `reconstruction.png`：源图、VAE 重建、部分／完整 ODE 同文本重建。
- `protocol.json`、`complete.json`、`metrics.csv`：完整提示、网格、数值身份检查、计数及源差指标。
- `states.pt`：真实潜变量和逐步轨迹；`artifact_audit.json`：产物哈希。

重建两条诊断另加 166 个单分支计算；保留的完整结果对应 564 个单分支、166 次 Transformer 调用。初始进程收到 SIGTERM（退出码 143），在完整反演的反向阶段中断；从检查点复用已完成的 FlowEdit、正向反演、部分反演，只重算完整反向阶段，使用 `resume_and_finalize.py`。中断前完整反向额外至少执行了 46×4=184 个单分支，不能把保留结果计数当作总实际花费。恢复细节与脚本身份见 `complete.json`。

成批调用与单分支计数不同；各阶段时间包含同批重建支路，不能直接当作各编辑方法的独立速度比较。像素源差包括合法改色／形态变化，不能单凭该指标判定编辑质量或姿态正确性。

初始结果中的源重建失真明显，因此另补 `heun_check.py` 的部分反演 Heun 数值对照：同一 33 个区间、相同端点／提示／CFG，两方向每区间两次速度评价。**它并未改善重建**，生成了源图没有的房屋；重建 RMSE 从 Euler 的 0.105639 上升为 0.206370。其额外花费为 132 次 Transformer、396 个单分支（含重建诊断）；反演＋目标编辑本身为 264 个单分支，不能声称与 FlowEdit 等计算量。输出位于 `outputs/heun_partial_v1/`，并通过 `compare_heun.py` 生成同源图和重建对照。

所有保留结果合计 298 次 Transformer、960 个单分支；中断额外已记录的 184 个单分支另计。Heun 在此网格失败不证明精确反演失败，也不证明高阶求解器一般更差；不将此次示例推广为全部反演方法的优劣结论。

[实验报告](../../docs/FlowEdit与普通ODE反演编辑_单张真实熊图对照_2026-11-05.md)。算法依据：[FlowEdit 原文的 ODE 反演及免反演构造](https://arxiv.org/html/2412.08629v1)，实现参照本地 `vendor/flowedit/FlowEdit_utils.py`。本实验只提供一个源的直观证据，不代表所有反演方法或一般任务上的结论。
