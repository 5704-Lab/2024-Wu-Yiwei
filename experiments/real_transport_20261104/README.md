# 源锚定输运：真实图像实验（2026-11-04）

[完整报告](../../docs/源锚定关系输运_真实图像对应形变与有限恢复实验_2026-11-04.md)。完成真实基准源图的原生全空间描述子、源锚定对应、多点形变读出和受限有限恢复。

20 个已有源，按本轮 4 开发／4 校准／12 留出划分；340 个源条件化观测、320 个唯一潜变量，含 80 个原有实际 FlowEdit 干净状态。**已知全图二维变换不等于合法类别编辑；整体旋转恢复不能替代局部头转／类别形态约束。**

原 SD3 Medium、VAE 及文本编码冻结；无新训练模块、LLM、反演、外部正确目标或空间更新掩码。源纹理和文本软角色只用于测量，不与速度空间相乘。亮度子任务是显式协议，不是自由文本许可解析。

主要代码：

| 文件 | 作用 |
|---|---|
| `prepare.py`、`prepare_trajectories.py` | 已知变换的真实图像与实际干净状态；真值离线使用 |
| `tap.py` | 完整 O/C、post-AdaLN/K 原生钩子；不替换 processor |
| `match.py` | 无几何真值的对应，显式全局／对象／局部标架 |
| `weighted_frames.py` | 固定源 RGB 纹理拟合权重；不改变对应映射 |
| `run.py` | 共享／独立噪声，流式描述子与相似度，存映射而非巨量完整特征 |
| `analyze.py` | 开发锁定、校准、留出检验 |
| `reader.py` | 选定配置对实际 y 的有限重算 |
| `repair.py` | 旋转动作的成对 VAE 潜变量提升、亮度任务、有限验收 |
| `evaluate_repair.py`、`affine_audit.py` | 离线正确暗图参考、独立 RGB 配准及重采样匹配反向对照 |
| `summarize.py`、`verify.py` | 独立图表、统计、有限条件复核和溯源 |

正式结果：

- `outputs/analysis_v3/selection.json`：独立噪声开发选出的配置，同时固定共享噪声同配置对照。
- `outputs/analysis_v3/thresholds.json`、`heldout.json`：只用四源校准，12 源留出。旋转 44/48 有支撑检出，4 未覆盖；非旋转 4/72 误报。
- `outputs/repair_v1/`：初轮窄测量容差失败，6 源只有 owl 接受两步；保留失败。
- `outputs/repair_v2/`：校准仿射测量误差后，6 源均恢复整体旋转、保留 0.6 亮度编辑；其中 4 个源用于修复协议确认。
- `outputs/repair_v3_shared/`、`repair_shared_rgb_audit_v1/`：与原设计一致的共享噪声恢复及独立 RGB 评价，同六源消融。RMSE 降低中位数约 46.0%，不能将重复源再当独立样本。
- `outputs/repair_rgb_audit_v1/`、`affine_rgb_audit_v1/`：评价参考不进入运行；独立 RGB 确认剩余转动、全局伸缩和平移。
- `outputs/trajectory_weighted_v1/`：真实 FlowEdit 状态；未伪造终点类别正确性标签。
- `outputs/summary_v1/`：主统计、CSV、全部恢复图与实际 FlowEdit 图。
- `outputs/verification_v1/`：数值验收、权重与脚本身份，以及“经验相容不等于严格鲁棒认证”的明确边界。
- `outputs/final_audit_v1/`：共享恢复验收、对照图、最终源码快照与哈希；不覆盖早期记录。

`smoke_v1`、早期 `analysis_v1/v2`、`development_weighted_v1` 是内部开发产物。正式源纹理重拟合使用 `development_weighted_v2`；该版本修正早期中间文件未重算的 matrix/translation 评价字段，所有正式统计以修正后文件为准。

完整运行请用独立副本／新标签，已有结果不覆盖。主要阶段顺序：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/real_transport_20261104/prepare.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/real_transport_20261104/prepare_trajectories.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/real_transport_20261104/run.py --tag development_v1 --pairs bear__00 brown_owl__00 bus__00 cat_crown__00
.venv/bin/python experiments/real_transport_20261104/weighted_frames.py --input development_v1 --output development_weighted_v2
.venv/bin/python experiments/real_transport_20261104/analyze.py --stage lock
# Use selection.json for calibration_v1, heldout_v1 and trajectory_v1 runs.
# Refit frames to calibration_weighted_v1 / heldout_weighted_v1 / trajectory_weighted_v1.
.venv/bin/python experiments/real_transport_20261104/analyze.py --stage calibrate
.venv/bin/python experiments/real_transport_20261104/analyze.py --stage test
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/real_transport_20261104/repair.py --tag repair_v2 --bank independent
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/real_transport_20261104/evaluate_repair.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/real_transport_20261104/affine_audit.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/real_transport_20261104/repair.py --tag repair_v3_shared --bank shared
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/real_transport_20261104/evaluate_repair.py --tag repair_v3_shared --output repair_shared_rgb_audit_v1
.venv/bin/python experiments/real_transport_20261104/summarize.py
.venv/bin/python experiments/real_transport_20261104/finalize.py
```

严格复现早期版本需在同目录结构的独立副本恢复对应代码快照。`repair_v1` 使用的是当时静态光度容差版本，不能用当前 `repair.py` 冒充复跑该失败版本。

新理论完整版本仍缺多假设语义界、解剖部件绑定、类别完成困难负例与每步非零 FlowEdit 速度联合控制。当前可确认的是受限真实图像几何机制与有限恢复，不是完整语义编辑方法有效。
