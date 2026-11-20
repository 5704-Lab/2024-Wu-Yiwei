# 原生闭环真实模型验证（2026-10-30）

[完整报告](../../docs/原生闭环真实模型实验_配对观测实例绑定与转头辨识_2026-10-30.md)。本轮实现了原 SD3 的 O/C/B 配对旁路，完成实际模型观测、求导、新源选择性和软实例绑定调整。**部件关系读出未通过准入，未运行完整硬反馈轨迹，未确认编辑质量收益。**

24 个受控源／312 状态，另一个真实源／9 状态；其中新源 8 个／120 状态。每状态四种相同提示、时刻与噪声的源／当前配对配置。源／当前都是实际干净潜变量。原模型冻结，无额外训练、LLM、参考编辑图、反演或空间更新掩码。软概率仅作测量。

| 文件 | 功能 |
|---|---|
| `encode.py` | 原 T5 概念初值与主提示内容 token 掩码（非编辑区域掩码） |
| `probe.py` | 局部钩子读取投影前 O/C；动态／源冻结只读 B；不更改主 attention processor |
| `run.py` | 主观测、恒等／只读接口检查、模型／输入身份记录 |
| `fresh.py` | 原有渲染器的新源与原 VAE 干净编码；真值不入测量 |
| `analyze.py` | 开发与四源合法阈值锁定，所有配置公开 |
| `audit.py` | 历史源与新源分别审计，禁止重拟合阈值 |
| `diagnostics.py` | 实际 y 的自动求导／中心差分核对 |
| `binding.py` | 源图软实例图调整；仅观测窗口，不乘编辑更新 |
| `landmarks.py` | 离线独立投影部件中心，仅评价使用 |
| `natural.py` | 一张真实熊图、实际干净轨迹与几何／光度控制 |
| `summarize.py` | 图表、CSV 和简表 |
| `evidence.py` | 同源配对排序、相机分层、溯源与代码快照 |

正式主产物：`outputs/native_v3/`、`outputs/native_fresh_v1/`。`native_v1` 是旁路投影钩子冲突的失败尝试；`native_v2` 为四状态冒烟，均不并入正式计数。

关键结果入口：

- `outputs/summary_v1/fresh_metrics.csv`：代表配置合法接受／错误转头拒绝。
- `outputs/analysis_v1/limits.json`：新源提取前锁定的阈值。
- `outputs/analysis_v1/native_fresh_v1_audit.json`：全部 84 种观测统计与逐状态能量。
- `outputs/binding_v1/fresh_holdout.json`：24 种软实例图配置，无新源调参。
- `outputs/landmarks_v1/result.json`：部件位置误差与角度跟随诊断。
- `outputs/evidence_v1/paired.json`：同源同形态完成度配对与相机分层。
- `outputs/evidence_v1/snapshots/`：实际脚本快照。
- `outputs/summary_v1/*.png`：独立图表与部件错配可视化。

实际执行顺序如下，路径按仓库根目录。产物不可覆盖，完整复跑请使用独立工作副本或调整输出目录及审计读入路径；已有结果可直接读取。

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/native_feedback_20261030/encode.py
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/native_feedback_20261030/run.py --tag native_v3
.venv/bin/python experiments/native_feedback_20261030/analyze.py --stage dev
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/native_feedback_20261030/fresh.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/native_feedback_20261030/run.py --dataset /mnt/windows_F/wyw/CorrectFlow/experiments/native_feedback_20261030/outputs/fresh_v1 --latents /mnt/windows_F/wyw/CorrectFlow/experiments/native_feedback_20261030/outputs/fresh_v1/latents --tag native_fresh_v1
.venv/bin/python experiments/native_feedback_20261030/audit.py
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/native_feedback_20261030/diagnostics.py
.venv/bin/python experiments/native_feedback_20261030/binding.py
.venv/bin/python experiments/native_feedback_20261030/landmarks.py
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/native_feedback_20261030/natural.py
.venv/bin/python experiments/native_feedback_20261030/summarize.py
.venv/bin/python experiments/native_feedback_20261030/evidence.py
```

只读性和恒等配对最大差为零，开发源单标量导数一致；这些不证明语义正确。新源动态竞争角度合法 39/40、两方向错误转头拒绝 0/16；冻结 QK 合法 40/40、拒绝 0/16。源软图改善对象分离，仍不能可靠跟随部件。原始 O/C 同源配对排序有转头信息，但距离阈值受相机与合法外观影响。后续应改善结构化读出，再接入已有最小约束控制器。
