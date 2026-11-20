# FlowEdit源与目标CFG均为1：轨迹与频谱对照

日期：2026-10-10。四个相同真实样本已重新运行，源CFG=1、目标CFG=1。样本、完整提示词、SD3 Medium权重、512分辨率、时间表、种子0、n_max=33、n_min=0、n_avg=1保持不变，未添加优化修正。

## 查看结果

- [CFG=1全部50步及每步变化频谱](../experiments/flowedit_frequency_trajectory/outputs_cfg1/all_50_steps.png)
- [33个活动步](../experiments/flowedit_frequency_trajectory/outputs_cfg1/all_33_active_steps.png)
- 分段大图：[00–09](../experiments/flowedit_frequency_trajectory/outputs_cfg1/steps_00_09.png)、[10–19](../experiments/flowedit_frequency_trajectory/outputs_cfg1/steps_10_19.png)、[20–29](../experiments/flowedit_frequency_trajectory/outputs_cfg1/steps_20_29.png)、[30–39](../experiments/flowedit_frequency_trajectory/outputs_cfg1/steps_30_39.png)、[40–49](../experiments/flowedit_frequency_trajectory/outputs_cfg1/steps_40_49.png)
- [原CFG=3.5/13.5与CFG=1/1终点对比](../experiments/flowedit_frequency_trajectory/outputs_cfg1/cfg_final_comparison.png)
- [两组CFG的变化率和高频占比对照](../experiments/flowedit_frequency_trajectory/outputs_cfg1/cfg_frequency_comparison.png)
- [CFG=1频谱演变](../experiments/flowedit_frequency_trajectory/outputs_cfg1/frequency_evolution.png)、[步宽归一化](../experiments/flowedit_frequency_trajectory/outputs_cfg1/frequency_rate_evolution.png)

源文本、目标文本、条件激活条及CFG数值均已更新。旧结果保留在outputs，新结果独立保存在outputs_cfg1。前17步仍为idle，后33步编辑；没有将全部50步改成活动步骤。

## 主要观察

CFG=1表示使用条件预测但不额外放大：v_uncond+1*(v_cond-v_uncond)，不是关闭文本。当前四图的源姿态、外观、背景大体保留，但目标编辑明显减弱：熊仍偏棕色、猫头鹰仍偏棕色、面包车未明显变SUV、猫仍戴皇冠。不能将这种欠编辑称为编辑精度提升。

| 样本 | CFG1逐步变化能量之和／原CFG | 早期高频占比 | 中期高频占比 | 后期高频占比 |
|---|---:|---:|---:|---:|
| 熊 | 0.252% | 17.92% | 22.35% | 31.44% |
| 猫头鹰 | 0.236% | 8.63% | 11.02% | 14.58% |
| 车辆 | 1.759% | 20.32% | 26.51% | 35.79% |
| 猫 | 0.811% | 11.75% | 15.34% | 18.59% |

上述能量是相邻浮点RGB增量功率的逐步求和，**不是最终图像编辑能量、完成度或准确率**。高频比例是每阶段能量加权比例，四图仍以后期较高，但绝对变化非常小。两组谱使用相同源图参考功率，因此亮度可比较。

本次最大Parseval绝对误差约1.69e-21。与旧缓存终点的潜变量距离约0.306–0.378，是CFG改变后的结果差异，不是数值复现错误；记录中same_cfg_as_reference=false。

源和目标CFG同时改变，因此不能把差异归因于仅降低目标CFG，亦不能据此断言此前所有额外变化都是众数吸引。

## 复现

```bash
.venv/bin/python experiments/flowedit_frequency_trajectory/run.py --source-cfg 1 --target-cfg 1 --output-dir experiments/flowedit_frequency_trajectory/outputs_cfg1
.venv/bin/python experiments/flowedit_frequency_trajectory/export_spectra.py --output-dir experiments/flowedit_frequency_trajectory/outputs_cfg1
.venv/bin/python experiments/flowedit_frequency_trajectory/analyze.py --output-dir experiments/flowedit_frequency_trajectory/outputs_cfg1
.venv/bin/python experiments/flowedit_frequency_trajectory/render_conditions.py --output-dir experiments/flowedit_frequency_trajectory/outputs_cfg1
.venv/bin/python experiments/flowedit_frequency_trajectory/compare_cfg.py
```

图像、潜轨迹、每步功率数组、指标及协议均保留。没有修改原数据、模型权重或环境。
