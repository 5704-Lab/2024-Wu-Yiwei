# FlowEdit 与近期高保真反演编辑：同一 FLUX 对照（2026-11-05）

选用 **SelFix（2026）官方实现**作为近期 SOTA 反演候选，替代上一轮自行实现的普通 ODE／Heun 基线。使用相同真实熊图、完整源／目标提示、FLUX.1-dev 和原 VAE，1024×1024。不同主干的 SD3 旧结果不作为本轮方法比较。

依据：[SelFix 原文](https://arxiv.org/html/2606.17584v1)、[作者代码](https://github.com/SeminKim/selfix)。锁定 commit `c549b2f934c8cde7d3ff46a020dc05422059d435`。导入原始 `flux.sampling.denoise_selfix`、原始 `Flux`、原始单流层 Value 注入；不更改选根、动量或注入算法。仓库许可证保留于 `vendor/selfix/LICENSE`。

方法筛选同时核查了 [SlerpFlow（2026-07）](https://arxiv.org/html/2607.21326v1)、[DirectEdit（2026-05）](https://arxiv.org/html/2605.02417v2)、[SteerFlow（2026-04）](https://arxiv.org/html/2604.01715v1)等。选择 SelFix 的原因是公开原码、直接的 FLUX 重建／编辑证据以及可增加固定点预算的高保真路线。不同论文预算不同，不凭跨论文 PSNR 大小宣布唯一全局 SOTA。

## 实现与输入

- `prepare.py`：原始 1024 图，VAE 后验众数；现有本地 CLIP／T5 生成完整条件，原生 BF16。
- `common.py`：将本地 Diffusers FLUX 权重布局转换回官方原生 Flux 布局；全部 1160 个张量经官方转换函数往返逐元素相等检查，原生模型严格加载。未下载替代生成器或改动权重值。
- `run_comparison.py`：原码 SelFix，配套源重建，FlowEdit 速度差算法的原生 FLUX 调用。所有模型参数冻结。
- `evaluate.py`：源重建 PSNR／SSIM、固定源坐标背景窗口及对比图。窗口只用于离线评价，不进入推理，也不作为空间编辑掩码。

老 FlowEdit 包装器依赖早期 Diffusers 的图像尺寸 API；本轮在原生 FLUX 上实现同一速度差更新，直接使用正确的 4096 个图像 token，避免旧 API 对 1024 图像的尺寸换算不兼容。不声称已做该包装器的逐元素重放验证。

## 配置与计算量

| 配置 | 反演 | 目标生成 | 反演＋编辑 NFE | 作用 |
|---|---:|---:|---:|---|
| SelFix 作者默认编辑 | 15 步×4 次，μ=.1，δ=2 | 15 步，guidance=2 | 75 | 原推荐编辑配置 |
| SelFix 高保真预算 | 15 步×10 次，μ=.5，δ=1/8 | 15 步，guidance=2／3.5 分别保留 | 每个 165 | 把作者重建配置的反演预算用于编辑，明确为不同配置 |
| FlowEdit 作者推荐 | 无反演 | T=28，n_max=24，源／目标 guidance=1.5／5.5 | 48 | 原推荐主对照 |
| FlowEdit 引导强度对照 | 无反演 | 同上，源／目标 guidance=1／2 | 48 | 辅助排查引导强度影响 |

两种 SelFix 都是同一源反演噪声用于源重建和目标编辑；源重建使用 guidance=1、**不做特征注入**，不能用特征回填伪造反演质量。目标编辑保留作者设置：第一个反向步、single layers 0–37、Value replacement。没有源／目标区域掩码、外部正确目标、新 LLM 或新训练模块。

预先登记的较严格源重建筛选目标为：相对未编辑 BF16 潜变量的 VAE 解码图，PSNR≥28dB 且 SSIM≥0.90。该门槛只检查反演保真度，不认证完整目标形态或实际姿态语义。实际源图指标和目视结果同时报告，不能降低目标编辑程度来冒充保真度收益。

FLUX guidance 是蒸馏引导输入，不是 SD3 四分支 CFG；上述 NFE 为单图 Transformer 调用。完整初始协议计划 381 次，包括两条源重建、两种高保真目标 guidance 和 FlowEdit 引导强度对照；不伪装成同计算量比较。文本／VAE 前向另计。

## 产物与状态：已完成

- `outputs/inputs_v1/`：源图、VAE 图、完整条件与输入身份。
- `outputs/comparison_v1/protocol.json`：运行前固定的配置与门槛。
- `weight_layout_audit.json`：权重布局、逐元素往返与严格加载审计。
- `checkpoint.pt`：已完成阶段，便于中间检查。
- `results.pt`、`complete.json`：全部真实终点、轨迹和实际 NFE。
- `metrics.json`：最终独立测量；`preview_metrics.json` 仅是首个默认配置的中间检查。
- `comparison.png`：源图、FlowEdit 推荐、高保真 SelFix g=2 的主比较。
- `all_recipes.png`、`reconstruction.png`、`guidance_controls.png`：默认配置、对应重建和全部引导对照。

默认源反演 PSNR 26.773dB／SSIM .952727，未达到预设的 28dB；高保真源反演 **31.767dB／SSIM .969658，通过门槛**。上述参考为未编辑 packed BF16 潜变量的 VAE 图；相对原照片，高保真重建为 **27.647dB／SSIM .876926**，VAE 自身参考为 29.779dB／.907110。不得混用参考夸大原照片的保真度。

381 次真实 Transformer 调用全部完成，约 630.6 秒（含权重适配和解码），没有使用断掉的旧任务结果。源重建、编辑及所有配置均保存；不据单图声明普遍优越。输入协议的 `text_lengths.clip` 实为非零 token 计数，CLIP 的 EOS padding 非零；实际无截断长度和缓存一致性以 `outputs/text_input_audit.json` 为准，模型输入未因此改变。

脚本对已有输出目录拒绝覆盖；重新实验须使用新的版本目录，保留原始配置与失败。运行顺序为 `prepare.py` → `run_comparison.py` → `evaluate.py`，使用仓库 `.venv/bin/python`。长任务由独立后台进程执行，日志持续写入本目录，不因工具短调用结束而重启模型。

最终解释和方法选择证据见[本轮报告](../../docs/FlowEdit与SelFix高保真反演编辑_同FLUX真实图像对照_2026-11-05.md)。
