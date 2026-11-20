# 当前干净状态联合反馈约束实验（2026-10-28）

对应[系统调研](../../docs/干净状态源图文本联合反馈约束_主流工作与跨学科理论系统调研_2026-10-28.md)。目标是检验源图、文本、当前干净状态联合观测能否允许合法形态、识别附带编辑并驱动修复。结果见[实验报告](../../docs/当前干净状态联合语义约束_候选观测与反馈修复实验_2026-10-28.md)。

所有推理仅使用原 SD3 Medium、其 VAE 和项目已有 SigLIP2；权重冻结，无额外 LLM、外部参考、反演或空间编辑掩码。`ORACLE` 是明确分开的离线参考辅助子空间，不是部署方案。受控图像中的因素只进入离线评价。

## 数据与信息边界

- `dataset_v1`、`dataset_v2` 各 16 个源、192 张图，两个种子区间不重叠。每组前 4 源用于开发、接着 4 源用于阈值校准、后 8 源留出。
- `dataset_v3` 为语义因素文字探测额外生成 8 个新源、96 张图；全部作为新留出，不按其继承的开发／校准字段分组。
- `dataset_v4` 为相机两侧的透视混杂诊断生成 8 个新源、96 张图；全部使用第二轮锁定阈值评价，无新调参。
- `dataset_challenge` 复用第二组源，新增 48 张朝向相机的图；不是新的独立源。必须阅读[姿态对照修正](POSE_SCOPE_CORRECTION.md)，原两轮的 `pose_expected_positive` 字段无效。
- 上述为参数可控的三维玩具形态／外观任务，不能外推为真实物种替换已验证。
- `natural_v1` 为 4 张真实源图的 42 个干净状态，包含保留轨迹与源图扰动；没有人为给这些编辑终点添加“正确类别形态”的真值。

## 主要入口

| 文件 | 功能 |
|---|---|
| `dataset.py` | 解析椭球射线渲染，分别控制形态、头部关节、整体朝向、颜色、另一对象及背景 |
| `encode.py` | 原有 SD3 文本编码器 FP32 缓存 |
| `extract.py` | 对实际干净潜变量标准化重加噪，提取多层 patch 特征、关系与条件证据 |
| `analyze.py` | 第一轮开发选择、校准和留出评价；含单独命名的参考辅助诊断 |
| `induced.py`、`analyze_induced.py` | 源文本速度实际作用于干净潜变量后，读取观测效应并测试消元 |
| `repair.py` | 当前残差的短程 GN 修复、任务半空间、有限解码验收和独立终点比较 |
| `typed.py` | 仅用已有 SigLIP2 的固定语义因素文字探测；无新增训练，有限词表不是自由文本解析器 |
| `challenge.py`、`analyze_challenge.py` | 修正姿态许可对照并检查两种旋转方向 |
| `phase_audit.py`、`analyze_balanced.py` | 解码图像的相位／复响应基线，以及对象面积／视角混杂诊断 |
| `full.py` | 真实自然图完整 FlowEdit，每步约束实际干净状态并重新构造下一步查询 |
| `scaled.py` | 使用对应反馈轨迹的步长比率调度作无语义反馈对照 |
| `summarize.py` | 汇总数字、审计与独立图表 |

## 执行顺序

使用仓库 `.venv/bin/python`。主要推理使用 GPU 1；其余 GPU 空闲后用于独立补充验证。命令中的设备设置是物理 GPU 映射，脚本内部使用 `cuda:0`。

```bash
.venv/bin/python experiments/semantic_feedback_20261028/dataset.py
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/semantic_feedback_20261028/encode.py
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/semantic_feedback_20261028/extract.py
.venv/bin/python experiments/semantic_feedback_20261028/analyze.py --stage dev
.venv/bin/python experiments/semantic_feedback_20261028/analyze.py --stage holdout

.venv/bin/python experiments/semantic_feedback_20261028/dataset.py --tag dataset_v2 --seed-base 282031
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/semantic_feedback_20261028/extract.py --dataset dataset_v2 --tag features_v2
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/semantic_feedback_20261028/induced.py
.venv/bin/python experiments/semantic_feedback_20261028/analyze_induced.py --stage dev
.venv/bin/python experiments/semantic_feedback_20261028/analyze_induced.py --stage holdout
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/semantic_feedback_20261028/repair.py --tag repair_gn_v2

.venv/bin/python experiments/semantic_feedback_20261028/typed.py --stage cache
.venv/bin/python experiments/semantic_feedback_20261028/typed.py --stage calibrate
.venv/bin/python experiments/semantic_feedback_20261028/dataset.py --tag dataset_v3 --seed-base 283031 --count 8
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/semantic_feedback_20261028/typed.py --stage fresh

.venv/bin/python experiments/semantic_feedback_20261028/challenge.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/semantic_feedback_20261028/extract.py --dataset dataset_challenge --tag features_challenge
.venv/bin/python experiments/semantic_feedback_20261028/analyze_challenge.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/semantic_feedback_20261028/phase_audit.py
.venv/bin/python experiments/semantic_feedback_20261028/dataset.py --tag dataset_v4 --seed-base 284031 --count 8 --balanced-camera
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/semantic_feedback_20261028/extract.py --dataset dataset_v4 --tag features_v4
CUDA_VISIBLE_DEVICES=0 .venv/bin/python experiments/semantic_feedback_20261028/analyze_balanced.py
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/semantic_feedback_20261028/natural.py
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/semantic_feedback_20261028/full.py
CUDA_VISIBLE_DEVICES=1 .venv/bin/python experiments/semantic_feedback_20261028/scaled.py
.venv/bin/python experiments/semantic_feedback_20261028/summarize.py
```

已有产物按不可覆盖协议处理；复跑应使用新目录或独立工作副本。各阶段保存了代码哈希、阈值、逐状态结果与关键脚本快照。`dataset_v1` 和 `features_v1` 的快照对应当时版本；后续增加 CLI 和相机因素未改变已保存数据。

## 结果入口及解释限制

- `outputs/analysis_v1/heldout.json`：第一轮代表观测的留出结果；姿态许可字段按修正说明作废。
- `outputs/analysis_v2/heldout.json`：新源上的原始／文本消元／速度诱导／属性分解消元比较。
- `outputs/repair_gn_v2/summary.json`：36 条短程反馈轨迹。接收的模型残差下降不是编辑质量成功标签。
- `outputs/typed_v1/fresh.json`：预设语义词表的新源检验。
- `outputs/challenge_analysis.json`：正确旋转方向的姿态任务及错误转头诊断。
- `outputs/balanced_analysis.json`：新视角测试，分别报告合法接受与错误拒绝，防止仅凭拒绝率误判。
- `outputs/full_bear_v2/`：原始基线与两种自动约束的完整自然图轨迹。`full_bear_v1` 是仅数学注意力后端导致基线不完全一致的已停止适配尝试，不能算作新方法结果。

评价以源为独立单位；同源多个状态不当作独立样本。开发、适配和补充诊断不能混合汇报成一次独立留出成功。原始相位／复响应没有自由文本许可，因此其拒绝率也不等于语义约束能力。
