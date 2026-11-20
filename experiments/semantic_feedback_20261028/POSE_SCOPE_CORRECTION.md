# 姿态许可对照修正

本轮检查渲染器的世界坐标和相机后发现：头部轴原指向 +x，相机位于 +x、+z；`rotation(+0.85)` 将 +x 转向 -z，实际是背离相机。因此原 `pose_only` 的图像不符合 `POSE_PROMPT` 中的“towards the camera”。

以下字段作废，不能作为姿态任务许可的正面或负面证据：

- `analysis_v1/heldout.json` 中 `task_selectivity.*.pose_expected_positive`。
- `analysis_v2/heldout.json` 中 `task_selectivity.*.pose_expected_positive`。

原形态任务的正确／错误标签、合法编辑接受率、错误转头拒绝率、形态完成度和颜色／形态任务对照不受影响：未请求的正向或负向转头都属于该任务的错误。原 `wrong_head` 应准确表述为背离相机的局部转头。

`challenge.py` 新增 `pose_front_only`、`wrong_head_front`、`wrong_heading_front`，使用 -0.85 弧度，分别检验真正朝向相机的合法姿态编辑和形态任务中的两种错误。保持所有旧数据和结果，以新产物单独报告；这些源复用第二轮场景，属于追加方向诊断，不能计为新的独立留出源。
