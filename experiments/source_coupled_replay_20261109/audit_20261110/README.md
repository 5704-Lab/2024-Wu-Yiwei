# 2026-11-10 数学与假设审查

只读审查v1理论、参考算子与计划，不执行图像模型，不覆盖原文件。

- [审查报告](../../../docs/源速度局部提升与配对回放_数学与问题假设审查_2026-11-10.md)
- `review_checks.py` → [11项审查例](review_checks.json)：包含原FlowEdit与反演映射不同、逆条件数放大、Heun折叠、源门的有限含义和残差修正损害保护因素的反例。
- `revise_plan.py` → [protocol_v2.json](protocol_v2.json)、[plan_v2.json](plan_v2.json)、[revision_summary.json](revision_summary.json)。

```bash
.venv/bin/python experiments/source_coupled_replay_20261109/audit_20261110/review_checks.py
.venv/bin/python experiments/source_coupled_replay_20261109/audit_20261110/revise_plan.py
```

v2仅修复开发门槛缺作业，并补主方法选中输入的源／目标映射数值稳定性检查：新增4个主方法开发作业和4个映射检查，合计432个计划作业、48348次条件性调用上限。原v1保留。稳定性检查不等于全域可逆性或语义证明，真实模型实验尚未执行。

审查结论：多数公式在其前提下正确；该方法更接近近似部分反演与源残差锚定，对原FlowEdit随机平均及保持假设只部分对齐。原37项算子核对不能替代这些新审查。
