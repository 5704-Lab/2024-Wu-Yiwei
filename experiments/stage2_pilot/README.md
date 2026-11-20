# 四图机制先导

[协议](protocol.json)在生成前冻结。四个机制源图组、固定首个目标、种子 0、中噪声点 33；五种方向各做正/负同幅潜变量干预，之后冻结噪声、关闭校正并继续原更新式。

整个干预检查点及局部预测 FP64，共同候选和干预提交为 FP32，后续统一 FP32 单样本数学 SDPA。独立 SigLIP2 仅用于评价，不参与生成。另保存相同噪声的官方 batch=4 基线，区分数值布局和局部干预的影响。

```bash
.venv/bin/python scripts/download_siglip2.py
.venv/bin/python experiments/stage2_pilot/prepare_inputs.py --device cuda:1
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/stage2_pilot/run.py --device cuda:0 --cases 0,1
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/stage2_pilot/run.py --device cuda:1 --cases 2,3
.venv/bin/python experiments/stage2_pilot/evaluate.py --device cpu
```

图像、终点潜变量、数值审查、局部预测、噪声哈希和评价全部位于 outputs/。每图 overview.png 包含源图、VAE 重构、共同基线、官方基线和全部正/负干预。保存日志和模型/输入/源码哈希，不因质量删例。

这是机制可行性先导，不能作为完整阶段 2 通过，也不能把图文余弦当作准确语义成功率。没有掩码评价或准确非编辑区域一致性，没有少步或速度结论。等幅正负 QP 方向是干预探针，不等同执行原始约束 QP 控制器；负干预不承诺其约束。
