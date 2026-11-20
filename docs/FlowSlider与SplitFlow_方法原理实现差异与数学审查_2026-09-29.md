# FlowSlider 与 SplitFlow：方法原理、实现差异与数学审查

日期：2026-09-29。依据：[FlowSlider 原文，2026-04 arXiv 预印本](https://arxiv.org/html/2604.02088)、[SplitFlow 原文，NeurIPS 2025](https://arxiv.org/html/2510.25970)、[SplitFlow NeurIPS 会议页](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e124f1547f7ac87e33d348b827d4291b-Abstract-Conference.html)与[官方公开代码](https://github.com/Harvard-AI-and-Robotics-Lab/SplitFlow/blob/main/SplitFlow_utils.py)。本文区分**作者主张**、**公式直接保证的内容**和**本文推论/反例**。

## 1. 共同的 FlowEdit 骨架

统一记源图潜变量为 $x$、源与目标文本为 $c_s,c_a$，干净数据时间 $t=0$、噪声时间 $t=1$。以同一噪声 $\epsilon_t$ 构造源桥 $q_s=(1-t)x+t\epsilon_t$，以当前编辑变量 $z_t$ 构造目标评估状态 $q_a=z_t+q_s-x$。FlowEdit 的向噪声时间速度差是

$$
\Delta v_{\mathrm{FE}}=
v(q_a,t,c_a)-v(q_s,t,c_s),\qquad
z_{t-\Delta}=z_t-\Delta\,\Delta v_{\mathrm{FE}},\quad\Delta>0.
$$

实际模型还可有不同的源/目标 CFG 强度；后面的简式先假设二者相同。该速度差的两个模型评估处于**不同位置**，因此混合了文本差与状态差。两篇论文都沿用无逐图反演的 FlowEdit 框架，但处理的问题不同。[FlowEdit 原文](https://arxiv.org/html/2412.08629)。

## 2. FlowSlider：分解完整速度差，只缩放同点文本项

### 2.1 公式与单步流程

在目标评估位置 $q_a$ 插入并减去 $v(q_a,t,c_s)$，得到严格的恒等式

$$
\Delta v_{\mathrm{FE}}=
\underbrace{v(q_a,t,c_a)-v(q_a,t,c_s)}_{V_{\mathrm{steer}}}
+
\underbrace{v(q_a,t,c_s)-v(q_s,t,c_s)}_{V_{\mathrm{fid}}}.
$$

前一项是**同点、不同提示词**的速度差；后一项是**同提示词、不同位置**的速度差。FlowSlider 的编辑器采用

$$
\Delta v_{\mathrm{FS}}(s)=V_{\mathrm{fid}}+sV_{\mathrm{steer}},
\qquad s=1\Longrightarrow\Delta v_{\mathrm{FS}}=\Delta v_{\mathrm{FE}}.
$$

每步求出 $v(q_s,c_s)$、$v(q_a,c_s)$、$v(q_a,c_a)$，按上式更新。多出的同点源条件速度是相对 FlowEdit 的第三个**引导速度评估**。显式 CFG 的每个“引导速度”还可对应有条件/无条件原始分支；主干调用与批处理成本应另计。方法无训练、无微调、无反演和显式掩码。[FlowSlider §3.3 与算法 1](https://arxiv.org/html/2604.02088)。

作者的动机是：直接用 $s\Delta v_{\mathrm{FE}}$ 会同时放大文本推动与源条件状态差；只改变 $sV_{\mathrm{steer}}$ 时，$V_{\mathrm{fid}}$ 的**显式系数**保持为 1。论文把后者称作保真项，并报告二者夹角通常接近 $90^\circ$。这两个名称是作用的经验解释；恒等式本身不能证明它们分别等于真实语义和非编辑区域一致性。

### 2.2 “近似正交”的数学边界

令 $F=V_{\mathrm{fid}},\,S=V_{\mathrm{steer}}$。瞬时速度长度满足

$$
\|F+sS\|^2=\|F\|^2+s^2\|S\|^2+2s\langle F,S\rangle.
$$

如果当下 $\langle F,S\rangle\approx0$，增大 $s$ 不会**直接**大幅改变该时刻沿 $F$ 的速度投影。论文用夹角经验分布支持“弱耦合”。但它不是终点保真的充分条件。反例：在反向时间 $\tau$ 的二维系统中取 $S=(1,0)$、$F(x,y)=(0,x)$，二者每点严格正交；从 $(0,0)$ 积分 $(x',y')=F+sS$ 至时间 $T$，却得 $x(T)=sT,\ y(T)=sT^2/2$。增强 $S$ 通过后续状态依赖改变了 $F$ 控制的坐标。故论文的几何式子只约束**瞬时向量**，源图最终保留率仍需实测。$F$ 的逆时间状态 Jacobian 也未必收缩。[FlowSlider 夹角分析与失败案例](https://arxiv.org/html/2604.02088)。

### 2.3 论文报告与失效条件

论文在 FLUX.1-dev 与 SD3 Medium 上使用 28 步、编辑窗口 $n_{\max}=20$、源与目标共同 CFG 3.5，并以 $s\in\{1,2,3,4,5\}$ 比较强度序列；连续编辑基准有 250 个样本，其中 31 个来自 PIE-Bench。指标包括编辑方向 CLIP-dir、源图距离 DreamSim、强度单调性和平滑性。结果支持**滑杆控制**，不能直接外推到所有离散对象替换或少步编辑。[实验设置与表 1、表 3](https://arxiv.org/html/2604.02088)。

论文自己列出两类失败：强度过大时结构塌陷/过饱和；兔子变猫等离散概念在 $s=1$ 已完成后继续放大，只产生颜色和锐化伪影。这说明全同点文本项的强度增益可饱和，但不证明某个正交余量会改善它。[FlowSlider 附录失败案例](https://arxiv.org/html/2604.02088)。

## 3. SplitFlow：先拆复杂目标，再聚合多条轨迹

### 3.1 子提示词与独立 FlowEdit 路径

SplitFlow 面向“牧羊犬同时戴墨镜、跳跃、张嘴”一类复合目标。它让冻结的 Mistral-7B 根据源/目标描述生成通常不超过 3 个子目标提示词；每个 $c_i$ 独立跑一条 FlowEdit 式路径：

$$
\Delta v_i=v(q_i,t,c_i)-v(q_{s,i},t,c_s).
$$

这里的“分解”是**文本任务分解并行化**，不是对同一条速度向量做 Jacobian 或切/法向分解。随后在短的早期窗口聚合，并沿完整目标提示词的统一轨迹继续编辑。论文采用 SD3/SD3.5、$T=50$、$n_{\max}=33$、聚合点 $\eta_{\mathrm{dec}}=28$；有效编辑前约 5 步同时运行多分支。[SplitFlow §4.1、§5.1](https://arxiv.org/html/2510.25970)。

它保持零训练、零微调与无反演，但引入**额外冻结 LLM**和多个早期模型分支。论文的“梯度冲突”是条件速度与多任务梯度的类比；没有真实目标成功率或像素保留率梯度。

### 3.2 LTP：潜变量投向完整目标分支

设完整目标分支潜变量 $z_{\mathrm{full}}$、子分支潜变量 $z_i$。在每个潜变量格点的通道向量上，Latent Trajectory Projection（LTP）为

$$
\hat z_{\mathrm{full}}=\frac{z_{\mathrm{full}}}{\|z_{\mathrm{full}}\|},\qquad
\tilde z_i=\langle z_i,\hat z_{\mathrm{full}}\rangle\hat z_{\mathrm{full}},
\qquad
z_{\mathrm{proj}}=\frac1N\sum_i\tilde z_i.
$$

它把子轨迹投到完整目标轨迹张成的**一维通道方向**，再平均。这里的正交投影是线性代数操作，不是数据概率流形或真实非编辑区域的投影。与完整目标方向正交的子流成分会被删除；当完整目标向量接近零时方向不可靠，代码用极小常数保护归一化。作者的消融显示，LTP 比简单平均有更高的目标 CLIP 相似度，但背景保留略逊于简单平均。[SplitFlow LTP 公式及消融](https://arxiv.org/html/2510.25970)。

### 3.3 VFA：基于余弦相似的共识加权

Velocity Field Aggregation（VFA）在潜变量空间每个格点将子流速度按通道归一化，计算 $S_{ij}=\langle\hat g_i,\hat g_j\rangle$，再按与其他方向的相似度总和作 softmax：

$$
a_i=\sum_{j\ne i}S_{ij},\qquad
w_i=\frac{\exp(a_i)}{\sum_k\exp(a_k)},\qquad
\bar g=\sum_iw_i g_i.
$$

**与多数分支相似的方向权重较大**，所以这是共识加权；与其他两个相反的独有方向权重更小。它没有把冲突方向真的投成正交向量。特别地，$N=2$ 时 $a_1=a_2=S_{12}$，恒有 $w_1=w_2=1/2$，VFA 退化为普通平均。论文“强调不同方向”的表述须与这个实际规则区别。[SplitFlow VFA 公式](https://arxiv.org/html/2510.25970)。

### 3.4 论文给 VFA 的不等式证明存在缺口

论文声称 $\langle\bar g,g_{\mathrm{avg}}\rangle\ge\|g_{\mathrm{avg}}\|^2$，其中 $g_{\mathrm{avg}}=\frac1N\sum_i g_i$。证明令 $a_i=\sum_j\langle\hat g_i,\hat g_j\rangle$，随后把 $\langle g_i,g_{\mathrm{avg}}\rangle$ 当成 $a_i/N$。**若速度长度不同，这一步不成立：权重由单位化余弦计算，聚合却使用未单位化的 $g_i$。**

可复算反例：单格点的三个一维速度 $g_1=g_2=1,\ g_3=-3$。此时 $a_1=a_2=0,\ a_3=-2$，故 $w_1=w_2=1/(2+e^{-2})\approx0.46831,\ w_3\approx0.06338$；于是 $g_{\mathrm{avg}}=-1/3,\ \bar g\approx0.74648$，从而

$$
\langle\bar g,g_{\mathrm{avg}}\rangle\approx-0.24883
<0.11111=\|g_{\mathrm{avg}}\|^2.
$$

所以公开不等式对**原算法的一般速度场不成立**；即使在额外单位范数假设下成立，它也只说明与平均向量对齐，不能推出语义正确率或背景保留。此审查不否定论文报告的经验改善。[SplitFlow 主文证明及附录 C](https://arxiv.org/html/2510.25970)。

### 3.5 公开代码与成本口径

公开代码在每个子提示词循环内单独抽噪：各子流的源/目标状态内部共享该分支噪声，**跨子流噪声不同**。在 LTP 后下一步，代码以共同聚合编辑状态计算各提示词速度并做 VFA；这比论文“在各自投影后的子状态计算速度”的抽象公式更具体，也使跨分支余弦混入噪声差异。[官方代码 169–219 行](https://github.com/Harvard-AI-and-Robotics-Lab/SplitFlow/blob/main/SplitFlow_utils.py#L169-L219)。

论文正文按 3 个子目标估算，有效分支等效步骤从 33 增至约 $33+3(33-28)=48$；文字报告 700 张图的 FlowEdit 约 57 分钟、SplitFlow 约 83 分钟，另计 LLM 分解约 20 分钟。**附录成本表的 FlowEdit/Ours 行与文字相反**，因此无法仅据该表确定哪种方法对应哪一计时；更可靠的是额外模型分支和额外 LLM 的存在，实际耗时需重测。[SplitFlow §5.3 与附录 E](https://arxiv.org/html/2510.25970)。

PIE-Bench 上 SD3 默认设置，论文报告结构距离从 FlowEdit 的 27.24 降至 25.96，编辑区域 CLIP 相似度从 23.67 升至 23.83：属于**小幅平均改善**。带上标 † 的强保真变体还改变了分解阶段的源状态构造，不能把其明显更好的数值直接归因于默认 LTP+VFA。数据集掩码可作事后评估，不是 SplitFlow 推理输入；实际部署仍无法获取真实编辑区域真值。[SplitFlow 表 1、表 2 与强保真变体说明](https://arxiv.org/html/2510.25970)。

\n## 4. 与信号系统分析的关系

FlowSlider 对输入条件变化与状态反馈作有限差分拆分，SplitFlow 对多条件通道作聚合；两者都不等于文本条件自动微分。分析它们的控制作用时，应分别检查输入增益、状态反馈、跨通道耦合与完整推理成本。保留某个速度分量不能直接证明源内容不变。
