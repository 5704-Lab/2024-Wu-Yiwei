# SigLIP 干净状态目标响应：FlowEdit 方向校正与数学闭环

日期：2026-10-09。状态：首轮局部实验已运行，导数链路得到支持，代理辨识与有限步保证仍待验证。见[结果与理论边界](SigLIP干净状态响应_首轮数值与语义预检结果_2026-10-09.md)。

用户要求：利用 SigLIP 的干净图像文本匹配分数，对编辑状态 y 求导，定义目标响应；保留相邻方向校正和 q 同步。模型、VAE、文本条件和 SigLIP 权重均冻结，无训练、微调、反演或空间编辑掩码。

本文件替代速度文本 VJP 的当前主定义。已有本地 SigLIP2 Base 权重作为 SigLIP 家族的具体首轮底座，不将 SigLIP2 标作原版 SigLIP。配套[独立实验设计](SigLIP干净状态目标响应_分阶段验证实验设计_2026-10-09.md)。

## 1. 固定的 SigLIP 图文分数

令 x 为源图编码，y 为 FlowEdit 的干净编辑潜变量。VAE 实际解码：

$$
I(y)=\operatorname{clip}\left[
\frac{D(y/s_{\rm VAE}+b_{\rm VAE})+1}{2},0,1
\right].
$$

P 是固定可微预处理，f(y)=f_I(P(I(y))) 为 SigLIP 图像特征，e=f_T(c_tar) 为缓存目标文本特征：

$$
z(y)=\frac{f(y)}{\|f(y)\|},\qquad
t_c=\frac{e}{\|e\|}.
$$

实际 SigLIP logit：

$$
\ell(y)=\alpha z(y)^\top t_c+b,
\qquad
\alpha=\exp(\text{logit scale})>0.
$$

alpha、b 使用预训练权重，不重新训练或拟合。图文匹配概率及稳定分数：

$$
P_{\rm pair}(y)=\sigma(\ell(y)),
\qquad
S(y)=\log\sigma(\ell(y))
=-\operatorname{softplus}(-\ell(y)).
$$

采用单目标配对分数，不默认在一批候选文本上作 softmax；SigLIP 训练的 sigmoid 配对事件与候选集合类别概率不同。该概率是模型匹配信念，不是真实编辑成功率。

来源：[SigLIP，ICCV 2023](https://arxiv.org/abs/2303.15343)、[SigLIP 2](https://arxiv.org/html/2502.14786)。本地 transformers SiglipModel.forward 已核对为归一化特征点积乘 logit_scale.exp() 后加 logit_bias。

## 2. 目标响应改为编辑状态梯度

明确主定义：

$$
\boxed{r(y)=\nabla_yS(y).}
$$

不对文本求梯度，不对速度能量求梯度，不使用速度模型的辅助时间。噪声 q 只参与 FlowEdit 的基础更新。

令 g(y)=grad_y ell(y)，有：

$$
r(y)=\sigma(-\ell(y))g(y).
$$

而：

$$
g(y)=
\frac{\alpha}{\|f(y)\|}
D_yf(y)^\top(I-z(y)z(y)^\top)t_c.
$$

D_y f 包含 VAE 缩放、解码、裁剪、resize、归一化和图像编码器的全部链式导数。文本特征固定，文本编码器无需在每步反传。

当 r 非退化时，r 方向表示提高 SigLIP 目标配对分数的状态方向：

$$
S(y+\eta r)-S(y)=\eta\|r\|^2+O(\eta^2).
$$

这是一阶代理保证，不是自然语言事实或源一致性保证。

## 3. Sigmoid 饱和与方向计算

有限实数 logit 下 sigma(-ell)>0，所以：

$$
\frac{r(y)}{\|r(y)\|}
=\frac{g(y)}{\|g(y)\|}.
$$

直接计算概率梯度可能在浮点中饱和；方向模块使用 logit 梯度 g，目标匹配仍由实际 logit、log-sigmoid 和概率记录。

令 H=grad_y^2 ell，a=sigma(-ell)，p=sigma(ell)：

$$
\nabla_y^2S=aH-ap\,gg^\top.
$$

设 u=g/||g||，P_u=I-uu^T。因为 P_u g=0：

$$
\frac{P_u\nabla^2S}{\|r\|}
=\frac{P_uH}{\|g\|}.
$$

因此归一化响应方向的导数可以精确用 logit 的 Hessian 计算，sigmoid 的秩一曲率项被消去。不是将不同分数的非归一化梯度混用。

g 本身退化时仍须跳过方向控制；不能用 logit 导数绕过真正的平坦状态。高配对概率也不是准确任务完成，不能仅因其达到阈值就宣称成功。

## 4. FlowEdit 预测步与相邻参考

用 k 表示迭代，t_k 是下降的生成噪声时间，h_k=t_k-t_k+1>0：

$$
y_0=x,\qquad
q_{s,k}=(1-t_k)x+t_k\epsilon_k,
\qquad q_{a,k}=q_{s,k}+y_k-x.
$$

$$
\Delta v_k=v_{\rm tar}(q_{a,k},t_k)-v_s(q_{s,k},t_k),
\qquad
\bar y_{k+1}=y_k-h_k\Delta v_k.
$$

全部生成 CFG、时间表、n_avg、n_max 与主基线对齐，首轮限定 n_min=0。

在干净状态上：

$$
u_{\rm ref}=\frac{g(y_k)}{\|g(y_k)\|},
\qquad
u=\frac{g(\bar y_{k+1})}{\|g(\bar y_{k+1})\|}.
$$

无辅助时间、无输出探针、无控制用噪声。只要模型、文本和预处理固定，上一接受状态的梯度可以缓存复用。稀疏控制时参考仍是当前原始步前的 y_k，不能偷换成上次校正点。

定义余弦 F(0)=u_ref^T u。方向偏转是 SigLIP 匹配分数上升方向的变化，不是模型速度的变化。

## 5. 方向损失的状态回映

修正 w 后：

$$
U(w)=\frac{g(\bar y+w)}{\|g(\bar y+w)\|},
\qquad
F(w)=u_{\rm ref}^\top U(w).
$$

假设当前邻域内 ell 的 Hessian 存在，g 范数有正下界；需 Taylor 二阶余项时，方向映射导数局部 Lipschitz。

令 H=grad^2 ell(bar y)，则：

$$
A=D_wU(0)=\frac{P_uH}{\|g\|},
$$

$$
\boxed{
b_{\rm dir}=\nabla_wF(0)
=H^\top\frac{P_uu_{\rm ref}}{\|g\|}.
}
$$

光滑标量模型下 H 对称，故 H^T=H。实现使用 Hessian-vector product，不构造完整 Hessian。

也可直接对 F 反传到 y；必须保留 logit 梯度的图，并冻结 u_ref。和旧方案相比，这里是明确图文分数的状态 Hessian，不是速度对文本与状态的混合导数。

## 6. 不能只追随上一方向

沿 b_dir 运动可提高响应方向对齐，但可能降低目标 logit：

$$
D\ell[b_{\rm dir}]=g^\top b_{\rm dir}.
$$

该内积可能为负。即使 SigLIP 目标本身合理，合法编辑也可能要求上升方向转动，旧方向不是 oracle。

因此新主方案要求校正同时支持：

1. 目标匹配进展；
2. 非退化情况下的相邻方向对齐。

纯 SigLIP 一阶上升是必要基线，不能将其收益归因方向连续性。

## 7. 最小共同上升方向：闭式推导

设 g 与 b_dir 非退化，定义单位向量：

$$
u=\frac{g}{\|g\|},\qquad
v=\frac{b_{\rm dir}}{\|b_{\rm dir}\|},\qquad
c=u^\top v.
$$

在非反平行 c>-1 情形，考虑：

$$
\min_p\frac12\|p\|^2
\quad\text{s.t.}\quad
u^\top p\geq m,\quad v^\top p\geq m,\qquad m>0.
$$

KKT：

$$
p=\lambda_1u+\lambda_2v,\qquad
\lambda_i\geq0.
$$

对 -1<c<1，两个约束在最优点活跃，解：

$$
\lambda_1=\lambda_2=\frac{m}{1+c},
\qquad
p^*=\frac{m}{1+c}(u+v).
$$

Hessian 为 I，原问题严格凸，因此 p 唯一。c=1 时两个约束重复，唯一原解 p=m u，公式仍成立，乘子可不唯一。

单位共同方向：

$$
\boxed{
\hat p=\frac{u+v}{\sqrt{2+2c}}.
}
$$

$$
g^\top\hat p
=\|g\|\sqrt{\frac{1+c}{2}}>0,
$$

$$
b_{\rm dir}^\top\hat p
=\|b_{\rm dir}\|\sqrt{\frac{1+c}{2}}>0.
$$

因此非退化时，该方向使 logit 与响应余弦均一阶上升。严格反平行时，两正进展约束不可行；接近反平行时共同进展极弱，需要数值分辨检查。

带预算 ||p||<=R 的可行条件：

$$
m\sqrt{\frac{2}{1+c}}\leq R.
$$

共同方向可用指定范数 R 缩放；它是最小两项单位一阶进展的方向，不是原非线性优化问题的全局解。

## 8. 退化与优先级

状态机：

- 若原始 d=0，R=0，跳过修正。
- 若目标 logit 梯度不可辨认，跳过，不能将其自动解释为完成任务。
- 若方向偏转未超过阈值且基础 logit 未明显下降，保留原 FlowEdit 候选，无需方向 Hessian。
- 若只需改善基础候选的目标分数，使用纯语义方向 u。
- 若方向偏转需控制且 b_dir 可辨认、共同方向可靠，使用第 7 节共同方向。
- 若方向与目标进展近反平行，优先目标：退回纯语义方向，并记录无法同时校正。
- b_dir=0 时，可能已对齐、反对齐或方向不可控；不能宣称对齐成功。反对齐不强行求任意旋转。

纯语义回退是登记分支，不能在结果中算作方向约束成功。所有分支均需有限目标分数验收。

## 9. 位移预算与预测上界

绝对预算：

$$
R_{\rm abs}=\beta\|h_k\Delta v_k\|.
$$

它使额外控制速度相对原速度差有界。

SigLIP归一化特征满足理论 logit 上界 ell_max=alpha+b。余弦上界是1。共同方向下可附加预测余量预算：

$$
R_\ell=\zeta_\ell
\frac{\ell_{\max}-\ell(\bar y)}
{g^\top\hat p},
$$

$$
R_F=\zeta_F
\frac{1-F(0)}
{b_{\rm dir}^\top\hat p},
$$

$$
R=\min(R_{\rm abs},R_\ell,R_F).
$$

zeta 在(0,1)。仅在分母可辨认、上界余量非负时使用。纯语义分支不使用 R_F。

预测上界预算防止“预测余弦增量大于剩余提升空间”这样的明显外推，不是有限位移正确性保证；曲率仍可导致失败。

## 10. 有限位移验收及数学存在性

候选 w=s R hat p，s in (0,1]。记录：

$$
a_\ell=g^\top R\hat p>0,\qquad
a_F=b_{\rm dir}^\top R\hat p>0.
$$

若 ell 和 F 的梯度在邻域 Lipschitz：

$$
\ell(\bar y+w)-\ell(\bar y)
\geq s a_\ell-\frac{L_\ell}{2}s^2R^2,
$$

$$
F(w)-F(0)
\geq s a_F-\frac{L_F}{2}s^2R^2.
$$

因此精确算术下，足够小s能同时提高两项。Armijo：

$$
\Delta\ell\geq c_\ell s a_\ell+\tau_\ell,
\qquad
\Delta F\geq c_F s a_F+\tau_F.
$$

实际部署先生成正式存储精度的 y_new，再以 w_actual=y_new-bar y 做预算和 b^T w_actual 检查。tau覆盖实测误差；梯度范数及所有数值须有限。

检查两次候选s=1与1/2，两次失败则w=0。有限次数及正容差不保证一定找到下降位移，不能不断缩小到舍入误差通过。

纯语义分支只要求 logit Armijo，不声称 F 增加。共同分支两项均通过；若硬角度阈值仍未达到，只称软改进。

logit上升严格意味着 sigmoid 和 log-sigmoid 上升。记录实际S，但不因浮点概率饱和而拒绝所有可分辨 logit 进展。

## 11. 源内容一致性的范围

单靠SigLIP目标分数和方向连续性，不保证姿势、亮度、背景或身份不变。全图匹配存在对象绑定和奖励投机风险。

首轮主算法不假定已知非编辑区域。可单列无掩码源特征约束：

$$
L_{\rm keep}(y)=1-z(y)^\top z(x).
$$

在实际候选上要求 L_keep(y_new)<=L_keep(bar y)+epsilon_keep。它衡量整体特征接近，不是准确非编辑区域一致性，且可能抑制合法编辑；必须作为独立开关消融。

若该约束不成立，主算法仍可记录质量，不自动加源文本差、掩码或更强CFG补偿。独立评价与盲评必须与优化SigLIP分开。

## 12. 积分闭环

接受后：

$$
\boxed{
y_{k+1}=y_k-h_k\Delta v_k+w_k.
}
$$

$$
\boxed{
q_{a,k+1}
=(1-t_{k+1})x+t_{k+1}\epsilon_{k+1}
+y_{k+1}-x.
}
$$

$$
\Delta v_k^{\rm eff}=\Delta v_k-w_k/h_k.
$$

q 不独立积分，w不重复添加；下一步速度在正式校正查询上重新计算。SigLIP没有生成时间参数，校正不需要噪声端速度接口。

R_abs使 ||w/h||<=beta ||Delta v||，但会改变原FlowEdit控制场，不保留原终点分布的保证。若要求对原动力学保持积分一致性，可以令R_abs=beta_0 h ||d||，使w=O(h^2)；这是另一个缩放版本，不能同时期待大幅校正。

源/目标文本相同且原始位移为零时，本修正为零；普通FlowEdit不同CFG仍可能产生非零原始位移，不能宣称整体RGB恒等。VAE重建误差单列。

## 13. 可微预处理与平滑性

本地具体 SigLIP2 配置：256×256，image_mean=image_std=(0.5,0.5,0.5)，PIL resample=2，即 bilinear。

主控制使用浮点tensor链：VAE解码 -> [0,1]裁剪 -> bilinear resize（登记 antialias）->均值方差归一化 -> SigLIP。不对已在[0,1]的tensor再乘1/255；不转PIL/numpy、不detach图像、不对状态路径使用no_grad。

PIL图像处理与tensor插值可能存在量化/插值差，必须做前向一致性检查，不能假定逐位相同。

裁剪是分段光滑。上述Hessian和Taylor保证需要在所用邻域的活动裁剪集合稳定，或实际方向损失满足相应光滑条件；跨裁剪边界的候选由实际验收处理，不宣称处处C^2。二阶不支持或不可靠时只能保留纯一阶分支，并标明方向校正未实施。

## 14. 结论与原创性边界

闭环对象明确：干净y -> SigLIP匹配分数 -> 状态梯度 -> 相邻方向Hessian回映 -> 最小共同上升方向 ->有限双验收 ->y/q同步。

可证明的内容是局部代理进展、方向对齐的共同可实现条件和离散同步。真实编辑精准度、源保持、总成本和少步收益依然需要实验。

图文奖励引导、状态梯度、Hessian-vector product和多目标最小范数方向均有公开基础；本文件不确认原创性。方向模块必须与纯SigLIP梯度上升比较，不能把一般奖励引导收益称为新方向收益。
