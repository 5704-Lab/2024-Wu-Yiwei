# FlowEdit 与普通 ODE 反演编辑：单张真实熊图对照

后续已按用户要求使用近期 SelFix 官方实现完成[同一 FLUX 的高保真反演对照](FlowEdit与SelFix高保真反演编辑_同FLUX真实图像对照_2026-11-05.md)，高保真源重建通过预设门槛。本篇的普通 ODE／Heun 失败保留作数值诊断，不能代表当前强反演编辑方法。

日期：2026-11-05。状态：**已执行真实 SD3 推理，已保存原图、编辑结果、重建控制、轨迹与计算记录。** 本次只有一个真实源图，是示例对照，不是一般性基准评测。

## 1. 主对照与直观结果

![源图、FlowEdit 与普通部分 ODE 反演编辑](../experiments/inversion_comparison_20261105/outputs/bear_v1/comparison_matched.png)

从左至右：真实源图、FlowEdit、普通部分 ODE 反演后切换目标提示生成。任务为将棕熊编辑为黑熊，源／目标描述其余文字保持一致。

本例人工视觉检查显示：

- **FlowEdit**：生成了黑色熊外观，溪流和草地的总体布局保留较多；熊头由侧向转为更朝向镜头，身体轮廓和前肢也发生变化。因此不能把这一例视为“只改必要内容”的成功。
- **普通 Euler 部分反演**：同样生成黑色熊，但出现大面积红黑字形、边框等未要求内容；头向、身体和场景纹理也发生变化。
- 图像差异支持“此配置下 FlowEdit 的场景保持较好”，**不证明两者均完成了用户要求的完整物种／形态编辑**。未使用此前已经失败的形态 SigLIP 分数为本例认证成功，也未把人工视觉描述冒充定量姿态测量。

## 2. 设置与公平性

使用原 **SD3 Medium**、原 VAE、冻结缓存的完整文本条件，FP32、512×512。源图来自原 FlowEdit 数据集，文件 `data/flowedit/Images/bear.png`。没有额外训练、反演优化、注意力注入、源特征替换或空间更新掩码。

源提示：

> A large brown bear walking through a stream of water. The stream appears to be shallow, as the bear is able to walk through it without any difficulty.

目标提示：

> A large black bear walking through a stream of water. The stream appears to be shallow, as the black bear is able to walk through it without any difficulty.

网格为原 shift=3.0 的 50 点 Euler 时间网格；所有方法源 CFG=3.5，目标 CFG=13.5，空负提示。FlowEdit 的 `n_max=33, n_min=0, n_avg=1`，CPU 噪声种子 0。

| 方法 | 图像到中间状态 | 中间状态到编辑结果 | 起始噪声时刻 | 反演＋编辑的单分支次数 |
|---|---|---|---|---:|
| FlowEdit | 无 ODE 反演；每步从固定源图直接加新噪声 | 33 步源／目标速度差更新 | 0.85014296 | 132 |
| 普通 Euler 部分反演 | 源提示正向 Euler 33 步 | 目标提示反向 Euler 33 步 | 0.85014296 | 132 |
| 普通 Euler 完整反演 | 源提示正向 Euler 50 步 | 目标提示反向 Euler 50 步 | 1.0 | 200 |
| Heun 部分反演，数值诊断 | 源提示正向 Heun 33 区间 | 目标提示反向 Heun 33 区间 | 0.85014296 | 264 |

主比较前两项具有相同的时刻范围、源／目标 CFG、Euler 网格及理论单分支次数。分批方式不同，不据此声称 wall time 相等。完整反演和 Heun 属于额外诊断，不拿它们与 FlowEdit 冒充等计算量比较。

这里的反演指普通条件 ODE 反演，不是专门的 RF-Inversion 方法。该基本定义及 SD3 的 33 步、3.5／13.5 设置对应 [FlowEdit 原文 §3.2 与附录 B.2 表 S1](https://arxiv.org/html/2412.08629v1)。本次自行实现普通反演，不声称逐行复现作者未提供的反演求解代码。

## 3. 实现上的区别

普通反演先以源提示速度场从实际源潜变量积分到中间噪声状态，再以目标提示速度场反向积分。显式 Euler 正向在当前左端点评价速度；反向同样在当前状态评价速度，步宽为负。

FlowEdit 保留实际干净编辑变量 y。每步以固定源图和新噪声构造源查询，再把当前编辑位移加到源查询上得到目标查询，用两者速度差更新 y。它没有先求一条源图到噪声的数值逆轨迹。

这解释了两者受到的误差来源不同：普通反演包含源图往返的数值重建误差，也包含切换文本和 guidance 后的生成变化；FlowEdit 避开了这条逆轨迹，但目标速度差仍可带来未请求的姿态／内容变化。不能由免反演推出姿态保持。

## 4. 必须与编辑结果一起看的重建控制

![源图、VAE、Euler 和 Heun 的源提示重建](../experiments/inversion_comparison_20261105/outputs/bear_v1/reconstruction_with_solver.png)

控制中不切换文本，正／反方向都用源提示和 CFG=3.5。这样检验的是当前数值反演能否把输入找回来。

| 结果 | 相对 VAE 源重建的 RGB RMSE | 潜变量位移 RMS |
|---|---:|---:|
| FlowEdit 编辑 | 0.154764 | 0.359829 |
| Euler 部分反演编辑 | 0.214510 | 0.506285 |
| Euler 部分反演源重建 | 0.105639 | 0.259639 |
| Heun 部分反演编辑 | 0.260200 | 0.617590 |
| Heun 部分反演源重建 | 0.206370 | 0.492841 |
| Euler 完整反演编辑 | 0.461664 | 1.437214 |
| Euler 完整反演源重建 | 0.545471 | 1.637366 |

RGB 输入范围为 [0,1]，参考是 D(E(I_source))，以扣除共同 VAE 重建影响。**编辑结果的源差不是编辑质量指标**：合法颜色、外观和形态变化也会增大该值；不能把表中的大小直接解释成不必要编辑多少。

Euler 部分源重建已经有纹理、形态和水流改变。完整反演源重建严重失真，编辑结果成为黑白图形；这项不能用于单独证明文本语义耦合或反演范式的固有缺陷。

发现这点后，补做同一区间、两方向均使用 Heun 的诊断。结果没有改善，源重建出现原图不存在的房屋场景，RMSE 更高。说明**仅将这套 33 区间 Euler 换成显式 Heun，没有获得可信的高精度反演**；不能因为形式上二阶就称它更精确，也未通过步宽收敛检验证明该 ODE 已被精确求解。

因此，当前可确认的结论限定为：这张图上，给定普通反演配置存在明显重建与编辑伪影；FlowEdit 的视觉场景保持较好，但也有未请求转头。**尚不能从本例把反演编辑的全部附带变化归因于模型语义耦合，更不能推广为所有反演方法较差。**

完整反演失败图与未隐藏的初始结果见[四列原始对照](../experiments/inversion_comparison_20261105/outputs/bear_v1/comparison.png)；同范围求解器比较见[FlowEdit／Euler／Heun](../experiments/inversion_comparison_20261105/outputs/bear_v1/comparison_with_solver.png)；局部细节见[固定源坐标裁剪](../experiments/inversion_comparison_20261105/outputs/bear_v1/head_detail_with_solver.png)。裁剪仅用于显示，不参与模型更新，未对齐以掩盖位置变化。

## 5. 数值核查、执行记录与产物

- 新执行的 FlowEdit 最终潜变量与此前保存种子 0 基线的相对误差为 **0.0**，33 个噪声哈希完全一致。
- 重新编码源图与缓存源潜变量的相对误差为 **4.09×10⁻⁷**。所有保存结果有限，模型权重冻结。
- 初始任务在完整反演的反向阶段收到 SIGTERM，退出码 143。复用检查点中的 FlowEdit、完整正向反演和部分反演结果，仅重算完整反向生成；初始与恢复脚本身份分别保留，没有将中断算成功。
- 全部保留结果对应 960 个单分支计算，包含 298 次不同批量的 Transformer 调用；中断已记录的额外 184 个单分支另计，可能还有未记录完的在途计算。未据运行时做方法加速结论。
- 脚本、提示、状态、网格、计算量和哈希见[实验目录](../experiments/inversion_comparison_20261105/README.md)、[基础完成记录](../experiments/inversion_comparison_20261105/outputs/bear_v1/complete.json)及[Heun 诊断记录](../experiments/inversion_comparison_20261105/outputs/heun_partial_v1/complete.json)。

本例与当前语义约束研究的直接联系是：**FlowEdit 的未请求转头在没有数值反演过程时仍存在**。后续语义响应实验仍有必要，但普通反演不能在源重建未通过时充当“只有文本效应”的干净参照。
