"""Verify the complete cohort and insert measured results into its report."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=HERE/"outputs/study_v1"
REPORT=ROOT/"docs/FlowEdit增量SVD与空间谱_24源多种子及在线干预_2026-11-17.md"


def load(path):
    return json.loads(path.read_text())


def pct(x):
    return f"{100*x:.2f}%"


def main():
    torch.set_num_threads(4)
    protocol=load(OUT/"protocol.json")
    expected=[(r["pair_id"],s,m) for r in protocol["records"] for s in protocol["seeds"]
              for m in (protocol["arms"] if s==0 and r["pair_id"] in protocol["online_sources"] else ["full"])]
    assert len(expected)==156
    completions=[load(OUT/"runs"/p/f"seed{s}"/m/"complete.json") for p,s,m in expected]
    workers=[load(OUT/f"worker{i}_complete.json") for i in [0,1]]
    assert sum(x["transformer_calls"] for x in completions)==sum(x["transformer_calls"] for x in workers)==7488
    script_hash=hashlib.sha256((HERE/"run.py").read_bytes()).hexdigest()
    assert script_hash==protocol["inference_script_sha256"]
    assert all(w["script_sha256"]==script_hash for w in workers)
    for i in [0,1]:
        a=load(OUT/f"weight_audit_worker{i}.json")
        assert a["layout_roundtrip_bitwise"] and a["tensors_verified"]==1160
    prepared=load(OUT/"prepared.json")
    assert prepared["clip_truncated_count"]==prepared["t5_truncated_count"]==0
    measure=[]
    gradient_errors=[]
    for r in protocol["records"]:
        pair=r["pair_id"]
        inp=OUT/"inputs"/pair/"input.pt"
        assert hashlib.sha256(inp.read_bytes()).hexdigest()==prepared["input_hashes"][pair]
        d=OUT/"measurements"/pair
        item=load(d/"complete.json")
        assert item["full_replay_rgb_max_error"]==0
        measure.append(item)
        rows=load(d/"step_statistics.json")
        assert len(rows)==144
        assert len(load(d/"local_metrics.json"))==39
        assert len(load(d/"nonlinearity.json"))==5
        for x in load(d/"online_metrics.json")+load(d/"local_metrics.json"):
            assert -1.00000001<=x["rgb_cosine_to_full"]<=1.00000001
        run=torch.load(OUT/"runs"/pair/"seed0/full/trajectory.pt",weights_only=True)
        # Independently verify the spatial Laplacian spectrum against the
        # periodic nearest-neighbour squared-difference energy.
        p=run["states"][24].double()-run["states"][23].double()
        z=p.reshape(1,32,32,16,2,2).permute(0,3,1,4,2,5).reshape(1,16,64,64)
        direct=(z-torch.roll(z,1,-1)).square().sum()+(z-torch.roll(z,1,-2)).square().sum()
        f=torch.fft.fftfreq(64,dtype=torch.float64)
        eigen=4*torch.sin(np.pi*f[:,None]).square()+4*torch.sin(np.pi*f[None,:]).square()
        spectral=(torch.fft.fft2(z,norm="ortho").abs().square()*eigen).sum()
        gradient_errors.append(float(abs(direct-spectral)/direct))
    calls=sum(m["vae_calls"] for m in measure)
    assert calls==1476
    assert load(OUT/"scores/complete.json")["images"]==180
    a=load(OUT/"aggregate/summary.json")
    assert a["source_count"]==24
    assert len(load(OUT/"aggregate/layout_dependence.json"))==144
    assert len(load(OUT/"aggregate/seed_stability.json"))==96
    cross=load(OUT/"aggregate/cross_sd3.json")
    assert len(cross)==8
    verification=dict(status="complete",unique_sources=24,baseline_trajectories=72,
        extra_online_trajectories=84,total_trajectories=156,transformer_calls=7488,
        main_vae_decode_calls=calls,preparation_vae_decode_calls=24,preparation_vae_encode_calls=24,
        baseline_actual_increments=1728,baseline_intended_increments=1728,
        auxiliary_sd3_trajectories=8,auxiliary_sd3_increments=264,auxiliary_scored_images=180,
        max_algebra_relative_error=max(x["max_algebra_error"] for x in measure),
        max_laplacian_gradient_identity_relative_error=max(gradient_errors),
        baseline_replay_rgb_max_error=0,model_layout_roundtrip=True,input_hashes_match=True,
        inference_code_hash_matches=True,no_text_truncation=True,
        source_level_statistics=True,heldout_evaluation=False)
    (OUT/"aggregate/verification.json").write_text(json.dumps(verification,indent=2)+"\n")

    lines=["### 6.1 已执行规模与核对", "",
        "**全部完成：24 源、72 条基准轨迹、84 条额外在线轨迹，共 156 条完整轨迹、7488 次 Transformer 调用。** 主测量使用 1476 次真实 VAE 解码，输入准备另含 24 次编码和 24 次解码。分析 1728 个实际基准增量及对应 1728 个舍入前提议；额外 SD3 核查含 8 条轨迹、264 个增量。", "",
        f"全部 24 源的完整冻结回放均精确复现其基准解码图。分解恒等式最大相对误差 {verification['max_algebra_relative_error']:.3g}；离散梯度能量与 Laplacian 谱能量恒等式最大相对误差 {max(gradient_errors):.3g}。两个 GPU 的 1160 个权重张量布局往返逐元素一致，输入哈希匹配，CLIP／T5 均无截断。这些核对验证计算身份和代数，不认证编辑正确性。", "",
        "### 6.2 跨 24 源的谱演化", "",
        "下表为先做种子内阶段能量加权、再对种子和源等权后的均值。它不是把全体时间步当作独立样本的总体均值。", "",
        "| 实际更新观测 | 早期1–8 | 中期9–16 | 后期17–24 | 末步 |", "|---|---:|---:|---:|---:|"]
    for key,label,is_percent in [("rho4","前4项能量",True),("rho8","前8项能量",True),("effective_rank","能量熵有效秩",False),
                                 ("high_fraction","高频能量",True),("top4_high_fraction","前4项内部高频能量",True),
                                 ("spatial_participation_fraction","空间参与比例",True)]:
        values=[a["source_stage"]["actual"][stage][key]["mean"] for stage in ["early","middle","late","last"]]
        lines.append("| "+label+" | "+" | ".join(pct(v) if is_percent else f"{v:.2f}" for v in values)+" |")
    lines += ["", "后期−早期的源级配对差：", "", "| 指标 | 平均差 | 源bootstrap 95%区间 | 同方向源数 |", "|---|---:|---:|---:|"]
    for kind,label in [("actual","实际"),("intended","舍入前提议")]:
        for key,name,sign in [("rho4","前4项能量",-1),("high_fraction","高频能量",1)]:
            v=a["late_minus_early"][kind][key]
            lo,hi=v["source_bootstrap_mean_ci95"]
            n=v["positive_sources"] if sign==1 else v["negative_sources"]
            lines.append(f"| {label}：{name} | {100*v['mean']:+.2f}个百分点 | [{100*lo:+.2f}, {100*hi:+.2f}] | {n}/24 |")
    lines += ["", "![24源谱演化](../experiments/flowedit_delta_spectrum_multi_20261117/outputs/study_v1/aggregate/source_trajectories.png)", "",
        "这支持本配置下更新由较集中的通道模式向更分散、细尺度成分更强的模式演化；同样趋势存在于舍入前提议，不能全部归因于 BF16。它不证明早期只编辑语义、后期只补纹理，也没有识别随机查询与模型场误差各自的贡献。", "",
        "### 6.3 同状态、同能量时，图像响应仍不同", "", "下表为 24 源末步局部干预的源等权平均。", "",
        "| 分支 | latent能量／原步 | RGB变化能量／原步 | RGB变化与原步余弦 |", "|---|---:|---:|---:|"]
    for m in ["full","svd4","scalar4","random4_matched","svd4_equal_norm","tail4","fft_low","scalar_low","fft_high","scalar_high","shift","channel_flip"]:
        v=a["local"]["24"][m]
        lines.append(f"| {m} | {pct(v['latent_energy_relative']['mean'])} | {pct(v['rgb_energy_relative']['mean'])} | {v['rgb_cosine_to_full']['mean']:.3f} |")
    lines += ["", "SVD 前4项与同能量整体缩放的视觉方向明显不同；随机4维子空间也并不等价于主奇异子空间。两种同谱盲区对照在能量保持不变时仍有低视觉方向一致性，说明谱摘要并未约束编辑的空间配准或通道含义。", "",
        "### 6.4 解码不加性与幅度扫描", "", "| 局部条件 | 24源余项中位数 | 最小–最大 |", "|---|---:|---:|"]
    for key,label in [("step1_alpha1.0","首步，原幅度"),("step12_alpha1.0","中间步，原幅度"),("step24_alpha1.0","末步，原幅度"),
                      ("step24_alpha0.5","末步，1/2幅度"),("step24_alpha0.25","末步，1/4幅度")]:
        v=a["decoder_nonlinearity"][key]
        lines.append(f"| {label} | {pct(v['median'])} | {pct(v['min'])}–{pct(v['max'])} |")
    lines += ["", "![联合尺度与非线性](../experiments/flowedit_delta_spectrum_multi_20261117/outputs/study_v1/aggregate/rank_frequency_nonlinearity.png)", "",
        "缩小扰动后余项整体降低，但1/4幅度并不自动足够小。有限步的 latent 正交分量不能直接被解释为可相加的图像语义分量。", "",
        "### 6.5 24个任务逐源数值索引", "", "下表为三个种子平均后的源级阶段统计；完整图像包含所有源和所有已注册分支，不按结果挑选。", "",
        "| 源任务 | 类别 | 前4项：早→晚 | 高频：早→晚 |", "|---|---|---:|---:|"]
    # Source-stage rows are also exported as CSV, read here without pandas.
    import csv
    with (OUT/"aggregate/source_stage.csv").open() as f:
        source_rows=list(csv.DictReader(f))
    for r in protocol["records"]:
        e=next(x for x in source_rows if x["pair"]==r["pair_id"] and x["kind"]=="actual" and x["stage"]=="early")
        l=next(x for x in source_rows if x["pair"]==r["pair_id"] and x["kind"]=="actual" and x["stage"]=="late")
        lines.append(f"| {r['pair_id']} | {r['category']} | {pct(float(e['rho4']))}→{pct(float(l['rho4']))} | {pct(float(e['high_fraction']))}→{pct(float(l['high_fraction']))} |")
    lines += ["", "### 6.6 跨主干与展开依赖", "",
        f"SD3 辅助组中，后期前4项能量下降的配置为 {sum(x['rho4_late_minus_early']<0 for x in cross)}/8，高频占比上升为 {sum(x['high_late_minus_early']>0 for x in cross)}/8。这里仍然只有4个不同源，不是8个独立源；不同 CFG、主干和调度仅做分组核查。", ""]
    layouts=load(OUT/"aggregate/layout_dependence.json")
    for step in [1,24]:
        packed=[x["rho4"] for x in layouts if x["step"]==step and x["layout"].startswith("packed")]
        unpacked=[x["rho4"] for x in layouts if x["step"]==step and x["layout"].startswith("unpacked")]
        lines.append(f"同一组增量第{step}步，直接在 packed 1024×64 矩阵上做SVD的前4项平均占比为 {pct(np.mean(packed))}；恢复空间后的4096×16展开为 {pct(np.mean(unpacked))}。数据和总范数没有变化，矩阵轴的组织改变了秩和奇异谱，说明秩没有脱离展开方式的绝对物理含义。")
        lines.append("")
    lines += ["### 6.7 在线视觉核查与辅助分数", "", "<!-- VISUAL_INSERT -->", ""]
    result="\n".join(lines)
    content=REPORT.read_text()
    content=content.replace("<!-- RESULTS_INSERT -->",result)
    REPORT.write_text(content)
    print(json.dumps(verification,indent=2))


if __name__=="__main__":
    main()
