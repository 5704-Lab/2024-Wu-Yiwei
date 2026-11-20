"""Source-level aggregation, uncertainty and complete visual indexes."""
import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs/study_v1"
sys.path.insert(0, str(ROOT/"experiments/flowedit_delta_spectrum_20261117"))
from analyze import unpack, decompose
sys.path.insert(0, str(ROOT/"experiments/sota_inversion_20261105"))
from common import write

KEYS = ["rho1", "rho4", "rho8", "effective_rank", "high_fraction", "low_fraction", "dc_fraction",
        "top4_high_fraction", "spatial_participation_fraction", "spectral_centroid", "laplacian_roughness"]
STAGES = {"early":range(1,9), "middle":range(9,17), "late":range(17,25), "last":[24]}


def envelope(values):
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20261117)
    boot = a[rng.integers(len(a), size=(2000,len(a)))].mean(1)
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)), min=float(a.min()), max=float(a.max()),
                p10=float(np.quantile(a,.1)), p90=float(np.quantile(a,.9)),
                source_bootstrap_mean_ci95=np.quantile(boot,[.025,.975]).tolist())


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(4)
    protocol = json.loads((OUT/"protocol.json").read_text())
    records = [r for r in protocol["records"] if (OUT/"measurements"/r["pair_id"]/"complete.json").exists()]
    if not args.partial:
        assert len(records)==24
    summaries, steps, locals_, nonlin, online, rounding = [], [], [], [], [], []
    for r in records:
        d = OUT/"measurements"/r["pair_id"]
        rows = json.loads((d/"step_statistics.json").read_text())
        steps.extend(rows)
        for kind in ["actual", "intended"]:
            for stage, indices in STAGES.items():
                byseed = []
                for seed in protocol["seeds"]:
                    selected = [x for x in rows if x["kind"]==kind and x["seed"]==seed and x["step"] in indices]
                    total = sum(x["energy"] for x in selected)
                    byseed.append({key:sum(x[key]*x["energy"] for x in selected)/total for key in KEYS})
                item = dict(pair=r["pair_id"], category=r["category"], kind=kind, stage=stage,
                            **{key:float(np.mean([s[key] for s in byseed])) for key in KEYS})
                summaries.append(item)
        locals_.extend(json.loads((d/"local_metrics.json").read_text()))
        nonlin.extend([dict(pair=r["pair_id"],category=r["category"],**x) for x in json.loads((d/"nonlinearity.json").read_text())])
        online.extend(json.loads((d/"online_metrics.json").read_text()))
        actual = [x for x in rows if x["kind"]=="actual"]
        rounding.append(dict(pair=r["pair_id"], mean_rounding_relative=float(np.mean([x["rounding_relative"] for x in actual])),
                             mean_unchanged_fraction=float(np.mean([x["unchanged_fraction"] for x in actual]))))

    if not records:
        raise RuntimeError("No completed sources")
    aggregated = {kind:{stage:{key:envelope([x[key] for x in summaries if x["kind"]==kind and x["stage"]==stage])
                for key in KEYS} for stage in STAGES} for kind in ["actual","intended"]}
    trends = {}
    for kind in ["actual","intended"]:
        delta = {key:[] for key in KEYS}
        for r in records:
            a = next(x for x in summaries if x["kind"]==kind and x["stage"]=="early" and x["pair"]==r["pair_id"])
            b = next(x for x in summaries if x["kind"]==kind and x["stage"]=="late" and x["pair"]==r["pair_id"])
            for key in KEYS:
                delta[key].append(b[key]-a[key])
        trends[kind] = {key:dict(**envelope(v), positive_sources=int(sum(x>0 for x in v)), negative_sources=int(sum(x<0 for x in v))) for key,v in delta.items()}
    nonlinearity = {f"step{step}_alpha{alpha}":envelope([x["rgb_remainder"] for x in nonlin if x["step"]==step and x["alpha"]==alpha])
                    for step,alpha in [(1,1.),(12,1.),(24,1.),(24,.25),(24,.5)]}
    local_summary = {str(step):{method:{key:envelope([x[key] for x in locals_ if x["step"]==step and x["method"]==method])
                     for key in ["latent_energy_relative","rgb_energy_relative","rgb_cosine_to_full"]}
                    for method in sorted({x["method"] for x in locals_})} for step in [1,12,24]}
    paired_online = []
    for filtered,control in [("svd4","scalar4"),("fft_low","scalar_low"),("fft_high","scalar_high")]:
        valid = sorted({x["pair"] for x in online if x["method"]==filtered})
        if valid:
            pair_rows = []
            for pair in valid:
                a = next(x for x in online if x["pair"]==pair and x["method"]==filtered)
                b = next(x for x in online if x["pair"]==pair and x["method"]==control)
                pair_rows.append(dict(pair=pair,filtered=filtered,control=control,
                                      source_rms_difference=a["rgb_rms"]-b["rgb_rms"],
                                      baseline_error_difference=a["rgb_error_to_full"]-b["rgb_error_to_full"]))
            paired_online.extend(pair_rows)

    prefix = "partial_" if args.partial else ""
    dest = OUT/"aggregate"
    dest.mkdir(exist_ok=True)
    summary = dict(source_count=len(records), baseline_trajectories=len(records)*3,
                   actual_increments=len(records)*72, intended_increments=len(records)*72,
                   source_stage=aggregated, late_minus_early=trends, decoder_nonlinearity=nonlinearity,
                   local=local_summary, rounding=rounding,
                   online_paired={f:{key:envelope([x[key] for x in paired_online if x["filtered"]==f])
                       for key in ["source_rms_difference","baseline_error_difference"]}
                       for f in sorted({x["filtered"] for x in paired_online})},
                   uncertainty="2000 bootstrap resamples of source-level seed means; descriptive for this fixed development cohort",
                   spectrum_note="stage energy weighted within seed, then equal weight seeds and sources")
    write(dest/f"{prefix}summary.json",summary)
    save_csv(dest/f"{prefix}source_stage.csv",summaries)
    save_csv(dest/f"{prefix}all_steps.csv",steps)
    save_csv(dest/f"{prefix}online_pairs.csv",paired_online)

    fig, axes = plt.subplots(2,3,figsize=(15,8))
    for ax,key in zip(axes.flat,["rho4","effective_rank","high_fraction","top4_high_fraction","dc_fraction","rms"]):
        for r in records:
            rows = [x for x in steps if x["pair"]==r["pair_id"] and x["kind"]=="actual"]
            y = [np.mean([x[key] for x in rows if x["step"]==k]) for k in range(1,25)]
            ax.plot(range(1,25),y,alpha=.35,lw=1)
        ax.set_title(f"{key}: one curve per source, mean of 3 seeds")
        ax.set_xlabel("active step")
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(dest/f"{prefix}source_trajectories.png",dpi=160)
    plt.close(fig)

    colors = {c:plt.get_cmap("tab10")(i) for i,c in enumerate(sorted({r["category"] for r in records}))}
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    for stage,marker in [("early","o"),("late","^")]:
        for r in records:
            row = next(x for x in summaries if x["pair"]==r["pair_id"] and x["kind"]=="actual" and x["stage"]==stage)
            axes[0].scatter(row["rho4"],row["high_fraction"],c=[colors[r["category"]]],marker=marker,alpha=.75)
    axes[0].set(xlabel="top-4 energy fraction",ylabel="high-frequency fraction",title="Source means; circle=early, triangle=late")
    for alpha in [.25,.5,1.]:
        a = [x["rgb_remainder"] for x in nonlin if x["step"]==24 and x["alpha"]==alpha]
        axes[1].scatter([alpha]*len(a),a,alpha=.4,s=18)
        axes[1].plot(alpha,np.median(a),"k_",markersize=18)
    axes[1].set(xlabel="step-24 perturbation amplitude",ylabel="decoder non-additivity relative to full change",title="Every source; black bars are medians")
    fig.tight_layout()
    fig.savefig(dest/f"{prefix}rank_frequency_nonlinearity.png",dpi=160)
    plt.close(fig)

    # Complete visual pages, in the fixed protocol order, no selected winners.
    online_records = [r for r in records if r["pair_id"] in protocol["online_sources"]]
    for prefix2,chosen,filename in [("online",online_records,"online.png"),("replay",records,"frozen_replay.png"),("seeds",records,"seeds.png")]:
        for page in range((len(chosen)+3)//4):
            group = chosen[page*4:page*4+4]
            ims = [Image.open(OUT/"measurements"/r["pair_id"]/filename).convert("RGB") for r in group]
            w,h = max(im.width for im in ims),sum(im.height for im in ims)
            canvas=Image.new("RGB",(w,h),"white")
            y=0
            for im in ims:
                canvas.paste(im,(0,y));y+=im.height
            canvas.save(dest/f"{prefix}{prefix2}_page{page+1}.png")
    print(json.dumps(dict(sources=len(records), actual_increments=len(records)*72, output=str(dest), trends={k:{x:trends[k][x]["mean"] for x in ["rho4","high_fraction"]} for k in trends})),flush=True)


if __name__ == "__main__":
    main()
