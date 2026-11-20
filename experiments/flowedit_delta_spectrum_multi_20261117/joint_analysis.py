"""Joint impact observables from the completed cohort; no new model calls."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/"outputs/study_v1"
DEST=OUT/"joint_analysis"


def read(path):
    return json.loads(path.read_text())


def envelope(a):
    a=np.array(a,dtype=float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),min=float(a.min()),max=float(a.max()))


def main():
    DEST.mkdir(exist_ok=True)
    protocol=read(OUT/"protocol.json")
    rows=[]
    for record in protocol["records"]:
        pair=record["pair_id"]
        local=read(OUT/"measurements"/pair/"local_metrics.json")
        for step in [1,12,24]:
            full=next(x for x in local if x["step"]==step and x["method"]=="full")
            for x in [r for r in local if r["step"]==step]:
                q=x["rgb_rms"]/full["rgb_rms"]
                c=x["rgb_cosine_to_full"]
                error=x["rgb_error_to_full"]/full["rgb_rms"]
                rho=x["latent_energy_relative"]
                gain=q*q/rho
                rows.append(dict(pair=pair,step=step,method=x["method"],
                    latent_energy_ratio=rho,visual_energy_ratio=q*q,
                    decoder_relative_energy_gain=gain,visual_rms_ratio=q,
                    visual_direction_cosine=c,relative_impact_error=error,
                    amplitude_squared_error=(q-1)**2,
                    direction_squared_error=2*q*(1-c),
                    joint_identity_absolute_error=abs(error*error-(1+q*q-2*q*c))))
    error=max(r["joint_identity_absolute_error"] for r in rows)
    assert error<1e-9,error
    metrics=["latent_energy_ratio","visual_energy_ratio","decoder_relative_energy_gain",
             "visual_rms_ratio","visual_direction_cosine","relative_impact_error",
             "amplitude_squared_error","direction_squared_error"]
    summary={str(step):{method:{key:envelope([r[key] for r in rows if r["step"]==step and r["method"]==method]) for key in metrics}
                       for method in sorted({x["method"] for x in rows})} for step in [1,12,24]}
    scores=read(OUT/"scores/all_scores.json")
    online=[]
    for pair in protocol["online_sources"]:
        source=next(x for x in scores if x["pair"]==pair and x["method"]=="source")
        vis=read(OUT/"measurements"/pair/"online_metrics.json")
        for method in protocol["arms"]:
            s=next(x for x in scores if x["pair"]==pair and x["method"]==method and x["seed"]==0)
            v=next(x for x in vis if x["method"]==method and x["seed"]==0)
            online.append(dict(pair=pair,method=method,target_proxy_gain=s["margin_gain_from_source"],
                               source_rgb_rms=v["rgb_rms"],source_feature_cosine=s["image_source_feature_cosine"]))
    online_summary={m:{key:envelope([x[key] for x in online if x["method"]==m]) for key in ["target_proxy_gain","source_rgb_rms","source_feature_cosine"]} for m in protocol["arms"]}
    result=dict(source_count=24,local_conditions=len(rows),new_transformer_calls=0,new_vae_calls=0,
                identity="relative_error^2 = 1 + visual_rms_ratio^2 - 2*visual_rms_ratio*visual_direction_cosine",
                max_identity_absolute_error=error,local=summary,online=online_summary,
                interpretation="Measured effect decomposition, not a prediction model or semantic quality certificate")
    (DEST/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    (DEST/"local_rows.json").write_text(json.dumps(rows,indent=2,allow_nan=False)+"\n")
    (DEST/"online_rows.json").write_text(json.dumps(online,indent=2,allow_nan=False)+"\n")
    methods=["svd4","scalar4","fft_low","fft_high","shift","channel_flip"]
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for i,m in enumerate(methods):
        a=[x for x in rows if x["step"]==24 and x["method"]==m]
        color=plt.get_cmap("tab10")(i)
        axes[0].scatter([x["visual_rms_ratio"] for x in a],[x["visual_direction_cosine"] for x in a],label=m,c=[color],alpha=.5,s=20)
    axes[0].set(xlabel="RGB change RMS / original step RMS",ylabel="RGB change cosine with original step",
                title="24 sources, step 24: magnitude and direction")
    axes[0].legend(fontsize=8)
    for i,m in enumerate(protocol["arms"]):
        a=[x for x in online if x["method"]==m]
        axes[1].scatter([x["source_rgb_rms"] for x in a],[x["target_proxy_gain"] for x in a],label=m,alpha=.6,s=20)
    axes[1].set(xlabel="Source-to-endpoint RGB RMS (includes legal edits)",ylabel="Auxiliary target-vs-source text margin gain",
                title="12 online sources: task proxy and total source change")
    axes[1].legend(fontsize=7,ncol=2)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(DEST/"joint_impact.png",dpi=160)
    plt.close(fig)
    print(json.dumps(dict(sources=24,conditions=len(rows),max_identity_error=error,new_model_calls=0)))
    for m in ["svd4","scalar4","fft_low","scalar_low","fft_high","scalar_high","shift","channel_flip"]:
        x=summary["24"][m]
        print(m,{key:round(x[key]["mean"],5) for key in metrics})
    print("ONLINE",{m:{k:round(v["mean"],5) for k,v in d.items()} for m,d in online_summary.items()})


if __name__=="__main__":
    main()
