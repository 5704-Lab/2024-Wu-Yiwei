"""Numeric and image analysis; never supplies editing masks or references."""
import argparse
import csv
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent/"outputs"
PAIRS = ["bear__00","brown_owl__00","bus__00","cat_crown__00"]
METHODS = ["BASE","DRIVE","RAW","CAL","FILTER","ISO"]
COLORS = dict(BASE="black",DRIVE="#e69f00",RAW="#8e44ad",CAL="#0072b2",
              FILTER="#009e73",ISO="#d55e00")


def load_json(path):
    return json.loads(path.read_text())


def pixels(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)/255


def luma(image):
    return image@np.array([.2126,.7152,.0722])


def feature_displacement(source, final):
    """All-patch local descriptor match; neither a mask nor exact optical flow."""
    n=int(np.sqrt(source.shape[0]))
    A=source.reshape(n,n,-1);B=final.reshape(n,n,-1)
    shifts=[];similarities=[]
    for i in range(n):
        for j in range(n):
            options=[]
            for di in range(-2,3):
                for dj in range(-2,3):
                    if 0<=i+di<n and 0<=j+dj<n:
                        options.append((float(A[i,j]@B[i+di,j+dj]),di,dj))
            best=max(options,key=lambda v:v[0])
            similarities.append(best[0])
            shifts.append([best[1],best[2]])
    shifts=np.array(shifts)
    distances=np.linalg.norm(shifts,axis=1)*512/n
    return dict(local_patch_displacement_mean_px=float(distances.mean()),
                local_patch_displacement_rms_px=float(np.sqrt(np.mean(distances**2))),
                local_patch_match_cosine=float(np.mean(similarities)))


def metrics(pair,method):
    folder=OUT/pair/method
    summary=load_json(folder/"summary.json")
    data=torch.load(folder/"trajectory.pt",weights_only=True)
    source= pixels(OUT/pair/"source_vae.png")
    rgb_source=pixels(ROOT/data["record"]["source_image"])
    if rgb_source.shape!=source.shape:
        rgb_source=np.asarray(Image.open(ROOT/data["record"]["source_image"]).convert(
            "RGB").resize((512,512),Image.Resampling.BILINEAR),dtype=np.float64)/255
    final=pixels(folder/"final.png")
    delta=final-source
    gray_source=luma(source);gray_final=luma(final)
    fx=torch.load(OUT/pair/"source_features.pt",weights_only=True)["middle"].numpy()
    fy=data["middle"].numpy()
    power=np.mean(np.abs(np.fft.fft2(delta,axes=(0,1),norm="ortho"))**2,axis=2)/(512*512)
    rad=np.sqrt(np.fft.fftfreq(512)[:,None]**2+np.fft.fftfreq(512)[None,:]**2)
    total=float(power.sum())
    rows=data["rows"]
    record=dict(**summary,
        source_latent_cost=.5*.25*summary["latent_distance"]**2,
        source_feature_cost=summary["source_cost"]-.5*.25*summary["latent_distance"]**2,
        rgb_rmse_source_vae=float(np.sqrt(np.mean(delta**2))),
        rgb_rmse_original=float(np.sqrt(np.mean((final-rgb_source)**2))),
        grayscale_ssim=float(structural_similarity(gray_source,gray_final,data_range=1)),
        srgb_luma_shift=float(gray_final.mean()-gray_source.mean()),
        middle_patch_distance=float(np.mean(np.sum((fx-fy)**2,axis=1))),
        middle_same_position_cosine=float(np.mean(np.sum(fx*fy,axis=1))),
        edit_power=total,high_frequency_edit_power=float(power[rad>=.25].sum()),
        high_frequency_fraction=float(power[rad>=.25].sum()/max(total,1e-30)),
        **feature_displacement(fx,fy))
    if method!="BASE":
        record.update(
            mean_repair_alpha=float(np.mean([r["repair_alpha"] for r in rows])),
            accepted_nonzero=sum(r["repair_alpha"]>0 and r["metric_repair_norm"]>1e-8 for r in rows),
            capped_steps=sum(r["repair_cap_factor"]<1-1e-6 for r in rows),
            task_rank_min=min(r["task_rank"] for r in rows),
            max_task_projection_residual=max(r["task_projection_residual"] for r in rows),
            nuisance_residual_ratio=float(np.linalg.norm([r["nuisance_derivative_residual"] for r in rows]) /
                max(np.linalg.norm([r["nuisance_derivative_raw"] for r in rows]),1e-30)),
            source_reference_decrease_mean=float(np.mean([
                r["source_cost_reference"]-r["source_cost_after"] for r in rows])),
            mean_increment_ratio=float(np.mean([r["increment_ratio"] for r in rows])))
    return record,data


def comparison(pair,methods,size=320):
    rec=load_json(OUT/pair/"input.json")["record"]
    columns=[("Source RGB",Image.open(ROOT/rec["source_image"]).convert("RGB")),
             ("Source VAE",Image.open(OUT/pair/"source_vae.png").convert("RGB"))]
    columns += [("FlowEdit" if m=="BASE" else m,
                 Image.open(OUT/pair/m/"final.png").convert("RGB")) for m in methods]
    font=ImageFont.truetype("DejaVuSans.ttf",15)
    small=ImageFont.truetype("DejaVuSans.ttf",13)
    width=size*len(columns)
    source_lines=textwrap.wrap("SOURCE: "+rec["source_prompt"],width=max(70,width//8))
    target_lines=textwrap.wrap("TARGET: "+rec["target_prompt"],width=max(70,width//8))
    top=55+18*(len(source_lines)+len(target_lines))
    canvas=Image.new("RGB",(width,top+size+86),"white")
    dr=ImageDraw.Draw(canvas)
    dr.text((5,4),pair+" | Same seed, original FlowEdit conditions; step-local response control",font=font,fill="black")
    row=25
    for line in source_lines:
        dr.text((5,row),line,font=small,fill="#875400");row+=18
    for line in target_lines:
        dr.text((5,row),line,font=small,fill="#005b99");row+=18
    for i,(label,image) in enumerate(columns):
        left=i*size
        dr.text((left+5,top-18),label,font=font,fill="black")
        canvas.paste(image.resize((size,size)),(left,top))
        if i>=2:
            s=load_json(OUT/pair/methods[i-2]/"summary.json")
            dr.text((left+4,top+size+4),f"Target logit {s['target_logit']:.2f}",font=small,fill="black")
            dr.text((left+4,top+size+23),f"Source cost {s['source_cost']:.0f}",font=small,fill="black")
            dr.text((left+4,top+size+42),f"Time {s['seconds']:.1f}s",font=small,fill="black")
    dr.text((5,canvas.height-19),"Logits/descriptors are proxies. Lower source distance may reflect under-editing.",font=small,fill="black")
    canvas.save(OUT/pair/"comparison.png")
    return canvas


def trajectory(pair,methods):
    ks=[17,27,33,43,49];size=224;font=ImageFont.truetype("DejaVuSans.ttf",14)
    canvas=Image.new("RGB",(180+size*(len(ks)+1),len(methods)*(size+35)+40),"white")
    dr=ImageDraw.Draw(canvas)
    for r,m in enumerate(methods):
        top=35+r*(size+35)
        dr.text((5,top+12),"FlowEdit" if m=="BASE" else m,font=font,fill="black")
        dr.text((5,top+35),pair,font=font,fill="black")
        files=[OUT/pair/"source_vae.png"]+[OUT/pair/m/f"step_{k:02d}.png" for k in ks]
        for c,p in enumerate(files):
            left=180+c*size
            dr.text((left+4,top-20),"Source VAE" if c==0 else f"k={ks[c-1]}",font=font,fill="black")
            canvas.paste(Image.open(p).convert("RGB").resize((size,size)),(left,top))
    dr.text((5,5),"Full trajectory checkpoints; each row includes source; FlowEdit is the original baseline.",font=font,fill="black")
    canvas.save(OUT/pair/"trajectory_comparison.png")


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--partial",action="store_true");args=ap.parse_args()
    torch.set_num_threads(4)
    records=[];trajectories={};panels=[];complete_pairs=[]
    for pair in PAIRS:
        present=[m for m in METHODS if (OUT/pair/m/"summary.json").exists()]
        if not present or "BASE" not in present:continue
        if not args.partial and present!=METHODS:
            raise RuntimeError(f"incomplete pair {pair}: {present}")
        complete_pairs.append(pair)
        for m in present:
            r,d=metrics(pair,m);records.append(r);trajectories[pair,m]=d["rows"]
        panels.append(comparison(pair,present,size=256))
        selected=[m for m in ["BASE","DRIVE","RAW","CAL","FILTER","ISO"] if m in present]
        trajectory(pair,selected)
    if not records:raise RuntimeError("no finished baseline and methods")
    canvas=Image.new("RGB",(max(p.width for p in panels),sum(p.height for p in panels)),"white")
    top=0
    for p in panels:canvas.paste(p,(0,top));top+=p.height
    canvas.save(OUT/"all_comparison.png")
    baseline={r["pair"]:r for r in records if r["method"]=="BASE"}
    for r in records:
        b=baseline[r["pair"]]
        r["target_logit_delta_vs_base"]=r["target_logit"]-b["target_logit"]
        r["source_cost_change_pct"]=100*(r["source_cost"]/b["source_cost"]-1)
        r["middle_distance_change_pct"]=100*(r["middle_patch_distance"]/b["middle_patch_distance"]-1)
        r["rgb_rmse_change_pct"]=100*(r["rgb_rmse_source_vae"]/b["rgb_rmse_source_vae"]-1)
        r["ssim_delta_vs_base"]=r["grayscale_ssim"]-b["grayscale_ssim"]
        r["runtime_ratio_vs_base"]=r["seconds"]/b["seconds"]
    fields=sorted(set().union(*(r.keys() for r in records)))
    with (OUT/"metrics.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(records)
    (OUT/"analysis.json").write_text(json.dumps(dict(records=records,complete_pairs=complete_pairs,
        metrics_scope="global rendered PNG/source descriptors; no accurate non-edit mask or edit-success oracle"),
        ensure_ascii=False,indent=2)+"\n")
    fig,axes=plt.subplots(len(complete_pairs),3,figsize=(17,3.3*len(complete_pairs)),squeeze=False)
    for j,pair in enumerate(complete_pairs):
        for m in METHODS:
            if (pair,m) not in trajectories:continue
            rows=trajectories[pair,m];x=[r["step"] for r in rows]
            for i,key in enumerate(["target_logit","source_cost_after","rgb_rmse_to_source_vae"]):
                axes[j,i].plot(x,[r[key] for r in rows],label="FlowEdit" if m=="BASE" else m,color=COLORS[m])
        for i,title in enumerate(["Full target logit","Source descriptor cost","RGB RMSE to source VAE"]):
            axes[j,i].set_title(pair+" | "+title);axes[j,i].set_xlabel("Active step k")
            axes[j,i].legend(fontsize=8);axes[j,i].grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/"trajectory_metrics.png",dpi=150);plt.close(fig)
    # Calibration summary includes source and original FlowEdit intermediate/final states.
    cal=load_json(OUT/"calibration.json")["rows"]
    calibration=[]
    for pair in PAIRS:
        rr=[r for r in cal if r["pair"]==pair]
        raw=np.concatenate([np.array(r["raw"]).ravel() for r in rr])
        residual=np.concatenate([np.array(r["residual"]).ravel() for r in rr])
        finite=[(t["task_delta"],t["calibrated_delta"]) for r in rr for t in r["transforms"] if abs(t["strength"])==.03]
        calibration.append(dict(pair=pair,derivative_reduction=1-np.linalg.norm(residual)/np.linalg.norm(raw),
            small_finite_reduction=1-np.linalg.norm([t[1] for t in finite])/np.linalg.norm([t[0] for t in finite]),
            beta_norm_max=max(max(r["coefficient_norm"]) for r in rr)))
    (OUT/"calibration_summary.json").write_text(json.dumps(calibration,indent=2)+"\n")
    print(json.dumps(dict(finished_pairs=complete_pairs,rows=len(records)),indent=2))


if __name__=="__main__":
    main()
