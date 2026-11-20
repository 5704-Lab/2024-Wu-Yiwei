"""Numeric and visual comparisons; no editing references or masks."""
import csv
import json
import textwrap

import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from skimage.metrics import structural_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ROOT,OUT,PAIRS,VARIANTS,write


def pixels(p):return np.asarray(Image.open(p).convert("RGB"),dtype=np.float64)/255


def main():
    data=json.loads((OUT/"condition_summary.json").read_text())
    rows=data["rows"];metrics=[];canvases=[]
    extra=json.loads((OUT/"derivative_audit.json").read_text())["own_condition_rows"]
    font=ImageFont.truetype("DejaVuSans.ttf",15);small=ImageFont.truetype("DejaVuSans.ttf",12)
    for pair in PAIRS:
        source=pixels(OUT/pair/"source_vae.png");src=pixels(OUT/pair/"source_rgb.png")
        lum=np.array([.2126,.7152,.0722]);columns=[("Source RGB",OUT/pair/"source_rgb.png"),("Source VAE",OUT/pair/"source_vae.png")]
        base=next(r for r in rows if r["pair"]==pair and r["variant"]=="BASE")
        for variant in VARIANTS:
            candidates=[r for r in rows if r["pair"]==pair and r["variant"]==variant]
            if not candidates:continue
            r=candidates[0];path=OUT/pair/variant/"final.png";im=pixels(path)
            source_index=2 if variant in ["SHORT_SOURCE","SHORT_BOTH","MATCHED_CFG_SHORT_BOTH","COUNTERFACTUAL_SHORT_BOTH"] else 0
            target_index=0 if variant=="NO_EDIT_ASYMMETRIC_CFG" else (3 if variant in ["SHORT_TARGET","SHORT_BOTH","MATCHED_CFG_SHORT_BOTH","COUNTERFACTUAL_SHORT_BOTH"] else 1)
            scores=[r["source_logit"],r["target_logit"],r["short_source_logit"],r["short_target_logit"]]
            m=dict(**r,condition_source_logit=scores[source_index],condition_target_logit=scores[target_index],
                   gray_ssim=float(structural_similarity(source@lum,im@lum,data_range=1)),
                   rgb_rmse_original=float(np.sqrt(np.mean((im-src)**2))),
                   sRGB_luma_shift=float((im@lum).mean()-(source@lum).mean()),
                   K_change_pct_vs_BASE=100*(r["K"]/base["K"]-1),
                   target_logit_change_vs_BASE=r["target_logit"]-base["target_logit"])
            audit=next((x for x in extra if x["pair"]==pair and x["variant"]==variant),None)
            if audit:
                m.update({k:audit[k] for k in ["actual_condition_K","source_original_patch_relation_mse","source_original_position_feature_mse"]})
            metrics.append(m);columns.append(("FlowEdit" if variant=="BASE" else variant,path))
        size=300;width=size*len(columns);header=160
        canvas=Image.new("RGB",(width,header+size+180),"white");draw=ImageDraw.Draw(canvas)
        draw.text((5,4),pair+" | Same source, seed0, SD3 FP32; source/target CFG labeled per column",font=font,fill="black")
        for row,text in [(26,"Original SOURCE: "+base["source_text"]),(78,"Original TARGET: "+base["target_text"])]:
            for j,line in enumerate(textwrap.wrap(text,width=max(100,width//7))):
                draw.text((5,row+16*j),line,font=small,fill="#654000" if row==26 else "#004466")
        for i,(label,path) in enumerate(columns):
            left=i*size;draw.text((left+4,header-23),label,font=small,fill="black")
            canvas.paste(Image.open(path).convert("RGB").resize((size,size)),(left,header))
            if i>=2:
                variant=VARIANTS[i-2]
                m=next(r for r in metrics if r["pair"]==pair and r["variant"]==variant)
                src="Full source (shown above)" if m["source_text"]==base["source_text"] else m["source_text"]
                tar="Full target (shown above)" if m["target_text"]==base["target_text"] else ("Full source: no text edit" if variant=="NO_EDIT_ASYMMETRIC_CFG" else m["target_text"])
                texts=["S: "+src,"T: "+tar]
                y=header+size+5
                for t in texts:
                    for line in textwrap.wrap(t,width=44):
                        draw.text((left+4,y),line,font=small,fill="black");y+=14
                draw.text((left+4,header+size+140),"Original-target %.3f | K(full) %.4f"%(m["target_logit"],m["K"]),font=small,fill="black")
                label=("Fidelity/contrast: %.1f/%.1f (2x4 preds)" if variant.startswith("COUNTERFACTUAL") else "CFG source/target: %.1f/%.1f")%(m["source_cfg"],m["target_cfg"])
                draw.text((left+4,header+size+155),label,font=small,fill="black")
        canvas.save(OUT/pair/"conditions_comparison.png");canvases.append(canvas)
        # A readable subset for the mechanism comparison; full controls remain above.
        focused=[("Source RGB",OUT/pair/"source_rgb.png",None),
                 ("Source VAE",OUT/pair/"source_vae.png",None)]+[
                    (label,OUT/pair/v/"final.png",v) for v,label in [
                        ("BASE","Original FlowEdit"),
                        ("NO_EDIT_ASYMMETRIC_CFG","Same text, unequal CFG"),
                        ("COUNTERFACTUAL_FULL","Counterfactual full text"),
                        ("COUNTERFACTUAL_SHORT_BOTH","Counterfactual short text")]]
        compact=Image.new("RGB",(1920,580),"white");cd=ImageDraw.Draw(compact)
        cd.text((5,3),pair+" | Same source and seed; actual prompts and method labels",font=font,fill="black")
        yy=25
        for title,txt in [("SOURCE",base["source_text"]),("TARGET",base["target_text"])]:
            for line in textwrap.wrap(title+": "+txt,width=230):cd.text((5,yy),line,font=small,fill="black");yy+=15
        for i,(label,path,v) in enumerate(focused):
            left=320*i;cd.text((left+4,133),label,font=small,fill="black")
            compact.paste(Image.open(path).resize((320,320)),(left,155))
            if v:
                m=next(r for r in metrics if r["pair"]==pair and r["variant"]==v)
                cd.text((left+4,482),"RMSE %.4f | original-target %.3f"%(m["rgb_rmse_source_vae"],m["target_logit"]),font=small,fill="black")
                label="Target = full SOURCE above" if v=="NO_EDIT_ASYMMETRIC_CFG" else ("Full source/target above" if v in ["BASE","COUNTERFACTUAL_FULL"] else "S: "+m["source_text"]+" T: "+m["target_text"])
                for j,line in enumerate(textwrap.wrap(label,width=43)):cd.text((left+4,502+14*j),line,font=small,fill="black")
        compact.save(OUT/pair/"mechanism_comparison.png")
    stacked=Image.new("RGB",(max(x.width for x in canvases),sum(x.height for x in canvases)),"white")
    top=0
    for x in canvases:stacked.paste(x,(0,top));top+=x.height
    stacked.save(OUT/"conditions_all_comparison.png")
    write(OUT/"analysis.json",dict(metrics=metrics,identity_checks=data["identity_checks"]))
    if metrics:
        with (OUT/"metrics.csv").open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=sorted(set().union(*(r.keys() for r in metrics))));writer.writeheader();writer.writerows(metrics)
    fig,axs=plt.subplots(4,3,figsize=(16,12))
    for i,pair in enumerate(PAIRS):
        for variant in VARIANTS:
            path=OUT/pair/variant/"progress.json"
            if not path.exists():continue
            rec=json.loads(path.read_text())["rows"];k=[r["step"] for r in rec]
            for j,field in enumerate(["target_logit","K","source_predicted_clean_error"]):
                axs[i,j].plot(k,[r[field] for r in rec],label=variant)
                axs[i,j].set_title(pair+" | "+field)
        axs[i,0].legend(fontsize=6)
    fig.tight_layout();fig.savefig(OUT/"condition_trajectory_metrics.png",dpi=140);plt.close(fig)
    # Local comparisons always include the same FlowEdit candidate.
    local=json.loads((OUT/"local.json").read_text())["rows"]
    canvas=Image.new("RGB",(6*220,4*350),"white");draw=ImageDraw.Draw(canvas)
    for i,pair in enumerate(PAIRS):
        draw.text((3,i*350+3),pair+" | local repair lambda=0.3 (not full trajectories)",font=small,fill="black")
        record=torch.load(ROOT/"experiments/stage1_full/inputs"/(pair+".pt"),weights_only=True)["record"]
        yy=i*350+21
        for text in ["SOURCE: "+record["source_prompt"],"TARGET: "+record["target_prompt"]]:
            for line in textwrap.wrap(text,width=175):draw.text((3,yy),line,font=small,fill="black");yy+=14
        for j,step in enumerate([17,33,47]):
            for k,method in enumerate(["FlowEdit","PAIR_lam0.3"]):
                left=(2*j+k)*220;top=i*350+115
                image=Image.open(OUT/pair/"local"/(method+"_step%d.png"%step))
                canvas.paste(image.resize((220,220)),(left,top))
                draw.text((left+3,top-15),method+" k="+str(step),font=small,fill="black")
    canvas.save(OUT/"local_comparison.png")
    print("ANALYSIS_COMPLETE",len(metrics),flush=True)


if __name__=="__main__":main()
