import json
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from common import Paired, PAIRS, ROOT, OUT, write, protocol, save_image


def transforms(image):
    yield "identity", image
    yield "bright+0.08", image+.08*image*(1-image)
    yield "bright-0.08", image-.08*image*(1-image)
    yield "contrast+0.08", image+.08*image*(1-image)*(2*image-1)
    yield "contrast-0.08", image-.08*image*(1-image)*(2*image-1)
    for shift in [-8,8]:
        # All pixels shifted with replicated boundary; no edit-region mask.
        padded=F.pad(image,(8,8,0,0),mode="replicate")
        yield "horizontal_shift"+str(shift), padded[:,:,:,8-shift:8-shift+512]
    for scale in [.98,1.02]:
        theta=torch.tensor([[[scale,0,0],[0,scale,0]]],device=image.device,dtype=image.dtype)
        grid=F.affine_grid(theta,image.shape,align_corners=False)
        yield "coordinate_scale"+str(scale), F.grid_sample(image,grid,padding_mode="border",align_corners=False)


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    protocol("observe")
    exp=Paired();allrows=[];summaries=[];drawrows=[]
    for pair in PAIRS:
        start=time.monotonic();ctx=exp.setup(pair)
        saved=torch.load(ROOT/"experiments/flowedit_frequency_trajectory/outputs"/pair/"trajectory.pt",weights_only=True)
        source=exp.values(pair,ctx["fs"])
        rows=[];states=saved["states"]
        for idx in [0,*range(18,51)]:
            row,image,f=exp.evaluate(pair,states[idx].to(exp.device))
            row.update(pair=pair,state_index=idx,step=idx-1)
            rows.append(row)
        other=[]
        for method in ["DRIVE","RAW","CAL","FILTER","ISO"]:
            path=ROOT/"experiments/task_excess_response/outputs"/pair/method/"trajectory.pt"
            old=torch.load(path,weights_only=True)
            row,_,_=exp.evaluate(pair,old["final"].to(exp.device))
            row.update(pair=pair,method=method,existing_trajectory=True);other.append(row)
        pert=[]
        for state,image0 in [("source_RGB",ctx["source_rgb"]),("BASE_mid",exp.sc.image(states[34].to(exp.device)).detach())]:
            for name,image in transforms(image0):
                with torch.no_grad():f,_=exp.feature_image(image);row=exp.values(pair,f)
                row.update(state=state,transform=name);pert.append(row)
        ks=np.array([r["K"] for r in rows]);gs=np.array([r["global_distance"] for r in rows])
        summary=dict(pair=pair,source_original=source,source_vae=rows[0],final=rows[-1],
                     source_pair_cosine=ctx["source_pair_cosine"],source_rho=ctx["source_rho"],
                     text_angle_deg=float(torch.arccos((ctx["embeddings"][0]@ctx["embeddings"][1]).clamp(-1,1))*180/torch.pi),
                     lengths=ctx["lengths"],truncated=[n>64 for n in ctx["lengths"]],
                     K_global_distance_pearson=float(np.corrcoef(ks,gs)[0,1]),
                     max_abs_K_minus_global=float(np.max(np.abs(ks-gs))),
                     short_source_anchor_cosine=float(ctx["m"]@ctx["m_short_source"]),
                     existing_endpoints=other,perturbations=pert,seconds=time.monotonic()-start)
        summaries.append(summary);allrows.extend(rows)
        write(OUT/"observation.json",dict(rows=allrows,summaries=summaries))
        drawrows.append((pair,ctx,[0,18,28,34,44,50],rows))
        print(json.dumps(dict(pair=pair,stage="observe",K_final=rows[-1]["K"],
            correlation=summary["K_global_distance_pearson"],seconds=summary["seconds"])),flush=True)
    fig,axs=plt.subplots(4,3,figsize=(16,12))
    for i,s in enumerate(summaries):
        rows=[r for r in allrows if r["pair"]==s["pair"]];k=[r["step"] for r in rows]
        axs[i,0].plot(k,[r["target_logit"] for r in rows],label="target full")
        axs[i,0].plot(k,[r["source_logit"] for r in rows],label="source full")
        axs[i,0].set_title(s["pair"]+" | original FlowEdit");axs[i,0].legend(fontsize=8)
        for key in ["K","global_distance","K_short_source","K_short_both"]:
            axs[i,1].plot(k,[r[key] for r in rows],label=key)
        axs[i,1].legend(fontsize=7);axs[i,1].set_title("paired prior vs global image distance")
        axs[i,2].scatter([r["global_distance"] for r in rows],[r["K"] for r in rows],c=k)
        axs[i,2].set_xlabel("global distance");axs[i,2].set_ylabel("K")
        axs[i,2].set_title("correlation %.5f"%s["K_global_distance_pearson"])
    fig.tight_layout();fig.savefig(OUT/"observation_metrics.png",dpi=150);plt.close(fig)
    size=220;canvas=Image.new("RGB",(7*size,4*(size+140)),"white");draw=ImageDraw.Draw(canvas)
    import textwrap
    for i,(pair,ctx,indices,rows) in enumerate(drawrows):
        top=i*(size+140);draw.text((5,top),pair+" | Original FlowEdit trajectory",fill="black")
        for j,line in enumerate(textwrap.wrap("SOURCE: "+ctx["texts"][0],180)):
            draw.text((5,top+16+j*12),line,fill="#885000")
        for j,line in enumerate(textwrap.wrap("TARGET: "+ctx["texts"][1],180)):
            draw.text((5,top+52+j*12),line,fill="#005588")
        ims=[Image.open(OUT/pair/"source_rgb.png")]+[
            Image.open(ROOT/"experiments/flowedit_frequency_trajectory/outputs"/pair/"frames"/(
                "source_vae.png" if idx==0 else "step_%02d.png"%(idx-1))) for idx in indices]
        labs=["Source RGB"]+["Source VAE" if idx==0 else "FlowEdit k=%d"%(idx-1) for idx in indices]
        for j,(im,lab) in enumerate(zip(ims,labs)):
            left=j*size;draw.text((left+3,top+94),lab,fill="black")
            canvas.paste(im.resize((size,size)),(left,top+111))
    canvas.save(OUT/"observation_trajectory.png")
    print("OBSERVATION_COMPLETE",flush=True)


if __name__=="__main__":main()
