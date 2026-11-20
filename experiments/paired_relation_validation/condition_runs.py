"""Actual FlowEdit controls for source/target description removal and CFG."""
import json
import time
from types import SimpleNamespace
import sys

import torch
import torch.nn.functional as F
from diffusers import SD3Transformer2DModel

from common import Paired,ROOT,OUT,PAIRS,SHORT,VARIANTS,write,protocol,save_image


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    protocol("condition_runs",dict(T=50,n_max=33,n_min=0,n_avg=1,
        source_cfg=3.5,target_cfg=13.5,variants=VARIANTS,condition_codes_fp32=True))
    exp=Paired()
    encoder=torch.load(OUT/"short_condition_encodings.pt",weights_only=True)
    model=SD3Transformer2DModel.from_pretrained(ROOT/"pretrained_models/sd3-medium/transformer",
        torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to(exp.device)
    pipe=SimpleNamespace(transformer=model,do_classifier_free_guidance=True)
    sys.path.insert(0,str(ROOT/"vendor/flowedit"))
    from FlowEdit_utils import calc_v_sd3
    times=torch.tensor(json.loads((ROOT/"experiments/stage1_full/inputs/manifest.json").read_text())["timesteps"],device=exp.device)
    summaries=[json.loads(p.read_text()) for pair in PAIRS for p in
               [OUT/pair/v/"summary.json" for v in VARIANTS] if p.exists()]
    old_summary=json.loads((OUT/"condition_summary.json").read_text()) if (OUT/"condition_summary.json").exists() else {}
    identity=old_summary.get("identity_checks",[])
    for pair in PAIRS:
        ctx=exp.setup(pair);x=ctx["x"];data=ctx["data"]
        with torch.no_grad():
            _,o=exp.feature_image(ctx["source_vae"],hidden=True)
            ctx["source_middle"]=F.normalize(o.hidden_states[6][0],dim=-1)
        short=encoder["embeddings"][pair]
        full=data["embeddings"]
        conditions={"BASE":full,"SHORT_SOURCE":[short[0],full[1]],
            "SHORT_TARGET":[full[0],short[1]],"SHORT_BOTH":short,
            "NO_EDIT_ASYMMETRIC_CFG":[full[0],full[0]],
            "MATCHED_CFG_FULL":full,"MATCHED_CFG_SHORT_BOTH":short,
            "COUNTERFACTUAL_FULL":full,"COUNTERFACTUAL_SHORT_BOTH":short}
        gen=torch.Generator("cpu").manual_seed(0)
        noise=[torch.randn(x.shape,generator=gen).to(exp.device) for _ in range(33)]
        baseline=torch.load(ROOT/"experiments/flowedit_frequency_trajectory/outputs"/pair/"trajectory.pt",weights_only=True)
        # At y=x and unchanged condition, equal CFG must yield zero update.
        es=tuple(t.to(exp.device) for t in full[0])
        seq0=torch.cat([es[1],es[0],es[1],es[0]])
        pool0=torch.cat([es[3],es[2],es[3],es[2]])
        qs0=(1-times[17]/1000)*x+times[17]/1000*noise[0]
        with torch.no_grad():
            vs0,va0=calc_v_sd3(pipe,torch.cat([qs0]*4),seq0,pool0,3.5,3.5,times[17])
        identity=[r for r in identity if r["pair"]!=pair]+[dict(pair=pair,equal_cfg_same_text_velocity_difference=float((va0-vs0).norm()))]
        for variant in VARIANTS:
            dest=OUT/pair/variant
            if (dest/"summary.json").exists():
                continue
            started=time.monotonic();y=x.clone();rows=[];states=[y.cpu()]
            es,et=[tuple(t.to(exp.device) for t in e) for e in conditions[variant]]
            seq=torch.cat([es[1],es[0],et[1],et[0]])
            pool=torch.cat([es[3],es[2],et[3],et[2]])
            if variant in ["BASE","MATCHED_CFG_FULL","COUNTERFACTUAL_FULL"]:src=ctx["rec"]["source_prompt"];tar=ctx["rec"]["target_prompt"]
            elif variant=="SHORT_SOURCE":src=SHORT[pair][0];tar=ctx["rec"]["target_prompt"]
            elif variant=="SHORT_TARGET":src=ctx["rec"]["source_prompt"];tar=SHORT[pair][1]
            elif variant in ["SHORT_BOTH","MATCHED_CFG_SHORT_BOTH","COUNTERFACTUAL_SHORT_BOTH"]:src,tar=SHORT[pair]
            else:src=tar=ctx["rec"]["source_prompt"]
            source_cfg=13.5 if variant.startswith("MATCHED_CFG") else 3.5
            write(dest/"conditions.json",dict(source=src,target=tar,source_cfg=source_cfg,target_cfg=13.5))
            for k in range(17,50):
                t=times[k]/1000;tn=times[k+1]/1000 if k<49 else torch.zeros_like(t)
                h=float(t-tn);qs=(1-t)*x+t*noise[k-17];qa=qs+y-x
                with torch.no_grad():
                    if variant.startswith("COUNTERFACTUAL"):
                        # Keep the original four-item batch for all baseline predictions.
                        raw=model(hidden_states=torch.cat([qs,qs,qa,qa]),
                            timestep=times[k].expand(4),encoder_hidden_states=seq,
                            pooled_projections=pool,return_dict=False)[0]
                        us,cs,ua,cta=raw.chunk(4)
                        source_raw=model(hidden_states=torch.cat([qs,qs,qa,qa]),
                            timestep=times[k].expand(4),
                            encoder_hidden_states=torch.cat([es[1],es[0],es[1],es[0]]),
                            pooled_projections=torch.cat([es[3],es[2],es[3],es[2]]),
                            return_dict=False)[0]
                        us2,cs2,ua2,csa=source_raw.chunk(4)
                        vs=us+3.5*(cs-us)
                        source_at_target=ua+3.5*(csa-ua)
                        steering=13.5*(cta-csa)
                        difference=source_at_target-vs+steering
                        va=vs+difference
                        if k==17:
                            v4s,v4a=calc_v_sd3(pipe,torch.cat([qs,qs,qa,qa]),seq,pool,3.5,13.5,times[k])
                            raw_target=ua+13.5*(cta-ua)
                            batch_error=max(float((vs-v4s).norm()/v4s.norm()),
                                            float((raw_target-v4a).norm()/v4a.norm()),
                                            float((ua2-ua).norm()/ua.norm()))
                            write(dest/"batch_audit.json",dict(relative_error=batch_error,batch_sizes=[4,4],implementation="original four plus source-only four"))
                            if batch_error>1e-5:raise RuntimeError("four/four batch compatibility failed")
                    else:
                        vs,va=calc_v_sd3(pipe,torch.cat([qs,qs,qa,qa]),seq,pool,source_cfg,13.5,times[k])
                    d=-h*(va-vs)
                    mus=qs-t*vs;mua=qa-t*va
                    identity_d=h/float(t)*(mua-mus-(y-x))
                    new=y+d
                values,image,f=exp.evaluate(pair,new)
                row=dict(step=k,t=float(t),h=h,**values,
                    source_predicted_clean_error=float((mus-x).norm()/x.norm()),
                    posterior_mean_step_identity_relative_error=float((d-identity_d).norm()/d.norm().clamp_min(1e-10)),
                    nominal_increment_norm=float(d.norm()),q_sync_algebra_error=float((qa-qs-(y-x)).norm()),
                    editing_state_finite=bool(torch.isfinite(new).all()))
                rows.append(row);states.append(new.cpu());y=new
                if k in [17,27,33,43,49]:save_image(image,dest/("step_%02d.png"%k))
                write(dest/"progress.json",dict(pair=pair,variant=variant,rows=rows))
            vals,image,f=exp.evaluate(pair,y,hidden=True);save_image(image,dest/"final.png")
            error=float((y.cpu().double()-baseline["final"].double()).norm()/baseline["final"].double().norm())
            if variant=="BASE" and error>1e-7:raise RuntimeError("FlowEdit baseline reproduction failed")
            summary=dict(pair=pair,variant=variant,source_text=src,target_text=tar,
                source_cfg=source_cfg,target_cfg=13.5,
                guidance_mode="moderate-source-fidelity + matched text contrast" if variant.startswith("COUNTERFACTUAL") else "original two-branch CFG",
                active_steps=33,seconds=time.monotonic()-started,
                relative_latent_distance_to_BASE=error,**vals)
            torch.save(dict(record=ctx["rec"],source_text=src,target_text=tar,
                states=torch.stack(states),rows=rows,final=y.cpu()),dest/"trajectory.pt")
            write(dest/"summary.json",summary);summaries.append(summary)
            write(OUT/"condition_summary.json",dict(rows=summaries,identity_checks=identity))
            print(json.dumps(dict(pair=pair,variant=variant,complete=True,
                target_logit=summary["target_logit"],K=summary["K"],seconds=summary["seconds"])),flush=True)
    write(OUT/"condition_summary.json",dict(rows=summaries,identity_checks=identity))
    print("CONDITION_RUNS_COMPLETE",flush=True)


if __name__=="__main__":main()
