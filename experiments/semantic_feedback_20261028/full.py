"""Real FlowEdit pilot: current clean residual feedback after every raw update.

One retained development bear, seed zero, existing original model and prompts.
No oracle references, masks, inverse paths, new models or weight fitting.
"""
import hashlib
import json
import time

import torch
from torch.nn.attention import sdpa_kernel,SDPBackend

from repair import Repair,basis,cg,vector,HERE
from extract import ROOT
from common import seed_setup,write,save_image,panel


def main():
    seed_setup();run=Repair();out=HERE/'outputs/full_bear_v2';out.mkdir(exist_ok=True)
    data=torch.load(ROOT/'experiments/stage1_full/inputs/bear__00.pt',weights_only=True)
    es,et=data['embeddings'];x=data['x'].to(run.device)
    run.seq=torch.cat([es[0],et[0]]).to(run.device);run.pool=torch.cat([es[2],et[2]]).to(run.device)
    run.text,_=run.sc.encode([data['record']['source_prompt'],data['record']['target_prompt']])
    seq=torch.cat([es[1],es[0],et[1],et[0]]).to(run.device)
    pool=torch.cat([es[3],es[2],et[3],et[2]]).to(run.device)
    times=torch.tensor(json.loads((ROOT/'experiments/stage1_full/inputs/manifest.json').read_text())['timesteps'],device=run.device)
    eps=torch.randn(x.shape,generator=torch.Generator('cpu').manual_seed(19571028)).to(run.device)
    generator=torch.Generator('cpu').manual_seed(0)
    noises=[torch.randn(x.shape,generator=generator).to(run.device) for _ in range(33)]
    with torch.no_grad():
        zx=run.observation(x,eps).detach()
        # Obtain the true conditional-feature source column with the same probe.
        run.capture_grad=None
        q=((1-.35)*x+.35*eps).expand(2,-1,-1,-1)
        from repair import Captured
        try:run.model(hidden_states=q,timestep=torch.full((2,),350.,device=run.device),encoder_hidden_states=run.seq,pooled_projections=run.pool,return_dict=False)
        except Captured:pass
        h=run.capture_grad;run.capture_grad=None
        atext=(vector(h[1:2])-vector(h[0:1])).detach().cpu()
        source_image=run.sc.image(x)
    columns=[ ]
    for tau in [.35,.65]:
        with torch.no_grad():
            run.intercept=False
            q=((1-tau)*x+tau*eps).expand(2,-1,-1,-1)
            v=run.model(hidden_states=q,timestep=torch.full((2,),tau*1000,device=run.device),encoder_hidden_states=run.seq,pooled_projections=run.pool,return_dict=False)[0]
            run.intercept=True
            aux=x+.15*(v[0:1]-v[1:2])
            columns.append((run.observation(aux,eps)-zx).detach().cpu())
    protocol=dict(script_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest(),pair='bear__00',seed=0,
        source_cfg=3.5,target_cfg=13.5,steps=list(range(17,50)),observer_time=.35,observer_layer=11,cg_steps=6,
        correction_radius_relative_to_raw_step=.5,task_retention=.9,task_tolerance=.02,
        space='actual clean y; standardized source/current observer; freshly synchronized coupled FlowEdit query each step',
        scope='one-source natural pilot, not heldout quality validation',spatial_mask=False,extra_model=False,oracle_reference=False)
    write(out/'protocol.json',protocol);start=time.monotonic()
    for mode in ['BASE','TEXT','INDUCED']:
        dest=out/mode;dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists():continue
        Q=basis([atext] if mode=='TEXT' else columns,run.device)
        y=x.clone();states=[y.cpu()];rows=[]
        for k in range(17,50):
            t=times[k]/1000;tn=times[k+1]/1000 if k<49 else torch.zeros_like(t)
            qs=(1-t)*x+t*noises[k-17];qy=qs+y-x
            with torch.no_grad():
                run.intercept=False
                # Match the retained original forward's backend preference;
                # correction derivatives separately use the math backend.
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]):
                    v=run.model(hidden_states=torch.cat([qs,qs,qy,qy]),timestep=times[k].expand(4),
                        encoder_hidden_states=seq,pooled_projections=pool,return_dict=False)[0]
                run.intercept=True
                vs0,vss,vy0,vya=v.chunk(4)
                raw=(t-tn)*((vs0+3.5*(vss-vs0))-(vy0+13.5*(vya-vy0)))
                base=(y+raw).detach();previous=run.sc.logits_image(run.sc.image(y),run.text)[0]
                base_logs=run.sc.logits_image(run.sc.image(base),run.text)[0]
                before=float(previous[1]-previous[0]);mb=float(base_logs[1]-base_logs[0]);gain=mb-before
                floor=before+.9*max(gain,0)+min(gain,0)-.02
            info=dict(step=k,raw_norm=float(raw.norm()),task_previous=before,task_raw=mb,task_floor=floor,accepted=False)
            if mode!='BASE':
                point=base.detach().requires_grad_(True);r=run.residual(point,eps,zx,Q);energy=.5*r.square().sum()
                g=torch.autograd.grad(energy,point,retain_graph=True)[0].detach()
                radius=.5*float(raw.norm());lam=radius/max(float(g.norm()),1e-20)
                def jt(w):return torch.autograd.grad(r,point,grad_outputs=w,retain_graph=True)[0].detach()
                def jv(v):return torch.func.jvp(lambda q:run.residual(q,eps,zx,Q),(base,),(v,))[1].detach()
                def op(v):return v+lam*jt(jv(v))
                correction,ci=cg(op,-lam*g,6)
                pp=base.detach().requires_grad_(True);ll=run.sc.logits_image(run.sc.image(pp),run.text)[0]
                gt=torch.autograd.grad(ll[1]-ll[0],pp)[0].detach()
                gap=floor-mb-float((gt*correction).sum())
                if gap>0:
                    response,_=cg(op,gt,6);correction+=gap/max(float((gt*response).sum()),1e-20)*response
                correction*=min(1.,radius/max(float(correction.norm()),1e-20))
                energy0=float(energy);trials=[];new=base
                for retry in range(7):
                    candidate=base+(2.**-retry)*correction
                    with torch.no_grad():
                        rr=run.residual(candidate,eps,zx,Q);ee=float(.5*rr.square().sum())
                        im=run.sc.image(candidate);l=run.sc.logits_image(im,run.text)[0];mm=float(l[1]-l[0])
                    passed=ee<energy0-1e-10 and mm>=floor
                    trials.append(dict(scale=2.**-retry,energy=ee,margin=mm,passed=passed))
                    if passed:new=candidate.detach();info['accepted']=True;break
                info.update(observer_raw=energy0,observer_after=trials[-1]['energy'] if info['accepted'] else energy0,
                    correction_norm=float((new-base).norm()),trials=trials,solver=ci)
                del point,r,energy,g,jt,jv,op,pp,ll,gt,correction
                y=new
            else:y=base
            y=y.detach();states.append(y.cpu());rows.append(info)
            if k in [25,33,41,49]:
                with torch.no_grad():save_image(run.sc.image(y),dest/f'step{k}.png')
            write(dest/'progress.json',dict(rows=rows))
            print('FULL',mode,k,info['accepted'],round(time.monotonic()-start,2),flush=True)
        with torch.no_grad():
            final_image=run.sc.image(y);l=run.sc.logits_image(final_image,run.text)[0]
            endpoint=dict(task_margin=float(l[1]-l[0]),target_logit=float(l[1]),source_logit=float(l[0]))
        if mode=='BASE':
            saved=torch.load(HERE/'outputs/natural_v1/bear__00/baseline.pt',weights_only=True)['y'].to(run.device)
            endpoint['retained_baseline_relative_error']=float((y-saved).norm()/saved.norm())
            assert endpoint['retained_baseline_relative_error']<1e-5,endpoint
        save_image(final_image,dest/'final.png');torch.save(dict(states=torch.stack(states),final=y.cpu()),dest/'trajectory.pt')
        write(dest/'complete.json',dict(mode=mode,rows=rows,endpoint=endpoint,accepted=sum(r['accepted'] for r in rows),seconds=time.monotonic()-start))
    cols=[('Source',source_image)]+[(m,out/m/'final.png') for m in ['BASE','TEXT','INDUCED']]
    panel(cols,out/'comparison.png',title='Real FlowEdit clean-state feedback; one development source',size=320)
    print('FULL_PILOT_COMPLETE',flush=True)


if __name__=='__main__':
    with sdpa_kernel(SDPBackend.MATH):main()
