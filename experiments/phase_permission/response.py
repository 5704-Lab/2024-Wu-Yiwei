"""Fixed-query condition/state/CFG response decomposition on real FlowEdit states.

Diagnostic only. No phase component is automatically called semantically permitted.
"""
import sys
import json
import time
import math
import torch
import torch.nn.functional as F
from types import SimpleNamespace
from torch.nn.attention import SDPBackend,sdpa_kernel
from diffusers import SD3Transformer2DModel
from common import ROOT,OUT,PAIRS,write,seed_setup,code_hashes,orthogonal_basis
from real import Experiment


def geometry_design(phase,reference):
    n=reference.shape[-1]
    f=(torch.arange(n,device=reference.device,dtype=reference.dtype)+.5)/n
    yy,xx=torch.meshgrid(f,f,indexing='ij')
    basis=[torch.ones_like(xx),torch.cos(math.pi*xx),torch.cos(math.pi*yy),torch.cos(math.pi*xx)*torch.cos(math.pi*yy)]
    grid=torch.stack([2*xx-1,2*yy-1],-1)[None]
    cols=[];eps=.025
    with torch.no_grad():
        for axis in [0,1]:
            for field in basis:
                offset=torch.zeros_like(grid);offset[0,:,:,axis]=2*eps/n*field
                plus=F.grid_sample(reference,grid-offset,mode='bilinear',padding_mode='reflection',align_corners=False)
                minus=F.grid_sample(reference,grid+offset,mode='bilinear',padding_mode='reflection',align_corners=False)
                cols.append(((phase(plus)-phase(minus))/(2*eps)).flatten())
    return torch.stack(cols,1)


def main():
    seed_setup();exp=Experiment()
    sys.path.insert(0,str(ROOT/'vendor/flowedit'))
    from FlowEdit_utils import calc_v_sd3
    model=SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',torch_dtype=torch.float32,
        local_files_only=True).eval().requires_grad_(False).to(exp.device)
    pipe=SimpleNamespace(transformer=model,do_classifier_free_guidance=True)
    times=torch.tensor(json.loads((ROOT/'experiments/stage1_full/inputs/manifest.json').read_text())['timesteps'],device=exp.device)
    rows=[]
    for pair in PAIRS:
        ctx=exp.setup(pair);x=ctx['x'];phase=ctx['phase'];app=ctx['app']
        saved=torch.load(ROOT/'experiments/flowedit_frequency_trajectory/outputs'/pair/'trajectory.pt',weights_only=True)
        es,et=[tuple(t.to(exp.device) for t in emb) for emb in ctx['data']['embeddings']]
        seq=torch.cat([es[1],es[0],et[1],et[0]]);pool=torch.cat([es[3],es[2],et[3],et[2]])
        seqs=torch.cat([es[1],es[0],es[1],es[0]]);pools=torch.cat([es[3],es[2],es[3],es[2]])
        gen=torch.Generator('cpu').manual_seed(0);noise=[torch.randn(x.shape,generator=gen).to(exp.device) for _ in range(33)]
        for step in [17,33,47]:
            start=time.monotonic();y=saved['states'][step].to(exp.device);t=times[step]/1000
            qs=(1-t)*x+t*noise[step-17];qa=qs+y-x
            with torch.no_grad(),sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]):
                vs,va=calc_v_sd3(pipe,torch.cat([qs,qs,qa,qa]),seq,pool,3.5,13.5,times[step])
                vshi,vsa=calc_v_sd3(pipe,torch.cat([qs,qs,qa,qa]),seqs,pools,13.5,13.5,times[step])
            fields=dict(total=-(va-vs),condition=-(va-vsa),state=-(vsa-vshi),cfg=-(vshi-vs))
            with torch.no_grad():im=exp.sc.image(y);obs=phase(im)
            theta,_,fit=app.fit(phase,obs,maxiter=80)
            v=app.jacobian(phase,theta);uv,sv,rankv=orthogonal_basis(v)
            ref=app(theta).detach();m=geometry_design(phase,ref)
            qm=m-uv@(uv.T@m);uc,sc,rankc=orthogonal_basis(qm)
            components={};omegas={}
            for name,direction in fields.items():
                with sdpa_kernel(SDPBackend.MATH):
                    _,omega=torch.func.jvp(lambda z:phase(exp.sc.image(z)).flatten(),(y,),(direction,))
                omega=omega.detach();omegas[name]=omega
                wa=uv@(uv.T@omega);q=omega-wa;wg=uc@(uc.T@q);wu=q-wg
                energy=float(omega.square().sum())
                components[name]=dict(norm=energy**.5,appearance_fraction=float(wa.square().sum())/max(energy,1e-30),
                    geometry_fraction=float(wg.square().sum())/max(energy,1e-30),unknown_fraction=float(wu.square().sum())/max(energy,1e-30),
                    geometry_coefficients=torch.linalg.lstsq(qm.double(),q.double()).solution.cpu().tolist())
            closure=float((omegas['total']-omegas['condition']-omegas['state']-omegas['cfg']).norm()/omegas['total'].norm())
            row=dict(pair=pair,step=step,components=components,phase_additive_error=closure,
                     field_additive_error=float((fields['total']-fields['condition']-fields['state']-fields['cfg']).norm()/fields['total'].norm()),
                     appearance_rank=rankv,geometry_rank=rankc,geometry_singular_values=sc,appearance_fit=fit,
                     geometry_scope='8 fixed smooth displacement modes; not dense true optical flow',seconds=time.monotonic()-start)
            rows.append(row);write(OUT/'response.json',dict(rows=rows,code=code_hashes()))
            print(json.dumps(dict(pair=pair,step=step,closure=closure,components=components)),flush=True)
    print('RESPONSE_COMPLETE',flush=True)


if __name__=='__main__':main()
