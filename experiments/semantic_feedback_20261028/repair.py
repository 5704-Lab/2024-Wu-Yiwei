"""Finite latent feedback pilot, with an explicit offline oracle comparison.

Automatic observers see x, current y, text and original frozen models. ORACLE
uses known legal target displacements ONLY to locate observer/action bottlenecks.
"""
import argparse
import hashlib
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend

from extract import Extractor,HERE,patch_pool
from common import seed_setup,write,save_image,panel


class Captured(Exception):pass


def vector(h):
    h=F.normalize(h,dim=-1)
    return h.flatten()/math.sqrt(h.shape[-2])


def cached_z(record,layer=11):
    p=record['sd3']['0.35'];s=p['names'].index('source');a=p['names'].index('shape')
    return .5*(vector(p['layers'][layer][s:s+1])+vector(p['layers'][layer][a:a+1]))


class Repair(Extractor):
    def __init__(self):
        super().__init__()
        for h in self.handles:h.remove()
        self.capture_grad=None
        self.intercept=True
        def stop(module,inputs,output):
            if not self.intercept:return
            self.capture_grad=patch_pool(output[1]);raise Captured()
        self.stop_handle=self.model.transformer_blocks[11].register_forward_hook(stop)
        names=['source','shape']
        self.seq=torch.cat([self.emb['conditions'][n]['seq'] for n in names]).to(self.device)
        self.pool=torch.cat([self.emb['conditions'][n]['pool'] for n in names]).to(self.device)

    def observation(self,y,eps):
        self.capture_grad=None
        q=((1-.35)*y+.35*eps).expand(2,-1,-1,-1)
        try:
            self.model(hidden_states=q,timestep=torch.full((2,),350.,device=self.device),
                encoder_hidden_states=self.seq,pooled_projections=self.pool,return_dict=False)
        except Captured:pass
        h=self.capture_grad;self.capture_grad=None
        assert h is not None
        return .5*(vector(h[0:1])+vector(h[1:2]))

    def residual(self,y,eps,zx,Q):
        r=self.observation(y,eps)-zx
        return r-Q@(Q.T@r) if Q.shape[1] else r


def basis(columns,device):
    a=torch.stack(columns,-1).double();u,s,_=torch.linalg.svd(a,full_matrices=False)
    keep=s>max(float(s.max())*1e-4,1e-8)
    return u[:,keep].float().to(device)


def cg(op,rhs,count=6):
    solution=torch.zeros_like(rhs);r=rhs.clone();p=r.clone()
    rr=(r.double()*r.double()).sum();initial=float(rr.sqrt());used=0
    for k in range(count):
        if float(rr.sqrt())<max(initial*1e-3,1e-12):break
        v=op(p);den=(p.double()*v.double()).sum()
        assert float(den)>0,'Non-positive GN operator'
        alpha=(rr/den).float();solution+=alpha*p;r-=alpha*v
        nxt=(r.double()*r.double()).sum();p=r+(nxt/rr).float()*p;rr=nxt;used=k+1
    return solution,dict(iterations=used,relative_residual=float(rr.sqrt())/max(initial,1e-20))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',default='dataset_v2')
    ap.add_argument('--features',default='features_v2');ap.add_argument('--induced',default='induced_v2')
    ap.add_argument('--scenes',nargs='+',type=int,default=[0,1,8,9]);ap.add_argument('--steps',type=int,default=6)
    ap.add_argument('--modes',nargs='+',default=['TEXT','INDUCED','ORACLE']);ap.add_argument('--tag',default='repair_v2')
    ap.add_argument('--solver',choices=['gd','gn'],default='gn');ap.add_argument('--cg-steps',type=int,default=6)
    ap.add_argument('--task-mode',choices=['strict','goalset'],default='goalset');ap.add_argument('--backtracks',type=int,default=9)
    args=ap.parse_args();seed_setup();run=Repair();out=HERE/'outputs'/args.tag;out.mkdir(exist_ok=True)
    protocol=dict(args=vars(args),script_sha256=hashlib.sha256(open(__file__,'rb').read()).hexdigest(),
        stage='soft residual restoration under finite SigLIP target-minus-source floor',
        max_step_relative_to_initial_edit=.08,total_radius_relative_to_initial_edit=.35,
        siglip_floor_drop=.02,goalset_progress_retention=.9,observer_time=.35,layer=11,spatial_mask=False,
        oracle='Known legal color and morphology target columns, forbidden for deployment; diagnostic comparison only')
    write(out/'protocol.json',protocol);start=time.monotonic();rows=[]
    for scene in args.scenes:
        folder=HERE/'outputs'/args.features/f'scene{scene:02d}'
        records={n:torch.load(folder/f'{n}.pt',weights_only=True) for n in ['source','color_only','valid','wrong_head','wrong_heading']}
        x=records['source']['y'].to(run.device);valid=records['valid']['y'].to(run.device)
        zx=cached_z(records['source']).to(run.device)
        p=records['source']['sd3']['0.35'];inds=p['names']
        atext=vector(p['layers'][11][inds.index('shape'):inds.index('shape')+1])-vector(p['layers'][11][inds.index('source'):inds.index('source')+1])
        columns=dict(TEXT=[atext],ORACLE=[cached_z(records['color_only'])-zx.cpu(),cached_z(records['valid'])-zx.cpu()])
        if 'INDUCED' in args.modes:
            columns['INDUCED']=[]
            for tau in [.35,.65]:
                item=torch.load(HERE/'outputs'/args.induced/f'scene{scene:02d}/shape_{tau}.pt',weights_only=True)
                columns['INDUCED'].append(cached_z(item)-zx.cpu())
        eps=torch.randn(x.shape,generator=torch.Generator('cpu').manual_seed(19571028+scene)).to(run.device)
        with torch.no_grad():
            parity=float((run.observation(x,eps)-zx).norm())
            reference_image=run.sc.image(valid)
            source_logs=run.sc.logits_image(run.sc.image(x),run.text)[0]
            source_margin=float(source_logs[1]-source_logs[0])
        assert parity<2e-5,parity
        for name in ['valid','wrong_head','wrong_heading']:
            initial=records[name]['y'].to(run.device)
            with torch.no_grad():
                image0=run.sc.image(initial)
                logs=run.sc.logits_image(image0,run.text)[0]
                initial_margin=float(logs[1]-logs[0])
                floor=(initial_margin-.02 if args.task_mode=='strict'
                       else source_margin+.9*max(initial_margin-source_margin,0)-.02)
            editnorm=float((initial-x).norm());radius=.35*editnorm;step=.08*editnorm
            for mode in args.modes:
                dest=out/f'scene{scene:02d}'/name/mode;dest.mkdir(parents=True,exist_ok=True)
                if (dest/'complete.json').exists():
                    rows.append(json.loads((dest/'complete.json').read_text()));continue
                Q=basis(columns[mode],run.device);y=initial.clone();history=[];fdcheck=None
                for k in range(args.steps):
                    point=y.detach().requires_grad_(True)
                    residual=run.residual(point,eps,zx,Q)
                    energy=.5*residual.square().sum()
                    g=torch.autograd.grad(energy,point,retain_graph=True)[0].detach()
                    if float(g.norm())<1e-10:
                        history.append(dict(step=k,energy=float(energy),stationary=True));break
                    # Task halfspace at CURRENT CLEAN state, then finite checks.
                    pp=y.detach().requires_grad_(True);image=run.sc.image(pp)
                    ll=run.sc.logits_image(image,run.text)[0];margin=ll[1]-ll[0]
                    gt=torch.autograd.grad(margin,pp)[0].detach()
                    solver_info={}
                    if args.solver=='gn':
                        lam=step/max(float(g.norm()),1e-20)
                        def jt(w):return torch.autograd.grad(residual,point,grad_outputs=w,retain_graph=True)[0].detach()
                        def jv(v):return torch.func.jvp(lambda q:run.residual(q,eps,zx,Q),(y,),(v,))[1].detach()
                        def op(v):return v+lam*jt(jv(v))
                        d,solver_info['primary']=cg(op,-lam*g,args.cg_steps)
                        if k==0 and name=='wrong_head':
                            u=g/g.norm();j=jv(u)
                            left=float((residual.detach()*j).sum());right=float((g*u).sum())
                            solver_info['adjoint_relative_error']=abs(left-right)/max(abs(right),1e-20)
                    else:d=-g*(step/max(float(g.norm()),1e-20))
                    work=float((gt*d).sum())
                    required=0. if args.task_mode=='strict' else min(0.,floor-float(margin.detach()))
                    if work<required:
                        if args.solver=='gn':
                            response,solver_info['task']=cg(op,gt,args.cg_steps)
                        else:response=gt
                        d+=response*((required-work)/max(float((gt*response).sum()),1e-20))
                    dn=float(d.norm());d*=min(1.,step/max(dn,1e-20))
                    with torch.no_grad():
                        if k==0 and name=='wrong_head':
                            e=1e-1
                            u=g/g.norm()
                            plus=.5*run.residual(y+e*u,eps,zx,Q).square().sum()
                            minus=.5*run.residual(y-e*u,eps,zx,Q).square().sum()
                            fd=float((plus-minus)/(2*e));ad=float(g.norm())
                            fdcheck=dict(relative_error=abs(fd-ad)/max(ad,1e-20),finite_difference=fd,automatic=ad)
                    energy0=float(energy);attempts=[];accepted=False
                    for j in range(args.backtracks):
                        candidate=y+(2.**-j)*d
                        offset=candidate-initial
                        candidate=initial+offset*min(1.,radius/max(float(offset.norm()),1e-20))
                        with torch.no_grad():
                            rr=run.residual(candidate,eps,zx,Q);ee=float(.5*rr.square().sum())
                            im=run.sc.image(candidate);logs=run.sc.logits_image(im,run.text)[0];mm=float(logs[1]-logs[0])
                        passed=ee<energy0-1e-10 and mm>=floor
                        attempts.append(dict(scale=2.**-j,energy=ee,margin=mm,passed=passed))
                        if passed:y=candidate.detach();accepted=True;break
                    history.append(dict(step=k,energy=energy0,task_margin=float(margin.detach()),
                        gradient_norm=float(g.norm()),accepted=accepted,trials=attempts,solver=solver_info))
                    if args.solver=='gn':del jt,jv,op
                    del point,residual,energy,g,pp,image,ll,margin,gt,d
                    if not accepted:break
                with torch.no_grad():
                    final_image=run.sc.image(y)
                    final_energy=float(.5*run.residual(y,eps,zx,Q).square().sum())
                    delta=y-initial;repair=valid-initial;revert=x-initial
                    cosine=lambda v:float((delta*v).sum()/max(float(delta.norm()*v.norm()),1e-20))
                    imdist=lambda a,b:float((a-b).square().mean().sqrt())
                    result=dict(scene=scene,variant=name,mode=mode,parity=parity,fdcheck=fdcheck,
                        observer_start=history[0]['energy'],observer_final=final_energy,
                        target_rgb_rmse_before=imdist(image0,reference_image),target_rgb_rmse_after=imdist(final_image,reference_image),
                        latent_change=float(delta.norm()),edit_norm=editnorm,
                        repair_chord_cosine=cosine(repair),source_reversion_cosine=cosine(revert),task_floor=floor,
                        accepted_steps=sum(h.get('accepted',False) for h in history),history=history)
                torch.save(dict(initial=initial.cpu(),final=y.cpu()),dest/'states.pt')
                save_image(final_image,dest/'final.png')
                panel([('Current',image0),('Feedback',final_image),('Known legal (evaluation only)',reference_image)],dest/'comparison.png',title=f'{scene} {name} {mode}')
                write(dest/'complete.json',result);rows.append(result)
                print('REPAIR',scene,name,mode,result['accepted_steps'],round(result['observer_final']/max(result['observer_start'],1e-20),3),
                    round(result['target_rgb_rmse_after']-result['target_rgb_rmse_before'],6),round(time.monotonic()-start,2),flush=True)
    write(out/'summary.json',dict(protocol=protocol,rows=rows,seconds=time.monotonic()-start,
        peak_memory_bytes=torch.cuda.max_memory_allocated()))
    print('REPAIR_COMPLETE',flush=True)


if __name__=='__main__':
    with sdpa_kernel(SDPBackend.MATH):main()
