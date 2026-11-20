"""Real SD3 clean-state phase correction in a measured 3D search subspace.

This is an explicitly restricted Gauss--Newton solver, not the full latent QP.
Appearance is fitted with bounds to the fixed source, then locally profiled.
Every accepted proposal is decoded, refitted, and task-checked before use.
"""
import argparse
import gc
import json
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

from common import (ROOT,OUT,PAIRS,CleanScorer,Phase,Appearance,write,seed_setup,
                    save_image,panel,code_hashes,orthogonal_basis,source_rgb)

CONFIG=dict(order=2,fit_iterations=25,subspace=['negative phase gradient','target gradient','base increment'],
    radius_ratio=.20,task_positive_gain_retention=.5,task_tolerance=.01,
    strengths=[.3,1.],full_strength=1.,max_backtracks=5,seed=0,
    geometry_permission='none; bus and crown are out-of-family stress tests',
    optimizer='exact reduced convex quadratic with ball and task halfspace; finite candidate recheck',
    source_cfg=3.5,target_cfg=13.5,T=50,n_max=33,n_min=0,n_avg=1)


def reduced_qp(h,f,g,beta,radius):
    h,f,g=np.asarray(h,np.float64),np.asarray(f,np.float64),np.asarray(g,np.float64)
    if beta>radius*np.linalg.norm(g)+1e-10:
        return None,dict(feasible=False)
    def solution(mu):
        mat=h+mu*np.eye(len(f))
        v=np.linalg.solve(mat,-f);hg=np.linalg.solve(mat,g)
        nu=max(0.,(beta-g@v)/max(g@hg,1e-30))
        return v+nu*hg,nu
    z,nu=solution(0.);mu=0.
    if np.linalg.norm(z)>radius:
        lo,hi=0.,1.
        for _ in range(80):
            if np.linalg.norm(solution(hi)[0])<=radius:break
            hi*=2
        for _ in range(60):
            mid=(lo+hi)/2
            if np.linalg.norm(solution(mid)[0])>radius:lo=mid
            else:hi=mid
        mu=hi;z,nu=solution(mu)
    return z,dict(feasible=True,ball_dual=mu,task_dual=nu,
        stationarity=float(np.linalg.norm((h+mu*np.eye(len(f)))@z+f-nu*g)),
        norm=float(np.linalg.norm(z)),task_slack=float(g@z-beta))


class Experiment:
    def __init__(self):
        self.device=torch.device('cuda:0')
        self.sc=CleanScorer(self.device,torch.float32,'native')
        self.context={}

    def setup(self,pair):
        if pair in self.context:return self.context[pair]
        data=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True)
        x=data['x'].to(self.device)
        short={'bear__00':['A black bear.','A brown bear.'],
               'brown_owl__00':['A white owl.','A brown owl.'],
               'bus__00':['A modern SUV.','A yellow van.'],
               'cat_crown__00':['A cat wearing a top hat.','A cat wearing a crown.']}
        texts=[data['record']['target_prompt'],data['record']['source_prompt'],*short[pair]]
        embeddings,lengths=self.sc.encode(texts)
        with torch.no_grad():src=self.sc.image(x)
        phase=Phase(src)
        app=Appearance(phase.source_small,CONFIG['order'])
        ctx=dict(data=data,x=x,source=src.detach(),phase=phase,app=app,e=embeddings,lengths=lengths,texts=texts)
        self.context[pair]=ctx
        save_image(src,OUT/'real'/pair/'source_vae.png')
        return ctx

    def metrics_image(self,ctx,im):
        with torch.no_grad():
            scores=self.sc.logits_image(im,ctx['e'])[0].cpu().tolist()
            obs=ctx['phase'](im)
        theta,ref,fit=ctx['app'].fit(ctx['phase'],obs,maxiter=CONFIG['fit_iterations'])
        return dict(full_target=scores[0],full_source=scores[1],short_target=scores[2],short_source=scores[3],
                    full_margin=scores[0]-scores[1],short_margin=scores[2]-scores[3],
                    raw_phase=float((obs-ctx['phase'].source_obs).norm()),
                    permitted_phase=float((obs-ref).norm()),appearance_fit=fit,
                    rgb_rmse=float((im-ctx['source']).square().mean().sqrt()),
                    **ctx['phase'].diagnostics(im))

    def metrics(self,ctx,y):
        with torch.no_grad():im=self.sc.image(y)
        return self.metrics_image(ctx,im),im

    def correct(self,ctx,y,delta,method='PERMIT',strength=1.,audit=False):
        start=time.monotonic();phase,app=ctx['phase'],ctx['app']
        base=(y+delta).detach()
        with torch.no_grad():
            before=float(self.sc.logits(y,ctx['e'][:1])[0,0])
            im0=self.sc.image(base)
            obs0=phase(im0).detach()
            target0=float(self.sc.logits_image(im0,ctx['e'][:1])[0,0])
        if method=='PERMIT':
            theta,ref,fit=app.fit(phase,obs0,maxiter=CONFIG['fit_iterations'])
            v=app.jacobian(phase,theta)
            # Tikhonov penalty in bound-normalized increments.
            v=v*app.bound[None]
            gram=v.T.double()@v.double()
            ridge=max(float(torch.trace(gram))/len(gram)*1e-3,1e-9)
            factor=torch.linalg.cholesky(gram+ridge*torch.eye(app.n,device=v.device,dtype=torch.float64))
            def profile(r):
                coeff=torch.cholesky_solve((v.T.double()@r.double()).reshape(app.n,-1),factor)
                return r-v@coeff.to(r.dtype).reshape(app.n,*r.shape[1:])
        else:
            theta=torch.zeros(app.n,device=self.device);ref=phase.source_obs
            fit=dict(success=True,iterations=0);ridge=0.
            def profile(r):return r
        err=(obs0-ref).flatten()
        perr=profile(err)
        with sdpa_kernel(SDPBackend.MATH):
            point=base.detach().requires_grad_(True)
            im=self.sc.image(point)
            obs=phase(im).flatten()
            target=self.sc.logits_image(im,ctx['e'][:1])[0,0]
            b=torch.autograd.grad(obs,point,grad_outputs=perr,retain_graph=True)[0].detach()
            g=torch.autograd.grad(target,point)[0].detach()
        del point,im,obs,target
        vectors=[]
        for candidate in [-b,g,delta]:
            vv=candidate.flatten().clone()
            for old in vectors:vv-=torch.dot(vv,old)*old
            if float(vv.norm())>1e-6*max(float(candidate.norm()),1e-20):vectors.append(vv/vv.norm())
        basis=torch.stack(vectors,dim=1)
        def observation(z):return phase(self.sc.image(z)).flatten()
        columns=[]
        for i in range(basis.shape[1]):
            direction=basis[:,i].reshape_as(base)
            with sdpa_kernel(SDPBackend.MATH):
                _,jv=torch.func.jvp(observation,(base,),(direction,))
            columns.append(jv.detach())
        j=torch.stack(columns,dim=1)
        pj=profile(j)
        gramj=j.T.double()@pj.double()
        gramj=(gramj+gramj.T)/2
        scale=max(float(torch.trace(gramj))/len(gramj),1e-12)
        lam=strength/scale
        f=lam*(j.T.double()@perr.double())
        h=torch.eye(len(gramj),device=self.device,dtype=torch.float64)+lam*gramj
        gs=basis.T.double()@g.flatten().double()
        threshold=before+CONFIG['task_positive_gain_retention']*max(target0-before,0.)+min(target0-before,0.)
        radius=CONFIG['radius_ratio']*float(delta.norm())
        z,qp=reduced_qp(h.cpu().numpy(),f.cpu().numpy(),gs.cpu().numpy(),threshold-target0,radius)
        checks={}
        if audit:
            eps=.005*float(delta.norm())
            with torch.no_grad():fd=(observation(base+eps*basis[:,0].reshape_as(base))-observation(base-eps*basis[:,0].reshape_as(base)))/(2*eps)
            fd_error=float((fd-j[:,0]).norm()/j[:,0].norm().clamp_min(1e-20))
            reciprocity=abs(float(torch.dot(j[:,0].double(),perr.double())-torch.dot(basis[:,0].double(),b.flatten().double())))/max(abs(float(torch.dot(j[:,0].double(),perr.double()))),1e-20)
            checks=dict(jvp_finite_difference_relative_error=fd_error,jvp_vjp_reciprocity_error=reciprocity,
                        basis_orthogonality_error=float((basis.T@basis-torch.eye(len(gramj),device=self.device)).abs().max()))
            if fd_error>.08 or reciprocity>1e-3:raise RuntimeError(f'Derivative gate failed: {checks}')
        rawloss=float(err.square().sum())
        accepted=False;new=base;trials=[];newtheta=theta
        if z is not None:
            correction=(basis@torch.tensor(z,device=self.device,dtype=base.dtype)).reshape_as(base)
            for backtrack in range(CONFIG['max_backtracks']):
                alpha=.5**backtrack
                proposal=(base+alpha*correction).detach()
                with torch.no_grad():
                    image=self.sc.image(proposal)
                    newobs=phase(image)
                    task=float(self.sc.logits_image(image,ctx['e'][:1])[0,0])
                if method=='PERMIT':
                    th,rf,ft=app.fit(phase,newobs,initial=theta,maxiter=CONFIG['fit_iterations'])
                else:th,rf,ft=theta,ref,fit
                loss=float((newobs-rf).square().sum())
                cnorm=float((proposal-base).norm())
                # Original nonlinear profiled merit includes the state movement penalty.
                merit_change=.5*cnorm**2+.5*lam*(loss-rawloss)
                passed=bool(np.isfinite(loss) and task>=threshold-CONFIG['task_tolerance'] and
                            loss<rawloss and merit_change<=1e-7 and cnorm<=radius*(1+1e-5))
                trials.append(dict(alpha=alpha,task=task,residual=loss**.5,merit_change=merit_change,
                                   accepted=passed,fit=ft))
                if passed:new=proposal;newtheta=th;accepted=True;break
        with torch.no_grad():
            finaltask=float(self.sc.logits(new,ctx['e'][:1])[0,0])
        info=dict(method=method,strength=strength,accepted=accepted,fit_before=fit,task_before=before,
                  task_base=target0,task_threshold=threshold,task_after=finaltask,lambda_actual=lam,ridge=ridge,
                  phase_before=rawloss**.5,phase_after=trials[-1]['residual'] if accepted else rawloss**.5,
                  correction_norm=float((new-base).norm()),base_increment_norm=float(delta.norm()),
                  increment_ratio=float((new-y).norm()/delta.norm().clamp_min(1e-20)),
                  radius=radius,qp=qp,trials=trials,derivative_checks=checks,
                  seconds=time.monotonic()-start,subspace_dimension=len(gramj),theta=newtheta.detach().cpu().tolist())
        return new.detach(),info


def local(args):
    exp=Experiment();rows=[]
    dest=OUT/'local';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'results.json').exists():rows=json.loads((dest/'results.json').read_text())['rows']
    for pair in args.pairs:
        ctx=exp.setup(pair)
        saved=torch.load(ROOT/'experiments/flowedit_frequency_trajectory/outputs'/pair/'trajectory.pt',weights_only=True)
        for step in args.steps:
            y=saved['states'][step].to(exp.device);delta=(saved['states'][step+1].to(exp.device)-y)
            before,bim=exp.metrics(ctx,y+delta)
            cols=[('Original FlowEdit local',bim)]
            for method,strength in [('RAW_PHASE',1.),('PERMIT',.3),('PERMIT',1.)]:
                target=dest/pair/f'step{step}_{method}_{strength}.pt'
                if target.exists():
                    payload=torch.load(target,weights_only=False);new=payload['state'].to(exp.device)
                else:
                    new,info=exp.correct(ctx,y,delta,method,strength,audit=True)
                    after,image=exp.metrics(ctx,new)
                    row=dict(pair=pair,step=step,before=before,after=after,**info)
                    rows.append(row)
                    target.parent.mkdir(parents=True,exist_ok=True)
                    torch.save(dict(state=new.cpu(),row=row),target)
                    save_image(image,target.with_suffix('.png'))
                    write(dest/'results.json',dict(config=CONFIG,code=code_hashes(),rows=rows))
                    print(json.dumps(dict(pair=pair,step=step,method=method,strength=strength,accepted=info['accepted'],
                        phase_before=info['phase_before'],phase_after=info['phase_after'],seconds=info['seconds'])),flush=True)
                with torch.no_grad():cols.append((f'{method} {strength}',exp.sc.image(new)))
            panel(cols,dest/pair/f'step{step}_comparison.png',title=pair+f' step {step}')
            gc.collect();torch.cuda.empty_cache()
    print('LOCAL_COMPLETE',flush=True)


def full(args):
    import sys
    sys.path.insert(0,str(ROOT/'vendor/flowedit'))
    from FlowEdit_utils import calc_v_sd3
    from diffusers import SD3Transformer2DModel
    exp=Experiment()
    model=SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',
        torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to(exp.device)
    pipe=SimpleNamespace(transformer=model,do_classifier_free_guidance=True)
    meta=json.loads((ROOT/'experiments/stage1_full/inputs/manifest.json').read_text())
    times=torch.tensor(meta['timesteps'],device=exp.device)
    for pair in args.pairs:
        ctx=exp.setup(pair);x=ctx['x'];data=ctx['data']
        es,et=[tuple(t.to(exp.device) for t in emb) for emb in data['embeddings']]
        seq=torch.cat([es[1],es[0],et[1],et[0]]);pool=torch.cat([es[3],es[2],et[3],et[2]])
        gen=torch.Generator('cpu').manual_seed(0)
        noises=[torch.randn(x.shape,generator=gen).to(exp.device) for _ in range(33)]
        saved=torch.load(ROOT/'experiments/flowedit_frequency_trajectory/outputs'/pair/'trajectory.pt',weights_only=True)
        for method in args.methods:
            dest=OUT/'full'/pair/method;dest.mkdir(parents=True,exist_ok=True)
            if (dest/'trajectory.pt').exists():continue
            start=time.monotonic();y=x.clone();states=[y.cpu()];rows=[]
            norm_rows=None
            if method=='NORM_CONTROL':
                norm_rows=json.loads((OUT/'full'/pair/'PERMIT'/'metrics.json').read_text())['steps']
            # Checkpoint resume preserves noise index, state, and recorded corrections.
            checkpoint=dest/'checkpoint.pt'
            if checkpoint.exists():
                old=torch.load(checkpoint,weights_only=False);states=old['states'];rows=old['rows'];y=states[-1].to(exp.device)
            for k in range(17+len(rows),50):
                t=times[k]/1000;tn=times[k+1]/1000 if k<49 else torch.zeros_like(t);h=float(t-tn)
                qs=(1-t)*x+t*noises[k-17];qa=qs+y-x
                with torch.no_grad():
                    vs,va=calc_v_sd3(pipe,torch.cat([qs,qs,qa,qa]),seq,pool,3.5,13.5,times[k])
                    delta=-h*(va-vs)
                if method=='BASE':new=y+delta;info=dict(accepted=True,increment_ratio=1.,correction_norm=0.)
                elif method=='NORM_CONTROL':
                    ratio=norm_rows[k-17]['increment_ratio'];new=y+ratio*delta
                    info=dict(accepted=True,increment_ratio=ratio,correction_norm=float(((ratio-1)*delta).norm()))
                else:new,info=exp.correct(ctx,y,delta,method,CONFIG['full_strength'],audit=k==17)
                y=new.detach();states.append(y.cpu());rows.append(dict(step=k,h=h,**info))
                torch.save(dict(states=states,rows=rows),checkpoint)
                if k in [17,25,33,41,49]:
                    with torch.no_grad():save_image(exp.sc.image(y),dest/f'step{k}.png')
                print(json.dumps(dict(pair=pair,method=method,step=k,accepted=info['accepted'],
                                      correction_norm=info['correction_norm'],seconds=time.monotonic()-start)),flush=True)
            met,im=exp.metrics(ctx,y)
            rel=float((y.cpu().double()-saved['final'].double()).norm()/saved['final'].double().norm())
            if method=='BASE' and rel>1e-6:raise RuntimeError(f'Original baseline mismatch: {rel}')
            result=dict(pair=pair,method=method,metrics=met,steps=rows,seconds=time.monotonic()-start,
                        baseline_relative_error=rel,config=CONFIG,code=code_hashes())
            save_image(im,dest/'final.png');write(dest/'metrics.json',result)
            torch.save(dict(final=y.cpu(),states=torch.stack(states),steps=rows),dest/'trajectory.pt')
            print(json.dumps(dict(pair=pair,method=method,finished=True,metrics=met,baseline_error=rel)),flush=True)
    print('FULL_COMPLETE',flush=True)


if __name__=='__main__':
    seed_setup();ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['local','full'],required=True)
    ap.add_argument('--pairs',nargs='+',default=PAIRS);ap.add_argument('--steps',nargs='+',type=int,default=[17,33,47])
    ap.add_argument('--methods',nargs='+',default=['BASE','PERMIT','RAW_PHASE','NORM_CONTROL'])
    args=ap.parse_args();write(OUT/f'protocol_{args.stage}.json',dict(args=vars(args),config=CONFIG,code=code_hashes()))
    with sdpa_kernel(SDPBackend.MATH):
        (local if args.stage=='local' else full)(args)
