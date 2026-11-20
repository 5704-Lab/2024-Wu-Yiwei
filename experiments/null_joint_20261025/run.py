"""Real paired conditional/null interventions and source-restoring controls.

No pretrained-model training, inverse trajectory, extra LLM, editing mask, or
SigLIP feedback. Conditional response is model-relative, not semantic truth.
"""
import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'experiments/joint_permission'))
from control import Run, cg, dot
sys.path.insert(0, str(ROOT/'experiments/phase_permission'))
from common import Phase, panel, save_image, seed_setup, write

CFG = dict(source_cfg=3.5, target_cfg=13.5, seed=0, cg_steps=6,
           task_cg_steps=6, gradient_step=1.2, radius=1.5, retention=.5,
           backtracks=5, kappa=40., probe_relative=.05, branch_direction_rtol=1e-3,patch_rank=2,
           adaptive_source_factor=10.,adaptive_cg_steps=24)


class Engine(Run):
    def __init__(self):
        super().__init__()
        self.times = torch.tensor(json.loads((ROOT/'experiments/stage1_full/inputs/manifest.json').read_text())['timesteps'], device=self.device)

    def setup(self, pair):
        ctx = super().setup(pair)
        ctx['phase'] = Phase(ctx['source'])
        attrs = (['A bear with black fur.', 'A bear with brown fur.'] if pair=='bear__00'
                 else ['An owl with white feathers.', 'An owl with brown feathers.'])
        if pair in ['bear__00', 'brown_owl__00']:
            ctx['attr'], _ = self.sc.encode(attrs)
        return ctx

    def query(self, ctx, y, k, noise, six=True):
        t = self.times[k]/1000
        tn = self.times[k+1]/1000 if k<49 else torch.zeros_like(t)
        qs = (1-t)*ctx['x']+t*noise
        qy = qs+y-ctx['x']
        with torch.no_grad():
            pred = self.model(hidden_states=torch.cat([qs,qs,qy,qy]),
                 timestep=self.times[k].expand(4), encoder_hidden_states=ctx['seq'],
                 pooled_projections=ctx['pool'], joint_attention_kwargs=None, return_dict=False)[0]
        vs0, vss, vy0, vya = pred.chunk(4)
        gs, ga = vss-vs0, vya-vy0
        vs = vs0+CFG['source_cfg']*gs
        va = vy0+CFG['target_cfg']*ga
        base = (t-tn)*(vs-va)
        out = dict(base=base, empty=(t-tn)*(vs0-vy0), source_guidance=gs)
        # Keep the first four queries exactly in original batch order.
        if six:
            seq = torch.cat([ctx['seq'][3:4],ctx['seq'][1:2]])
            pool = torch.cat([ctx['pool'][3:4],ctx['pool'][1:2]])
            with torch.no_grad():
                extra = self.model(hidden_states=torch.cat([qs,qy]),timestep=self.times[k].expand(2),
                    encoder_hidden_states=seq,pooled_projections=pool,joint_attention_kwargs=None,return_dict=False)[0]
            vsa, vys = extra.chunk(2)
            gc_source = vsa-vss
            gc_current = vya-vys
            state = (t-tn)*((vs0+CFG['source_cfg']*gs)-(vy0+CFG['source_cfg']*(vys-vy0)))
            condition = -(t-tn)*CFG['target_cfg']*gc_current
            cf = state+condition
            pieces = [out['empty'], -(t-tn)*CFG['target_cfg']*gc_source,
                      -(t-tn)*CFG['target_cfg']*((vys-vy0)-gs),
                      -(t-tn)*CFG['target_cfg']*(gc_current-gc_source),
                      (t-tn)*(CFG['source_cfg']-CFG['target_cfg'])*gs]
            out.update(cf=cf, task=(vys-vya).detach(), pieces=pieces,
                       decomposition_error=float((sum(pieces)-base).abs().max()))
        out['calibrated'] = ctx['x']+(qy-t*va)-(qs-t*vs)
        out['calibration_error'] = float((y+(t-tn)/t*(out['calibrated']-y)-(y+base)).abs().max())
        if torch.equal(y,ctx['x']):
            out['empty_initial_error'] = float(out['empty'].abs().max())
        return out

    def endpoint(self, ctx, y):
        with torch.no_grad():
            im = self.sc.image(y)
            scores = self.sc.logits_image(torch.cat([im,im.flip(-1)]),ctx['e'])
            result = dict(full_target=float(scores[0,0]), short_contrast=float((scores[:,2]-scores[:,3]).mean()),
                geometry=float(ctx['geometry'].loss(ctx['geometry'](im),CFG['kappa'])),
                rgb_rmse=float((im-ctx['source']).square().mean().sqrt()),
                background_top64_rmse=float((im[:,:,:64]-ctx['source'][:,:,:64]).square().mean().sqrt()),
                **ctx['phase'].diagnostics(im))
            if 'attr' in ctx:
                a = self.sc.logits_image(torch.cat([im,im.flip(-1)]),ctx['attr'])
                result['attribute_contrast'] = float((a[:,0]-a[:,1]).mean())
        return result,im

    def correct(self, ctx, y, delta, task, retention):
        """GN restoration with a conditional-velocity halfspace, finite acceptance."""
        geom=ctx['geometry'];base=(y+delta).detach();started=time.monotonic()
        with torch.no_grad():
            base_obs=geom(self.sc.image(base));oldloss=float(geom.loss(base_obs,CFG['kappa']))
        point=base.detach().requires_grad_(True)
        obs=geom(self.sc.image(point))
        weights=geom.gn_weights(obs,CFG['kappa'])[:,:,None]/obs.shape[-1]
        residual=(obs-geom.source).detach()
        def jt(v):
            return torch.autograd.grad(obs,point,grad_outputs=v,retain_graph=True)[0].detach()
        gradient=jt(weights*residual)
        lam=CFG['gradient_step']*float(delta.norm())/max(float(gradient.norm()),1e-12)
        def jv(v):
            _,j=torch.func.jvp(lambda z:geom(self.sc.image(z)),(base,),(v,))
            return j.detach()
        def op(v):
            return v+lam*jt(weights*jv(v))
        correction,solver=cg(op,-lam*gradient,CFG['cg_steps'])
        normalized_task=task/max(float(task.norm()),1e-12)
        original_progress=float(dot(normalized_task,delta))
        floor=retention*max(original_progress,0)+min(original_progress,0)
        deficit=floor-original_progress-float(dot(normalized_task,correction))
        task_solver=None
        if deficit>0:
            compatible,task_solver=cg(op,normalized_task,CFG['task_cg_steps'])
            correction+=deficit/max(float(dot(normalized_task,compatible)),1e-12)*compatible
        correction*=min(1.,CFG['radius']*float(delta.norm())/max(float(correction.norm()),1e-20))
        predicted_progress=float(dot(normalized_task,delta+correction))
        # The raw candidate also satisfies this halfspace, so scaling a feasible
        # correction towards it preserves the declared linear condition.
        assert predicted_progress>=floor-1e-4,(predicted_progress,floor)
        trials=[];new=base;accepted=False;chosen_loss=oldloss
        for j in range(CFG['backtracks']):
            alpha=.5**j;candidate=(base+alpha*correction).detach()
            with torch.no_grad():loss=float(geom.loss(geom(self.sc.image(candidate)),CFG['kappa']))
            merit=.5*float((candidate-base).square().sum())+lam*(loss-oldloss)
            progress=float(dot(normalized_task,candidate-y))
            passed=loss<oldloss and merit<=1e-6 and progress>=floor-1e-4
            trials.append(dict(alpha=alpha,geometry=loss,merit=merit,progress=progress,passed=passed))
            if passed:
                new=candidate;accepted=True;chosen_loss=loss;break
        info=dict(accepted=accepted,geometry_raw=oldloss,geometry_after=chosen_loss,
             original_progress=original_progress,progress_floor=floor,
             final_progress=float(dot(normalized_task,new-y)),
             correction_norm=float((new-base).norm()),delta_norm=float(delta.norm()),
             lambda_actual=lam,solver=solver,task_solver=task_solver,trials=trials,
             seconds=time.monotonic()-started)
        del jt,jv,op,point,obs,gradient,correction,residual,weights
        return new.detach(),info


def noises(ctx,seed):
    gen=torch.Generator('cpu').manual_seed(seed)
    return [torch.randn(ctx['x'].shape,generator=gen).to(ctx['x'].device) for _ in range(33)]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--stage',choices=['probe','full'],required=True)
    ap.add_argument('--pairs',nargs='+',default=['bear__00','brown_owl__00'])
    ap.add_argument('--modes',nargs='+',default=['BASE','CF','NO_NULL','JOINT'])
    ap.add_argument('--steps',nargs='+',type=int,default=[17,25,33,47])
    ap.add_argument('--seed',type=int,default=0);ap.add_argument('--tag',default='v1')
    ap.add_argument('--retention',type=float,default=.5);ap.add_argument('--radius',type=float,default=1.5)
    ap.add_argument('--tube-multiplier',type=float,default=5.)
    ap.add_argument('--memory-fraction',type=float,default=1.)
    args=ap.parse_args();CFG.update(seed=args.seed,retention=args.retention,radius=args.radius)
    seed_setup()
    if args.memory_fraction<1:torch.cuda.set_per_process_memory_fraction(args.memory_fraction,0)
    run=Engine();out=HERE/'outputs'/args.tag
    protocol=dict(args=vars(args),config=CFG.copy(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       local_code_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
       scope='development intervention diagnostics; model-relative text response and source gradient geometry, neither validated semantic facts nor exact reachability')
    dest_protocol=out/f'protocol_{args.stage}_seed{args.seed}.json'
    if dest_protocol.exists():
        old=json.loads(dest_protocol.read_text())
        if old!=protocol:raise RuntimeError('Use a fresh tag when changing protocol')
    else:write(dest_protocol,protocol)
    snapshot=out/f'snapshot_{args.stage}_seed{args.seed}.py'
    if not snapshot.exists():snapshot.write_bytes(Path(__file__).read_bytes())
    rows=[]
    for pair in args.pairs:
        ctx=run.setup(pair);x=ctx['x'];noise=noises(ctx,args.seed)
        saved=torch.load(ROOT/'experiments/flowedit_frequency_trajectory/outputs'/pair/'trajectory.pt',weights_only=True)
        if args.stage=='probe':
            for k in args.steps:
                y=saved['states'][k].to(run.device)
                q=run.query(ctx,y,k,noise[k-17])
                baseline_error=float((y+q['base']-saved['states'][k+1].to(run.device)).abs().max())
                if args.seed==0:assert baseline_error<1e-6,baseline_error
                candidates={'BASE':q['base'],'CF':q['cf'],'NULL':q['empty'],
                            'NO_NULL':q['cf']-q['empty'],'TEXT_SOURCE':q['pieces'][1],
                            'NO_INTERACTION':q['cf']-q['pieces'][3]}
                task=q['task']/q['task'].norm()
                basis=list(q['pieces'])
                threshold=CFG['branch_direction_rtol']*float(q['base'].norm())
                active=[float(b.norm())>=threshold for b in basis]
                normalized=[b/max(float(b.norm()),1e-12) if good else torch.zeros_like(b) for b,good in zip(basis,active)]
                # Use a symmetric, source-independent finite state probe size.
                eps=CFG['probe_relative']*float(q['base'].norm());columns=[]
                for d in normalized:
                    with torch.no_grad():
                        pp=ctx['geometry'](run.sc.image(y+eps*d))
                        pm=ctx['geometry'](run.sc.image(y-eps*d))
                    columns.append(((pp-pm)/(2*eps)).flatten())
                K=torch.stack(columns,1).double();C=torch.tensor([float(dot(task,d)) for d in normalized],device=x.device,dtype=torch.float64)
                gram=K.T@K/K.shape[0];ridge=max(float(torch.trace(gram))*1e-6,1e-12)
                response_capacity=float(C@torch.linalg.solve(gram+ridge*torch.eye(len(basis),device=x.device),C))
                metrics={};columns_img=[('Source',ctx['source'])]
                for name,d in candidates.items():
                    m,im=run.endpoint(ctx,y+d);m['conditional_work_linear']=float(dot(task,d));metrics[name]=m
                    columns_img.append((name,im))
                row=dict(pair=pair,step=k,baseline_error=baseline_error,
                    decomposition_error=q['decomposition_error'],calibration_error=q['calibration_error'],
                    empty_initial_error=q.get('empty_initial_error'),
                    empty_radial_gain=float(dot(y-x,q['empty']))/max(float(dot(y-x,y-x)),1e-20),
                    branch_norms=[float(b.norm()) for b in basis],
                    branch_direction_threshold=threshold,active_branch_directions=active,
                    normalized_branch_gram=[[float(dot(a,b)) for b in normalized] for a in normalized],
                    geometry_response_gram=gram.cpu().tolist(),
                    geometry_response_eigenvalues=torch.linalg.eigvalsh(gram).cpu().tolist(),
                    conditional_response=C.cpu().tolist(),regularized_capacity=response_capacity,metrics=metrics)
                rows.append(row);write(out/'probe.json',dict(rows=rows,scope=protocol['scope']))
                panel(columns_img,out/'probe'/f'{pair}_step{k}.png',title=f'{pair} fixed state {k}',size=256)
                print('PROBE',pair,k,baseline_error,q['decomposition_error'],flush=True)
        else:
            for mode in args.modes:
                dest=out/'full'/pair/f'seed{args.seed}'/mode
                if (dest/'complete.json').exists():continue
                control_ctx=ctx
                if mode in ['PATCH','PATCH_RAW','PATCH_TUBE','PATCH_ADAPT']:
                    from patches import PatchGeometry,PatchRelation
                    patch=PatchRelation(run.sc,ctx['source'],rank=0 if mode=='PATCH_RAW' else CFG['patch_rank'],
                                        deadzone=mode!='PATCH_RAW')
                    control_ctx={**ctx,'geometry':PatchGeometry(patch)}
                tube_budget=None
                if mode in ['PATCH_TUBE','PATCH_ADAPT']:
                    calibration=json.loads((HERE/'outputs/patch_calibration/results.json').read_text())['rows']
                    legal=max(r['excess'] for r in calibration if r['pair']==pair and r['rank']==CFG['patch_rank'] and r['group']=='positive')
                    tube_budget=.5*args.tube_multiplier*legal
                y=x.clone();states=[y.cpu()];updates=[];started=time.monotonic()
                for k in range(17,50):
                    query=run.query(ctx,y,k,noise[k-17],six=mode!='BASE')
                    if mode=='BASE':new=y+query['base'];info=dict(accepted=False)
                    elif mode=='CF':new=y+query['cf'];info=dict(accepted=False)
                    elif mode=='NO_NULL':new=y+query['cf']-query['empty'];info=dict(accepted=False)
                    elif mode in ['JOINT','PATCH','PATCH_RAW','PATCH_TUBE','PATCH_ADAPT']:
                        with sdpa_kernel(SDPBackend.MATH):
                            new,info=run.correct(control_ctx,y,query['cf'],query['task'],args.retention)
                        if mode.startswith('PATCH'):info['source_observer']='patch/'+mode
                        if mode in ['PATCH_TUBE','PATCH_ADAPT']:
                            observer=control_ctx['geometry']
                            with torch.no_grad():
                                actual=float(observer.loss(observer(run.sc.image(new))))
                            attempts=[dict(source_factor=1.,cg_steps=CFG['cg_steps'],loss=actual,
                                           progress=info['final_progress'])]
                            chosen_factor=1.
                            if mode=='PATCH_ADAPT' and actual>tube_budget:
                                original_cfg={key:CFG[key] for key in ['cg_steps','task_cg_steps','gradient_step']}
                                try:
                                    CFG['cg_steps']=CFG['task_cg_steps']=CFG['adaptive_cg_steps']
                                    CFG['gradient_step']=original_cfg['gradient_step']*CFG['adaptive_source_factor']
                                    with sdpa_kernel(SDPBackend.MATH):
                                        alternative,alternative_info=run.correct(control_ctx,y,query['cf'],query['task'],args.retention)
                                finally:CFG.update(original_cfg)
                                with torch.no_grad():alternative_loss=float(observer.loss(observer(run.sc.image(alternative))))
                                attempts.append(dict(source_factor=CFG['adaptive_source_factor'],cg_steps=CFG['adaptive_cg_steps'],
                                    loss=alternative_loss,progress=alternative_info['final_progress']))
                                if alternative_loss<actual:
                                    new=alternative;info=alternative_info;actual=alternative_loss
                                    chosen_factor=CFG['adaptive_source_factor']
                            info['source_observer']='patch/'+mode
                            info['adaptive_attempts']=attempts
                            info['chosen_source_factor']=chosen_factor
                            scale=1.
                            if actual>tube_budget:
                                low,high=0.,1.;candidate=new
                                for _ in range(8):
                                    middle=(low+high)/2
                                    with torch.no_grad():loss=float(observer.loss(observer(run.sc.image(y+middle*(candidate-y)))))
                                    if loss<=tube_budget:low=middle
                                    else:high=middle
                                scale=low;new=(y+scale*(candidate-y)).detach()
                                with torch.no_grad():actual=float(observer.loss(observer(run.sc.image(new))))
                            g=query['task']/query['task'].norm().clamp_min(1e-12)
                            info.update(tube_budget=tube_budget,tube_scale=scale,tube_actual=actual,
                                final_progress=float(dot(g,new-y)),geometry_after=actual,
                                correction_norm=float((new-y-query['cf']).norm()))
                            info['task_floor_satisfied']=info['final_progress']>=info['progress_floor']-1e-4
                            assert actual<=tube_budget+1e-7,(actual,tube_budget)
                    else:raise ValueError(mode)
                    if mode=='BASE' and args.seed==0:
                        info['baseline_error']=float((new-saved['states'][k+1].to(run.device)).abs().max())
                        assert info['baseline_error']<1e-6,info
                    assert torch.isfinite(new).all();y=new.detach();states.append(y.cpu())
                    updates.append(dict(step=k,calibration_error=query['calibration_error'],**info))
                    write(dest/'progress.json',dict(rows=updates))
                    if k in [17,25,33,41,49]:
                        with torch.no_grad():save_image(run.sc.image(y),dest/f'step{k}.png')
                    print('FULL',pair,mode,k,info.get('geometry_after'),info.get('accepted'),flush=True)
                endpoint,im=run.endpoint(ctx,y)
                save_image(im,dest/'final.png');torch.save(dict(states=torch.stack(states),final=y.cpu()),dest/'trajectory.pt')
                write(dest/'complete.json',dict(pair=pair,seed=args.seed,mode=mode,config=CFG.copy(),rows=updates,
                      endpoint=endpoint,seconds=time.monotonic()-started))
                print('ENDPOINT',pair,mode,json.dumps(endpoint),flush=True)
                gc.collect();torch.cuda.empty_cache()
    print('NULL_JOINT_COMPLETE',flush=True)


if __name__=='__main__':
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]):
        main()
