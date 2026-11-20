"""Full FP32 numerical audit; preserves baseline trajectories, never trains."""
from pathlib import Path
import argparse, gc, hashlib, json, sys, time, traceback
from types import SimpleNamespace
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from diffusers import SD3Transformer2DModel
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'vendor/flowedit'))
from FlowEdit_utils import calc_v_sd3
from experiments.stage1.run_sd3_compute_audit import derivative_audit, dump_json, relative_error
from correctflow.rank2_qp import Rank2Weight

def sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

def save(path, obj):
    path.write_text(dump_json(obj, indent=2)+'\n')

def qp_audit(x, zE, delta, anchor, responses):
    D=(zE.double()-x.double()).flatten().cpu().numpy()
    a=anchor.flatten().double().cpu().numpy()
    radius=.1*float(delta.double().norm())
    W=Rank2Weight(*(v.flatten().double().numpy() for v in responses))
    result=W.solve(D,a,radius); w=result.displacement; n=result.normal
    force=W.alpha*D
    stat=W.hessian_action(w,result.mu)+force-result.lam*n
    denom=1+np.linalg.norm(D)+radius
    primal=max(0.,-n@w,np.linalg.norm(w)-radius)/denom
    comp=max(abs(result.lam*(n@w)),abs(result.mu*(np.dot(w,w)-radius**2)/2))/(1+abs(W.objective(D,np.zeros_like(w))))
    stationarity=np.linalg.norm(stat)/(1+np.linalg.norm(force)+np.linalg.norm(W.hessian_action(w)))
    if result.status=='policy_fallback': stationarity=0.; comp=0.
    proposed=(zE.double()+torch.from_numpy(w.reshape(zE.shape)).to(zE.device)).float()
    actual=(proposed.double()-zE.double()).flatten().cpu().numpy()
    rounded_primal=max(0.,-n@actual,np.linalg.norm(actual)-radius)/denom
    rounded_distance=(np.linalg.norm(D+actual)**2-np.linalg.norm(D)**2)/(1+np.linalg.norm(D)**2)
    dist=(np.linalg.norm(D+w)**2-np.linalg.norm(D)**2)/(1+np.linalg.norm(D)**2)
    obj=(W.objective(D,w)-W.objective(D,np.zeros_like(w)))/(1+abs(W.objective(D,np.zeros_like(w))))
    projector=max([abs(np.dot(v,v)-1) for v in W.units.values() if v is not None] or [0.])
    # This phase measures proposed corrections only: baseline is always continued.
    accepted=rounded_primal<=1e-7 and rounded_distance<=1e-7
    row=dict(radius=radius,anchor_norm=float(np.linalg.norm(a)),response_source_norm=float(responses[0].norm()),response_target_norm=float(responses[1].norm()),
        W_min_eigenvalue=W.minimum_weight_eigenvalue(),projector_idempotence=projector,stationarity=float(stationarity),primal=float(primal),complementarity=float(comp),source_distance_increase=float(dist),objective_increase=float(obj),
        rounded_primal=float(rounded_primal),rounded_source_distance_increase=float(rounded_distance),rounding_error_norm=float(np.linalg.norm(actual-w)),rounded_submission_accepted=bool(accepted),
        status=result.status,mu=result.mu,lam=result.lam,iterations=result.iterations)
    row['passed']=bool(primal<=1e-7 and stationarity<=1e-7 and comp<=1e-7 and dist<=1e-7 and obj<=1e-7 and accepted and row['W_min_eigenvalue']>=1-1e-10 and projector<1e-10)
    return row,torch.from_numpy(w.reshape(zE.shape)).float()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',required=True);parser.add_argument('--seed',type=int,required=True);parser.add_argument('--limit',type=int,default=24)
    args=parser.parse_args(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device(args.device); root=ROOT/'experiments/stage1_full'; output=root/'outputs';output.mkdir(exist_ok=True)
    meta=json.loads((root/'inputs/manifest.json').read_text()); entries=meta['inputs'][:args.limit]
    network_times=torch.tensor(meta['timesteps'],device=device,dtype=torch.float32)
    clocks=network_times/1000
    active=range(17,50); quantiles=np.quantile(clocks[17:].cpu().numpy(), [.8,.5,.2]); selected=sorted({min(active,key=lambda k:abs(float(clocks[k])-float(q))) for q in quantiles})
    print(dump_json({'device':args.device,'seed':args.seed,'selected_steps':selected}),flush=True)
    model=SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to(device)
    pipe=SimpleNamespace(transformer=model,do_classifier_free_guidance=True)
    start=time.monotonic(); summaries=[]
    for index,entry in enumerate(entries):
        dest=output/f"{entry['pair_id']}__seed{args.seed}";dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists():
            summaries.append(json.loads((dest/'complete.json').read_text()));print('resume '+dest.name,flush=True);continue
        started=time.monotonic(); rows=[]; states=[]; snapshots=[];controls=[];diagnostics=[];repeats=[]
        try:
            payload=torch.load(root/'inputs'/entry['file'],weights_only=True,map_location='cpu')
            x=payload['x'].to(device); z=x.clone()
            es,ea=[tuple(v.to(device) for v in e) for e in payload['embeddings']]
            seq=torch.cat([es[1],es[0],ea[1],ea[0]]);pool=torch.cat([es[3],es[2],ea[3],ea[2]])
            gen=torch.Generator('cpu').manual_seed(args.seed)
            noises=[torch.randn(x.shape,generator=gen,dtype=torch.float32) for _ in active]
            hashes=[sha(noise) for noise in noises];rng_frozen=sha(gen.get_state())
            for step in active:
                t=clocks[step]; h=float(t-(clocks[step+1] if step<49 else torch.zeros_like(t)))
                noise=noises[step-17].to(device);qs=(1-t)*x+t*noise;qa=z+qs-x
                # Exactly the official SD3 velocity wrapper and update ordering.
                vs,va=calc_v_sd3(pipe,torch.cat([qs,qs,qa,qa]),seq,pool,3.5,13.5,network_times[step])
                delta=(-h)*(va-vs);zE=z+delta
                if step in selected:
                    state_rows=[];responses=[];branch_tensors=[]
                    for branch,q,e,beta,ref in [('source',qs,es,3.5,vs),('target',qa,ea,13.5,va)]:
                        state={'q':q,'network_time':network_times[step],'h':h}
                        torch.cuda.reset_peak_memory_stats(device)
                        row,tensors=derivative_audit(model,state,e,beta,'fp32',20261005+step)
                        row.update(pair_id=entry['pair_id'],seed=args.seed,step=step,t=float(t),branch=branch,baseline_primal_error=relative_error(tensors['velocity'].to(device),ref))
                        row['passed']=bool(row['finite'] and row['dual_residual']<=1e-4 and row['baseline_primal_error']<=1e-4 and row['random_response_psd_inner']>=-1e-7)
                        rows.append(row);state_rows.append(row);responses.append(tensors['response']);branch_tensors.append(tensors)
                        if args.seed==0 and index<4 and step==selected[0]:
                            repeated,rt=derivative_audit(model,state,e,beta,'fp32',20261005+step)
                            errors={name:relative_error(rt[name],tensors[name]) for name in ('velocity','gseq','gpool','response')}
                            repeats.append(dict(branch=branch,step=step,errors=errors,passed=max(errors.values())<=1e-6))
                    qE=zE+qs-x
                    with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
                        conditional=model(hidden_states=torch.cat([qE,qE]),timestep=(network_times[step]).expand(2),encoder_hidden_states=torch.cat([es[0],ea[0]]),pooled_projections=torch.cat([es[2],ea[2]]),return_dict=False)[0]
                    src_pred,tar_pred=conditional.chunk(2);a=-(1-t)/t*(tar_pred-src_pred)
                    qpr,w=qp_audit(x,zE,delta,a,responses)
                    qpr.update(pair_id=entry['pair_id'],seed=args.seed,step=step,t=float(t),baseline_noise_stream_unchanged=rng_frozen==sha(gen.get_state()))
                    states.append(qpr)
                    snapshots.append(dict(step=step,t=float(t),h=h,x=x.cpu(),z=z.cpu(),qs=qs.cpu(),qa=qa.cpu(),zE=zE.cpu(),delta=delta.cpu(),anchor=a.cpu(),responses=responses,qp_displacement=w,branch_tensors=branch_tensors,noise_hash=hashes[step-17]))
                    if args.seed==0 and index<16:
                        null_seq=torch.cat([es[1],es[0],es[1],es[0]]);null_pool=torch.cat([es[3],es[2],es[3],es[2]])
                        vn1,vn2=calc_v_sd3(pipe,torch.cat([qs]*4),null_seq,null_pool,3.5,3.5,network_times[step])
                        vd1,vd2=calc_v_sd3(pipe,torch.cat([qs]*4),null_seq,null_pool,3.5,13.5,network_times[step])
                        q_null=qs+(-h)*(vd2-vd1)
                        with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
                            null_predictions=model(hidden_states=torch.cat([q_null,q_null]),timestep=network_times[step].expand(2),encoder_hidden_states=torch.cat([es[0],es[0]]),pooled_projections=torch.cat([es[2],es[2]]),return_dict=False)[0]
                        null_p1,null_p2=null_predictions.chunk(2)
                        null_a=-(1-t)/t*(null_p2-null_p1);null_weight=Rank2Weight(responses[0].numpy(),responses[0].numpy())
                        null_solution=null_weight.solve(((-h)*(vd2-vd1)).cpu().numpy().reshape(-1),null_a.cpu().numpy().reshape(-1),.1*float(delta.norm()))
                        controls.append(dict(step=step,matched_cfg_max_abs=float(((-h)*(vn2-vn1)).abs().max()),different_cfg_baseline_drift=float((h*(vd2-vd1)).norm()),same_condition_common_cfg_anchor_norm=float(null_a.norm()),added_correction_norm=float(np.linalg.norm(null_solution.displacement)),passed=bool(torch.equal(vn1,vn2) and not np.any(null_solution.displacement))))
                    if args.seed==0 and index<8 and step==selected[0] and float(a.norm())>1e-14:
                        aux=torch.Generator('cpu').manual_seed(98121+index)
                        u,v=[torch.randn(qE.shape,generator=aux).to(device) for _ in range(2)]
                        u=u/u.norm();v=v/v.norm()
                        def field(q):
                            pred=model(hidden_states=torch.cat([q,q]),timestep=(network_times[step]).expand(2),encoder_hidden_states=torch.cat([es[0],ea[0]]),pooled_projections=torch.cat([es[2],ea[2]]),return_dict=False)[0]
                            ps,pa=pred.chunk(2);return -(1-t)/t*(pa-ps)
                        with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
                            _,ju=torch.func.jvp(field,(qE,),(u,));_,jv=torch.func.jvp(field,(qE,),(v,))
                        left=float((u.double()*jv.double()).sum());right=float((v.double()*ju.double()).sum())
                        diagnostics.append(dict(step=step,u_Da_v=left,v_Da_u=right,antisymmetry=left-right,normalized_antisymmetry=abs(left-right)/(1+abs(left)+abs(right)),is_gate=False))
                    save(dest/'branch_rows.json',rows);save(dest/'qp_rows.json',states);save(dest/'controls.json',controls);save(dest/'conservativity.json',diagnostics);save(dest/'repeatability.json',repeats)
                    print(dump_json({'case':dest.name,'step':step,'dual_max':max(r['dual_residual'] for r in state_rows),'primal_max':max(r['baseline_primal_error'] for r in state_rows),'qp_passed':qpr['passed']}),flush=True)
                z=zE.detach()
            noise_unchanged=hashes==[sha(n) for n in noises] and rng_frozen==sha(gen.get_state())
            torch.save(dict(record=payload['record'],snapshots=snapshots,baseline_final=z.cpu(),noise_hashes=hashes,noise_seed=args.seed),dest/'states.pt')
            summary=dict(pair_id=entry['pair_id'],seed=args.seed,branches=len(rows),states=len(states),noise_unchanged=noise_unchanged,elapsed_seconds=time.monotonic()-started,
                passed=all(r['passed'] for r in rows+states+controls+repeats) and noise_unchanged,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            save(dest/'complete.json',summary);summaries.append(summary)
            print(dump_json({'completed':len(summaries),'total':len(entries),**summary}),flush=True)
        except Exception:
            error=traceback.format_exc();save(dest/'failure.json',{'error':error,'pair_id':entry['pair_id'],'seed':args.seed});print(error,flush=True)
            summaries.append(dict(pair_id=entry['pair_id'],seed=args.seed,passed=False,error=error))
        gc.collect();torch.cuda.empty_cache()
        save(output/f'worker_seed{args.seed}.json',dict(seed=args.seed,device=args.device,completed=len(summaries),total=len(entries),cases=summaries,selected_steps=selected,elapsed_seconds=time.monotonic()-start))
    print('WORKER_COMPLETE',flush=True)

if __name__=='__main__':main()
