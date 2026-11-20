"""Clean-state response pilot. Frozen SD3; no training or inversion."""
import argparse,gc,hashlib,json,sys,time,traceback
from pathlib import Path
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/target_response_direction/outputs'

def save(name,value):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def analytic():
    torch.manual_seed(20261009)
    records=[]
    for n,m in [(3,5),(5,3),(8,8)]:
        B=torch.randn(m,n,dtype=torch.float64);q=torch.randn(n,dtype=torch.float64,requires_grad=True)
        xi=torch.randn(m,dtype=torch.float64,requires_grad=True);c=torch.randn(m,dtype=torch.float64)
        g=torch.autograd.grad(xi@(B@q+c),xi,create_graph=True)[0]
        u=g/g.norm();ref=torch.randn(m,dtype=torch.float64);ref=ref/ref.norm()
        A=(torch.eye(m,dtype=torch.float64)-u.detach()[:,None]*u.detach()[None,:])@B/g.detach().norm()
        b=torch.autograd.grad(ref@u,q)[0];expected=A.T@ref
        lam=.7;w=torch.linalg.solve(A.T@A+lam*torch.eye(n,dtype=torch.float64),expected)
        R=.5*float(w.norm())
        lo,hi=0.,1.
        def solve(mu):return torch.linalg.solve(A.T@A+(lam+mu)*torch.eye(n,dtype=torch.float64),expected)
        while float(solve(hi).norm())>R:hi*=2
        for _ in range(80):
            mid=(lo+hi)/2
            if float(solve(mid).norm())>R:lo=mid
            else:hi=mid
        wc=solve(hi)
        err=float((b-expected).norm()/(1+expected.norm()))
        kkt=float(((A.T@A+(lam+hi)*torch.eye(n,dtype=torch.float64))@wc-expected).norm())
        records.append(dict(n=n,m=m,gradient_error=err,kkt_residual=kkt,radius_error=abs(float(wc.norm())-R),ascent=float(b@wc)))
    passed=all(r['gradient_error']<1e-9 and r['kkt_residual']<1e-8 and r['ascent']>0 for r in records)
    save('stage0.json',dict(passed=passed,cases=records,scope='analytic normalization, mixed derivative, budget KKT; not full stage0 coverage'))
    assert passed
    print('ANALYTIC_PASS',flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:0');ap.add_argument('--dtype',choices=['float32','float64'],default='float32');ap.add_argument('--pair',default='bear__00');ap.add_argument('--step',type=int,default=33);ap.add_argument('--analytic-only',action='store_true');args=ap.parse_args()
    analytic()
    if args.analytic_only:return
    from diffusers import SD3Transformer2DModel
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dtype=getattr(torch,args.dtype);device=torch.device(args.device)
    payload=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{args.pair}.pt',weights_only=True)
    saved=torch.load(ROOT/'experiments/stage1_full/outputs'/f'{args.pair}__seed0'/'states.pt',weights_only=True)
    snap=next(s for s in saved['snapshots'] if s['step']==args.step)
    seq,neg,pool,negpool=[t.to(device=device,dtype=dtype) for t in payload['embeddings'][1]]
    y=snap['z'].to(device=device,dtype=dtype);yb=snap['zE'].to(device=device,dtype=dtype)
    mask=torch.ones_like(seq);mask[:,:77,2048:]=0
    beta=13.5
    start=time.monotonic()
    model=SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',torch_dtype=dtype,local_files_only=True).eval().requires_grad_(False).to(device)
    print('MODEL_LOADED',args.dtype,flush=True)
    def pred(q,s,p,tau=0.):
        return model(hidden_states=q,timestep=torch.tensor([tau*1000],device=device,dtype=dtype),encoder_hidden_states=s,pooled_projections=p,return_dict=False)[0]
    def response(q,a,graph=False):
        s=seq.detach().requires_grad_(True);p=pool.detach().requires_grad_(True)
        v=pred(q,s,p);g=torch.autograd.grad(beta*(v*a).sum(),(s,p),create_graph=graph)
        return torch.cat([(g[0]*mask).flatten(),g[1].flatten()])
    result=dict(pair=args.pair,step=args.step,trajectory_noise_time=snap['t'],response_time=0.,input='clean y',dtype=args.dtype,weights_frozen=True,padding_excluded=True,cached_original_flowedit_state=True,source_sha256=payload['record']['source_sha256'],code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        with sdpa_kernel(SDPBackend.MATH):
            with torch.no_grad():
                vc=pred(yb,seq,pool);vn=pred(yb,neg,negpool);v=vn+beta*(vc-vn);a=(-v/v.norm()).detach()
            print('ENDPOINT_FORWARD',float(v.norm()),flush=True)
            gr=response(y,a).detach();ur=gr/gr.norm()
            q=yb.detach().requires_grad_(True);g=response(q,a,True);norm=g.norm();u=g/norm;F=ur@u
            b=torch.autograd.grad(F,q,retain_graph=True)[0]
            zeta=((ur-u.detach()*(u.detach()@ur))/norm.detach()).detach()
            bm=torch.autograd.grad(g@zeta,q)[0]
            result.update(reference_gradient_norm=float(gr.norm()),candidate_gradient_norm=float(norm.detach()),cosine_before=float(F.detach()),angle_before_degrees=float(torch.acos(F.detach().clamp(-1,1))*180/torch.pi),b_norm=float(b.norm()),mixed_chain_error=float((bm-b).norm()/(1+b.norm())),finite=bool(torch.isfinite(b).all()))
            u0=u.detach();f0=float(F.detach());gn=float(norm.detach())
            del q,g,u,norm,F,gr,vc,vn,v,bm
            gc.collect();torch.cuda.empty_cache()
            R=.05*float((yb-y).norm());bn=float(b.norm());lam=bn/R if bn>0 and R>0 else 1.
            direction=b/bn if bn>0 else b
            records=[]
            for scale in [1.,.5,.1]:
                w=(R*scale)*direction
                gnew=response(yb+w,a).detach();fnew=float(ur@(gnew/gnew.norm()))
                gain=fnew-f0;predicted=float((b*w).sum());cost=.5*lam*float(w.square().sum())
                records.append(dict(scale=scale,norm=float(w.norm()),cosine_after=fnew,cosine_gain=gain,predicted_gain=predicted,gradient_norm_ratio=float(gnew.norm())/gn,regularized_loss_gain=gain-cost,armijo_without_numerical_margin=gain>=cost+.1*predicted))
                del gnew
                print('TRIAL',records[-1],flush=True)
            result.update(trials=records,radius=R,lambda_value=lam,endpoint_probe_velocity_norm=float(v.norm()) if 'v' in locals() else None,elapsed_seconds=time.monotonic()-start,peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device),passed_numeric=bool(result['finite'] and result['mixed_chain_error']<=1e-5))
            torch.save(dict(y=y.cpu(),candidate=yb.cpu(),direction=direction.detach().cpu(),probe=a.cpu(),reference_direction=ur.cpu(),candidate_direction=u0.cpu()),OUT/f'{args.pair}_step{args.step}_{args.dtype}_tensors.pt')
    except Exception:
        result.update(error=traceback.format_exc(),elapsed_seconds=time.monotonic()-start,passed_numeric=False)
        print(result['error'],flush=True)
    save(f'{args.pair}_step{args.step}_{args.dtype}.json',result)
    print('PRECHECK_COMPLETE',json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
