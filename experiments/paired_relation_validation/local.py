"""Exact matrix-free Gauss--Newton action at real clean candidate states."""
import json
import time
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

from common import Paired,PAIRS,ROOT,OUT,residual,write,protocol,save_image


def dot(a,b):return (a.double()*b.double()).sum()


def cg(operator,b,iterations=8,rtol=1e-4):
    x=torch.zeros_like(b);r=b.clone();p=r.clone();rr=dot(r,r)
    initial=float(rr.sqrt());steps=0
    for i in range(iterations):
        if float(rr.sqrt())<=rtol*max(initial,1e-30):break
        ap=operator(p);den=dot(p,ap)
        if float(den)<=0:raise RuntimeError("nonpositive GN operator")
        alpha=(rr/den).float();x=x+alpha*p;r=r-alpha*ap
        new=dot(r,r);p=r+(new/rr).float()*p;rr=new;steps=i+1
    return x,dict(iterations=steps,relative_residual=float(rr.sqrt())/max(initial,1e-30),
                  converged=float(rr.sqrt())<=rtol*max(initial,1e-30))


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    protocol("local",dict(mu=1,lambda_grid=[.1,.3,1],budget_grid=[.05,.1,.2],
                          cg_max_iterations=8,cg_rtol=1e-4,finite_retraction_max=8))
    prior=json.loads((OUT/"local.json").read_text()) if (OUT/"local.json").exists() else {}
    exp=Paired();rows=prior.get("rows",[]);modelchecks=prior.get("modelchecks",[])
    for pair in PAIRS:
        ctx=exp.setup(pair)
        saved=torch.load(ROOT/"experiments/flowedit_frequency_trajectory/outputs"/pair/"trajectory.pt",weights_only=True)
        for step in [17,33,47]:
            if sum(r["pair"]==pair and r["step"]==step for r in rows)==3:
                continue
            start=time.monotonic();y=saved["states"][step+1].to(exp.device)
            d=y-saved["states"][step].to(exp.device)
            q=y.detach().requires_grad_(True)
            with sdpa_kernel(SDPBackend.MATH):
                f=exp.feature(q);a,n,rho=residual(f,ctx["embeddings"][1])
                err=n-ctx["m"];loss=.5*err.square().sum()
                b=torch.autograd.grad(loss,q,retain_graph=True)[0].detach()
                g=torch.autograd.grad(a,q,retain_graph=True)[0].detach()
            reference=exp.values(pair,f.detach());repeat,_im,_=exp.evaluate(pair,y)
            def nfunc(z):return residual(exp.feature(z),ctx["embeddings"][1])[1]
            def Jv(v):
                with sdpa_kernel(SDPBackend.MATH):
                    _,j=torch.func.jvp(nfunc,(y,),(v,))
                return j.detach()
            def JT(v):
                return torch.autograd.grad(n,q,grad_outputs=v,retain_graph=True)[0].detach()
            unitg=g/g.norm().clamp_min(1e-20)
            vg=Jv(unitg);vj=JT(vg)
            reciprocity=abs(float(dot(unitg,vj)-dot(vg,vg)))/max(float(dot(vg,vg)),1e-20)
            eps=.002*float(d.norm())
            with torch.no_grad():fd=(nfunc(y+eps*unitg)-nfunc(y-eps*unitg))/(2*eps)
            fd_error=float((fd-vg).norm()/vg.norm().clamp_min(1e-20))
            check=dict(pair=pair,step=step,JVP_VJP_reciprocity_error=reciprocity,
                state_finite_difference_error=fd_error,repeat_target_error=abs(repeat["target_cosine"]-reference["target_cosine"]),
                gradient_norm=float(b.norm()),target_gradient_norm=float(g.norm()),
                target_direction_residual_response=float(vg.norm()),nominal_increment_norm=float(d.norm()))
            modelchecks=[c for c in modelchecks if not(c["pair"]==pair and c["step"]==step)]+[check]
            for lam in [.1,.3,1]:
                if any(r["pair"]==pair and r["step"]==step and r["lambda_value"]==lam for r in rows):
                    continue
                def M(v):return v+lam*JT(Jv(v))
                r0,ci=cg(M,-lam*b)
                w,gi=cg(M,g)
                nu=max(0.,-float(dot(g,r0))/max(float(dot(g,w)),1e-30))
                r=r0+nu*w
                budget=.1*float(d.norm())
                capped=float(r.norm())>budget
                if capped:raise RuntimeError("Active dual budget required; do not substitute unregistered clipping")
                station=float((M(r)+lam*b-nu*g).norm())/max(float((lam*b).norm()),1e-20)
                alpha=1.;coefficient=0.;accepted=False;new=y
                tol=2e-7
                for back in range(8):
                    c=0.;point=y+alpha*r
                    if nu>0 and float(g.norm())>1e-8:
                        lift=g/dot(g,g).float()
                        for _ in range(8):
                            qp=point.detach().requires_grad_(True)
                            aa=residual(exp.feature(qp),ctx["embeddings"][1])[0]
                            gap=float(aa.detach()-a.detach())
                            if abs(gap)<tol:break
                            slope=torch.autograd.grad(aa,qp)[0]
                            derivative=float(dot(slope,lift))
                            if abs(derivative)<1e-8:break
                            c-=gap/derivative;point=y+alpha*r+c*lift
                    obs,_,_=exp.evaluate(pair,point)
                    if obs["target_cosine"]>=reference["target_cosine"]-tol and obs["K"]<reference["K"]:
                        new=point.detach();accepted=True;coefficient=c;break
                    alpha*=.5
                obs,image,_=exp.evaluate(pair,new)
                name="PAIR_lam%.1f"%lam
                if lam==.3:
                    save_image(image,OUT/pair/"local"/(name+"_step%d.png"%step))
                    save_image(exp.sc.image(y).detach(),OUT/pair/"local"/("FlowEdit_step%d.png"%step))
                row=dict(lambda_value=lam,method=name,reference=reference,
                    after=obs,**check,cg_primary=ci,cg_target=gi,KKT_relative_error=station,
                    target_constraint_active=nu>0,linear_target_change=float(dot(g,r)),
                    predicted_K_change=float(dot(b,r)),repair_norm=float((new-y).norm()),
                    unconstrained_repair_norm=float(r.norm()),repair_to_nominal_ratio=float((new-y).norm()/d.norm()),
                    requested_budget=budget,budget_active=capped,accepted=accepted,
                    backtrack_alpha=alpha if accepted else 0.,retraction_coefficient=coefficient,
                    actual_K_change=obs["K"]-reference["K"],actual_target_logit_change=obs["target_logit"]-reference["target_logit"],
                    jvp_backend="forward_mode_torch_func",seconds=time.monotonic()-start)
                rows.append(row);write(OUT/"local.json",dict(rows=rows,modelchecks=modelchecks))
            del q,f,n,a,loss
            torch.cuda.empty_cache()
            print(json.dumps(dict(pair=pair,step=step,local_complete=True,seconds=time.monotonic()-start)),flush=True)
    print("LOCAL_COMPLETE",flush=True)


if __name__=="__main__":main()
