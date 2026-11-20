"""Resolve large finite-state derivative discrepancies and algebraic equivalence."""
import json
import argparse
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel

from common import Paired,PAIRS,ROOT,OUT,residual,rotate,write,protocol


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--endpoints-only",action="store_true")
    args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    protocol("derivative_audit",dict(post_hoc=True,changes_no_editing_outputs=True))
    prior=json.loads((OUT/"derivative_audit.json").read_text()) if args.endpoints_only and (OUT/"derivative_audit.json").exists() else {}
    exp=Paired();rows=prior.get("rows",[]);own=prior.get("own_condition_rows",[])
    original=json.loads((OUT/"local.json").read_text())["modelchecks"]
    for pair in PAIRS:
        ctx=exp.setup(pair)
        with torch.no_grad():
            _,source_out=exp.feature_image(ctx["source_rgb"],hidden=True)
            source_patches=F.normalize(source_out.hidden_states[6][0],dim=-1)
            source_gram=source_patches@source_patches.T
        baseline=torch.load(ROOT/"experiments/flowedit_frequency_trajectory/outputs"/pair/"trajectory.pt",weights_only=True)
        for check in ([] if args.endpoints_only else [c for c in original if c["pair"]==pair]):
            step=check["step"];y=baseline["states"][step+1].to(exp.device)
            d=y-baseline["states"][step].to(exp.device)
            q=y.detach().requires_grad_(True)
            with sdpa_kernel(SDPBackend.MATH):
                f=exp.feature(q);a,n,rho=residual(f,ctx["embeddings"][1]);K=.5*(n-ctx["m"]).square().sum()
                g=torch.autograd.grad(a,q,retain_graph=True)[0].detach()
                b=torch.autograd.grad(K,q,retain_graph=True)[0].detach()
                rotated_source_cos=f@rotate(ctx["fs"],ctx["embeddings"][0],ctx["embeddings"][1])
                gh=torch.autograd.grad(rotated_source_cos,q)[0].detach()
            # Exact algebra: K = 1-(h-a_s*a)/(rho_s*rho).
            closed=1-(rotated_source_cos.detach()-ctx["source_pair_cosine"]*a.detach())/(ctx["source_rho"]*rho.detach())
            predicted=-gh/(ctx["source_rho"]*rho.detach())+(ctx["source_pair_cosine"]-a.detach()*rotated_source_cos.detach())/(ctx["source_rho"]*rho.detach()**3)*g
            identity_error=float((predicted-b).norm()/b.norm().clamp_min(1e-20))
            u=g/g.norm()
            def nfunc(z):return residual(exp.feature(z),ctx["embeddings"][1])[1]
            with sdpa_kernel(SDPBackend.MATH):_,j=torch.func.jvp(nfunc,(y,),(u,))
            sweep=[]
            for rel in [.002,.001,.0005,.0002,.0001]:
                eps=rel*float(d.norm())
                with torch.no_grad():fd=(nfunc(y+eps*u)-nfunc(y-eps*u))/(2*eps)
                sweep.append(dict(relative_state_step=rel,relative_error=float((fd-j).norm()/j.norm().clamp_min(1e-20))))
            rows.append(dict(pair=pair,step=step,original_fd_error=check["state_finite_difference_error"],
                sweep=sweep,closed_K_error=float(abs(closed-K.detach())),gradient_identity_error=identity_error))
            write(OUT/"derivative_audit.json",dict(rows=rows,own_condition_rows=own))
        for variant in ["BASE","SHORT_SOURCE","SHORT_TARGET","SHORT_BOTH","NO_EDIT_ASYMMETRIC_CFG","MATCHED_CFG_FULL","MATCHED_CFG_SHORT_BOTH","COUNTERFACTUAL_FULL","COUNTERFACTUAL_SHORT_BOTH"]:
            if any(r["pair"]==pair and r["variant"]==variant for r in own):continue
            path=OUT/pair/variant/"trajectory.pt"
            if not path.exists():continue
            saved=torch.load(path,weights_only=True)
            si=2 if variant in ["SHORT_SOURCE","SHORT_BOTH","MATCHED_CFG_SHORT_BOTH","COUNTERFACTUAL_SHORT_BOTH"] else 0
            ti=0 if variant=="NO_EDIT_ASYMMETRIC_CFG" else (3 if variant in ["SHORT_TARGET","SHORT_BOTH","MATCHED_CFG_SHORT_BOTH","COUNTERFACTUAL_SHORT_BOTH"] else 1)
            es,et=ctx["embeddings"][si],ctx["embeddings"][ti]
            with torch.no_grad():
                image=exp.sc.image(saved["final"].to(exp.device))
                f,image_out=exp.feature_image(image,hidden=True)
                patches=F.normalize(image_out.hidden_states[6][0],dim=-1)
                gram=patches@patches.T
            sa,ns,_=residual(ctx["fs"],es);ta,nt,_=residual(f,et)
            own.append(dict(pair=pair,variant=variant,source_pair_cosine=float(sa),target_pair_cosine=float(ta),
                actual_condition_K=float(.5*(nt-rotate(ns,es,et)).square().sum()),
                actual_source_logit=float(exp.sc.alpha*(f@es)+exp.sc.bias),
                actual_target_logit=float(exp.sc.alpha*ta+exp.sc.bias),
                source_original_patch_relation_mse=float((gram-source_gram).square().mean()),
                source_original_position_feature_mse=float((patches-source_patches).square().mean())))
        write(OUT/"derivative_audit.json",dict(rows=rows,own_condition_rows=own))
        print("DERIVATIVE_AUDIT_PAIR_COMPLETE",pair,flush=True)
    print("DERIVATIVE_AUDIT_COMPLETE",flush=True)


if __name__=="__main__":main()
