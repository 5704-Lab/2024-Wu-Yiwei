"""Independent finite-difference and adjoint audit of the new source observer."""
import sys
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from patches import CleanScorer,PatchRelation,ROOT,seed_setup,write


def main():
    seed_setup();sc=CleanScorer(torch.device('cuda:0'),torch.float32,'native');rows=[]
    for pair in ['bear__00','brown_owl__00']:
        x=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True)['x'].to(sc.device)
        with torch.no_grad():source=sc.image(x)
        relation=PatchRelation(sc,source,rank=2)
        z=torch.load(HERE/'outputs/v1_full/full'/pair/'seed0/CF/trajectory.pt',weights_only=True)['final'].to(sc.device)
        direction=(x-z);direction/=direction.norm()
        point=z.detach().requires_grad_(True);obs=relation(sc.image(point))
        cotangent=obs.detach()/obs.detach().norm()
        jt,=torch.autograd.grad(obs,point,grad_outputs=cotangent)
        _,jv=torch.func.jvp(lambda y:relation(sc.image(y)),(z,),(direction,))
        left=float((jv.double()*cotangent.double()).sum());right=float((jt.double()*direction.double()).sum())
        adjoint=abs(left-right)/max(abs(left),abs(right),1e-12)
        differences=[]
        for eps in [.05,.02,.01]:
            with torch.no_grad():fd=(relation(sc.image(z+eps*direction))-relation(sc.image(z-eps*direction)))/(2*eps)
            error=float((fd-jv).norm()/jv.norm().clamp_min(1e-12))
            differences.append(dict(epsilon=eps,relative_error=error))
        with torch.no_grad():source_error=float(relation(source).abs().max())
        passed=adjoint<2e-3 and all(r['relative_error']<.02 for r in differences) and source_error<1e-7
        rows.append(dict(pair=pair,adjoint_error=adjoint,finite_differences=differences,
                  source_error=source_error,passed=passed))
        write(HERE/'outputs/observer_derivatives.json',dict(rows=rows,scope='numerical source observer derivatives, not semantic validity'))
        assert passed,rows[-1]
        print(pair,rows[-1],flush=True)
    print('OBSERVER_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':
    with sdpa_kernel(SDPBackend.MATH):main()
