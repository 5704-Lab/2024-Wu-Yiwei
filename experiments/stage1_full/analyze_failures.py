"""Supplementary cancellation diagnostics; does not replace registered gates."""
import hashlib,json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/stage1_full/outputs'

def main():
    summary=json.loads((OUT/'summary.json').read_text());results=[]
    for row in summary['failures']['branches']:
        name=f"{row['pair_id']}__seed{row['seed']}"
        path=OUT/name/'states.pt'
        if not path.exists():continue
        payload=torch.load(path,map_location='cpu',weights_only=True)
        snap=next(s for s in payload['snapshots'] if s['step']==row['step'])
        tensors=snap['branch_tensors'][0 if row['branch']=='source' else 1]
        gen=torch.Generator('cpu').manual_seed(20261005+row['step'])
        tangents=[torch.randn(tensors[k].shape,generator=gen,dtype=torch.float32) for k in ['gseq','gpool']]
        norm_g=torch.sqrt(sum(tensors[k].double().square().sum() for k in ['gseq','gpool']))
        norm_de=torch.sqrt(sum(v.double().square().sum() for v in tangents))
        bound=float(norm_g*norm_de)
        results.append(dict(pair_id=row['pair_id'],seed=row['seed'],step=row['step'],branch=row['branch'],registered_dual_residual=row['dual_residual'],baseline_primal_error=row['baseline_primal_error'],absolute_dual_error=row['dual_absolute_error'],gradient_tangent_norm_product=bound,cancellation_ratio=bound/max(abs(row['dual_lhs']),abs(row['dual_rhs']),1e-30),supplementary_error_divided_by_gradient_tangent_bound=row['dual_absolute_error']/(1+bound),does_not_change_gate=True))
    report=dict(diagnostics=results,criterion='Registered scalar denominator retained. Norm product is a diagnostic, not a replacement gate.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'failure_cancellation_diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'completed_failure_diagnostics':len(results),'largest_cancellation_ratio':max([r['cancellation_ratio'] for r in results] or [0])}))

if __name__=='__main__':main()
