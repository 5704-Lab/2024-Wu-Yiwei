"""Aggregate all predeclared cases without dropping failed states."""
import json, hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/'experiments/stage1_full/outputs'

def metric(rows,key):
    values=[r[key] for r in rows if r.get(key) is not None]
    return dict(count=len(values),min=float(min(values)),median=float(np.median(values)),p95=float(np.quantile(values,.95)),max=float(max(values))) if values else None

def main():
    manifest=json.loads((ROOT/'experiments/stage1_full/inputs/manifest.json').read_text())
    expected=[(entry['pair_id'],seed) for entry in manifest['inputs'] for seed in [0,1]]
    branches=[];qp=[];controls=[];diagnostics=[];repeats=[];completed=[];missing=[];errors=[]
    for pair,seed in expected:
        dest=OUTPUT/f'{pair}__seed{seed}'
        if (dest/'complete.json').exists():completed.append(json.loads((dest/'complete.json').read_text()))
        else:
            missing.append(dict(pair_id=pair,seed=seed))
            if (dest/'failure.json').exists():errors.append(json.loads((dest/'failure.json').read_text()))
        for filename,records in [('branch_rows.json',branches),('qp_rows.json',qp),('controls.json',controls),('conservativity.json',diagnostics),('repeatability.json',repeats)]:
            if (dest/filename).exists():
                records.extend([{'pair_id':pair,'seed':seed,**r} for r in json.loads((dest/filename).read_text())])
    fields=['dual_residual','baseline_primal_error','random_response_psd_inner','response_norm','max_gpu_memory_bytes']
    qp_fields=['stationarity','primal','complementarity','rounded_primal','rounded_source_distance_increase','objective_increase','source_distance_increase','W_min_eigenvalue','rounding_error_norm']
    failures={name:[r for r in rows if not r.get('passed',False)] for name,rows in [('branches',branches),('qp',qp),('controls',controls),('repeats',repeats)]}
    complete=len(completed)==48 and len(branches)==288 and len(qp)==144 and len(controls)==48 and len(diagnostics)==8 and len(repeats)==8
    result=dict(stage=1,scope='Full development numerical audit, not semantic quality validation',model='SD3 Medium',precision='FP32',resolution=512,T=50,n_max=33,n_min=0,n_avg=1,source_guidance=3.5,target_guidance=13.5,
        source_images=24,seeds=[0,1],completed_trajectories=len(completed),branch_audits=len(branches),qp_states=len(qp),negative_control_states=len(controls),negative_control_scope='One-step controls at three selected states of 16 sources, seed 0; not separate complete null-edit trajectories',conservativity_states=len(diagnostics),repeatability_branch_audits=len(repeats),
        complete=complete,passed=complete and not missing and not errors and all(r['passed'] for r in completed) and not any(failures.values()),
        branch_metrics={k:metric(branches,k) for k in fields},qp_metrics={k:metric(qp,k) for k in qp_fields},conservativity=diagnostics,repeatability=repeats,all_noise_streams_unchanged=all(r.get('noise_unchanged',False) for r in completed),missing=missing,errors=errors,failures=failures,
        thresholds={'duality':1e-4,'baseline_primal':1e-4,'qp_kkt':1e-7,'fp32_rounded_constraints':1e-7,'repeat_relative_error':1e-6},
        training_steps=0,heldout_evaluated=False)
    for seed in [0,1]:
        path=OUTPUT/f'worker_seed{seed}.json'
        if path.exists():result[f'worker_seed{seed}']=json.loads(path.read_text())
    result['worker_shards']=[json.loads(p.read_text()) for p in OUTPUT.glob('worker_seed*_shard*.json')]
    result['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'experiments/stage1_full/run_audit.py',ROOT/'experiments/stage1_full/run_audit_shard.py',ROOT/'experiments/stage1/run_sd3_compute_audit.py',ROOT/'src/correctflow/rank2_qp.py',ROOT/'vendor/flowedit/FlowEdit_utils.py']}
    (OUTPUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    for name,rows in [('branches',branches),('qp',qp),('controls',controls)]:
        (OUTPUT/f'{name}.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    print(json.dumps({k:result[k] for k in ['complete','passed','completed_trajectories','branch_audits','qp_states','negative_control_states','conservativity_states','repeatability_branch_audits']}))

if __name__=='__main__':main()
