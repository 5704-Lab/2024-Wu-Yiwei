"""Audit completed artifacts without rerunning image generation."""
import json
import hashlib
import shutil
import numpy as np
import torch
from common import ROOT,HERE,OUT,PAIRS,Appearance,write,code_hashes

rows=[]
for pair in PAIRS:
    baseline=torch.load(ROOT/'experiments/flowedit_frequency_trajectory/outputs'/pair/'trajectory.pt',weights_only=True)
    for method in ['BASE','PERMIT','RAW_PHASE','NORM_CONTROL']:
        dest=OUT/'full'/pair/method
        result=json.loads((dest/'metrics.json').read_text())
        # These trusted, locally generated artifacts include NumPy scalar QP metadata.
        saved=torch.load(dest/'trajectory.pt',weights_only=False)
        assert len(result['steps'])==33 and saved['states'].shape[0]==34
        assert torch.isfinite(saved['states']).all()
        assert torch.equal(saved['states'][0],baseline['states'][17])
        base_error=None
        if method=='BASE':
            base_error=float((saved['states']-baseline['states'][17:]).abs().max())
            assert base_error==0.
        if method in ['PERMIT','RAW_PHASE']:
            bounds=Appearance(torch.zeros(1,3,128,128),2).bound.numpy()
            for r in result['steps']:
                assert r['accepted']
                assert r['task_after']>=r['task_threshold']-.010001
                assert r['correction_norm']<=r['radius']*(1+2e-5)
                assert np.max(np.abs(np.asarray(r['theta']))/bounds)<=1.00001
                assert r['qp']['stationarity']<1e-8
        rows.append(dict(pair=pair,method=method,steps=33,baseline_max_absolute_error=base_error,
                         accepted=sum(r['accepted'] for r in result['steps']),
                         nonzero_corrections=sum(r['correction_norm']>1e-8 for r in result['steps'])))
local=json.loads((OUT/'local/results.json').read_text())['rows']
mechanism=json.loads((OUT/'mechanism.json').read_text())['rows']
stress=json.loads((OUT/'appearance_stress.json').read_text())['rows']
response=json.loads((OUT/'response.json').read_text())['rows']
assert len(local)==36 and len(mechanism)==216 and len(stress)==24 and len(response)==12
assert max(r['phase_additive_error'] for r in response)<1e-5
assert json.loads((OUT/'solver_checks.json').read_text())['passed']==20
snapshot=OUT/'code_snapshot';snapshot.mkdir(exist_ok=True)
for p in HERE.glob('*.py'):shutil.copy2(p,snapshot/p.name)
summary=dict(complete=True,trajectories=len(rows),active_steps=sum(r['steps'] for r in rows),
             local_candidates=len(local),mechanism_cases=len(mechanism),out_of_family_cases=len(stress),
             response_states=len(response),rows=rows,code_sha256=code_hashes(),
             scope='numerical/implementation checks pass; semantic effectiveness is NOT asserted')
write(OUT/'completion_audit.json',summary)
print({k:summary[k] for k in ['complete','trajectories','active_steps','local_candidates','mechanism_cases','out_of_family_cases','response_states']})
