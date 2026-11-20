"""Summarize finite-state and every-step mechanism evidence, not oracle quality."""
import json
from pathlib import Path
import numpy as np
import torch
from run_adaptive_prox import OUT,PAIRS
local=json.loads((OUT/'local.json').read_text())['rows']
summary=dict(local=[],full=[],norm_control=[],scope='four development sources; optimized SigLIP is not independent evaluation')
for lam in [.25,1.,4.]:
 for method in ['G','ISO','ADAPT','SHUFFLE']:
  a=[r for r in local if r['nominal_lambda']==lam and r['method']==method]
  summary['local'].append(dict(lam=lam,method=method,n=len(a),mean_logit_vs_baseline=float(np.mean([r['delta_target_vs_baseline'] for r in a])),positive_vs_baseline=sum(r['delta_target_vs_baseline']>0 for r in a),mean_increment_ratio=float(np.mean([r['total_increment_ratio'] for r in a])),max_objective_delta=max(r['proxy_objective_delta'] for r in a),max_budget_ratio=max(r['budget_ratio'] for r in a)))
norm=json.loads((OUT/'norm_control.json').read_text())['rows']
for method in ['G','ISO','SHUFFLE']:
 diffs=[]
 for pair in PAIRS:
  for step in [33,47]:
   a={r['method']:r['target_logit'] for r in norm if r['pair']==pair and r['step']==step}
   diffs.append(a['ADAPT']-a[method])
 summary['norm_control'].append(dict(control=method,mean_adapt_minus_control=float(np.mean(diffs)),positive=sum(d>0 for d in diffs),n=len(diffs)))
for pair in PAIRS:
 for method in ['G','ISO','ADAPT']:
  p=OUT/f'{pair}_{method}_trajectory.pt'
  if not p.exists():raise RuntimeError('incomplete full cohort: '+str(p))
  d=torch.load(p,weights_only=True);rows=d['rows'];s=d['summary'].copy()
  s.update(all_finite=all(r['finite'] for r in rows),mean_increment_ratio=float(np.mean([r['increment_ratio'] for r in rows])),mean_correction_ratio=float(np.mean([r['correction_ratio'] for r in rows])),max_budget_ratio=max(r['budget_ratio'] for r in rows),constraint_active_steps=sum(r['mu']>0 for r in rows))
  if s['optimization_calls']!=33 or not s['all_finite']:raise RuntimeError('every-step or finite-state gate failed')
  summary['full'].append(s)
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False,indent=2))
