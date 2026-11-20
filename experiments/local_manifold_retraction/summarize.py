"""Audit completed cohorts and summarize proxy/scale evidence."""
import json
from pathlib import Path
import numpy as np
import torch
from run_real import OUT

toy=json.loads((OUT/'toy.json').read_text());local=json.loads((OUT/'local.json').read_text())['rows'];shared=[r for r in local if r['condition']=='EMPTY']
summary=dict(toy_passed=toy['passed'],toy_checks=len(toy['checks']),local_rows=len(local),shared_points=len(shared),local_scale={})
for field in ['first_order_residual','second_order_residual','difference_norm_ratio','dual_scale_relative_difference']:
 vals=[r[field] for r in shared];summary['local_scale'][field]=dict(mean=float(np.mean(vals)),min=min(vals),max=max(vals))
summary['local_scale']['first_order_better_count']=sum(r['first_order_residual']<r['second_order_residual'] for r in shared)
summary['local_scale']['repeat_dual_cosines']=[r['repeat_dual_cosine'] for r in shared if 'repeat_dual_cosine' in r]
ref=json.loads((OUT/'scale_refinement.json').read_text())['rows'];summary['refined_scales']=[]
for scales in [[.00625,.0125,.025],[.0125,.025,.05]]:
 a=[r for r in ref if r['scales']==scales]
 summary['refined_scales'].append(dict(scales=scales,n=len(a),mean_difference_norm_ratio=float(np.mean([r['difference_norm_ratio'] for r in a])),mean_first_order_residual=float(np.mean([r['first_order_residual'] for r in a])),mean_second_order_residual=float(np.mean([r['second_order_residual'] for r in a])),second_order_better_count=sum(r['second_order_residual']<r['first_order_residual'] for r in a)))
summary['cohorts']={}
for label,directory,expected in [('original',OUT,8),('refined',OUT/'refined',4)]:
 results=json.loads((directory/'full.json').read_text())['results'];corrected=[r for r in results if r['method']!='BASE']
 if len(corrected)!=expected:raise RuntimeError('incomplete cohort')
 total=sum(r['optimization_calls'] for r in corrected);nonzero=sum(r['nonzero_corrections'] for r in corrected)
 diagnostics=[]
 for r in corrected:
  saved=torch.load(directory/f"{r['pair']}_{r['method']}_trajectory.pt",weights_only=True)
  rows=saved['rows'];max_ratio=max(x['correction_ratio'] for x in rows)
  if len(rows)!=33 or r['optimization_calls']!=33 or not r['all_finite'] or max_ratio>.05001:raise RuntimeError('execution invariant failed')
  diagnostics.append(dict(pair=r['pair'],method=r['method'],max_actual_correction_ratio=max_ratio,mean_correction_base_cosine=float(np.mean([x['correction_base_cosine'] for x in rows]))))
 summary['cohorts'][label]=dict(results=results,optimization_calls=total,nonzero_corrections=nonzero,diagnostics=diagnostics,baseline_exact=all(r['relative_to_baseline']==0 for r in results if r['method']=='BASE'))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
