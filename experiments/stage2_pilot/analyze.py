"""Paired comparisons on common coverage; no group-independence claim."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/stage2_pilot/outputs'
s=json.loads((OUT/'summary.json').read_text());assert s['complete']
rows=[r for c in s['cases'] for r in c['paired_interventions'] if not r['missing']];common={};by={}
for outcome in ['target','preference']:
 v=[r for r in rows if r[f'local_{outcome}_valid'] and r[f'finite_{outcome}_valid']]
 common[outcome]={'pairs':len(v),'local_correct':sum(r[f'local_{outcome}_sign_correct'] for r in v),'finite_correct':sum(r[f'finite_{outcome}_sign_correct'] for r in v),'different_prediction_signs':sum((r['local_prediction']>0)!=(r['finite_prediction']>0) for r in v)}
for d in ['anchor','target_response','source_response','isotropic_qp','random_rank1']:
 by[d]={}
 for outcome in ['target','preference']:
  v=[r for r in rows if r['direction']==d and r[f'local_{outcome}_valid']]
  by[d][outcome]={'valid':len(v),'correct':sum(r[f'local_{outcome}_sign_correct'] for r in v)}
anchors=[{k:r[k] for k in ['pair_id','local_prediction','finite_prediction','delta_target_plus_minus','delta_preference_plus_minus']} for r in rows if r['direction']=='anchor']
report={'common_eligible_comparison':common,'local_by_direction':by,'anchor_direction_endpoints':anchors,'scope':'Descriptive four independent source groups, not 20 independent trials; no semantic success-rate inference','decision':'Do not expand the radial-probe controller or claim finite-work improvement; investigate task proxy and endpoint propagation first'}
(OUT/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
