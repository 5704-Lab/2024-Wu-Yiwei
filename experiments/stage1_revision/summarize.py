import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/stage1_revision/outputs'
protocol=json.loads((OUT.parent/'protocol.json').read_text());rows=[];reference=[];missing=[];errors=[]
for r in protocol['states']:
 dest=OUT/f"{r['pair_id']}__seed{r['seed']}__step{r['step']}"
 for prec in ['fp32']+(['fp64'] if r['fp64_reference'] else []):
  p=dest/f'{prec}.json'
  if p.exists():(rows if prec=='fp32' else reference).append(json.loads(p.read_text()))
  else:missing.append({'record':r,'precision':prec})
  e=dest/f'{prec}_error.json'
  if e.exists():errors.append({'record':r,'precision':prec,'resolved':p.exists(),'error':json.loads(e.read_text())['error']})
branches=[b for c in rows for b in c['branches']];refs=[b for c in reference for b in c['branches']]
complete=len(rows)==26 and len(reference)==3
result={'version':'R1','scope':protocol['scope'],'complete':complete,'passed':complete and all(c['passed'] for c in rows+reference),'states':len(rows),'fp32_branches':len(branches),'fp64_states':len(reference),'fp64_branches':len(refs),'missing':missing,'execution_errors':errors,'old_gate_unchanged':True,
 'metrics':{k:max([b[k] for b in branches] or [0]) for k in ['normwise_dual_residual','energy_identity_normwise','matched_interface_primal','restored_original_scalar_dual_residual']},
 'min_fp64_response_cosine':min([b['paired_response_cosine'] for b in refs] or [1]),'max_same_anchor_qp_relative':max([c['qp']['same_anchor_qp_change_over_radius'] for c in reference] or [0]),'max_full_qp_relative':max([c['qp']['full_qp_change_over_radius'] for c in reference] or [0]),
 'failures':[c for c in rows+reference if not c['passed']],
 'code_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'src/correctflow/conditional_response.py',OUT.parent/'run.py',OUT.parent/'protocol.json']}}
(OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['complete','passed','states','fp32_branches','fp64_branches','metrics','min_fp64_response_cosine','max_full_qp_relative']},ensure_ascii=False))
