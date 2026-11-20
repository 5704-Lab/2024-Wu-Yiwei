import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'experiments/stage1_adaptive/outputs'
p=json.loads((OUT.parent/'protocol.json').read_text());r1=json.loads((ROOT/'experiments/stage1_revision/outputs/summary.json').read_text());promoted=[];missing=[]
for r in p['states']:
 path=OUT/f"{r['pair_id']}__seed{r['seed']}__step{r['step']}"/'complete.json'
 if path.exists():promoted.append(json.loads(path.read_text()))
 else:missing.append(r)
refs_pass=all(b['passed'] for c in [] for b in c['branches'])
ref_files=list((ROOT/'experiments/stage1_revision/outputs').glob('*/fp64.json'));refs_pass=len(ref_files)==3 and all(json.loads(f.read_text())['passed'] for f in ref_files)
result={'version':'R2','scope':p['scope'],'complete':r1['complete'] and len(promoted)==3,'passed':r1['complete'] and not missing and all(c['passed'] for c in promoted) and refs_pass,'states':26,'branches':52,'fp32_states_retained':23,'fp64_promoted_states':len(promoted),'promoted':promoted,'missing':missing,'parent_R1_passed':r1['passed'],'original_full_gate_passed':False,'fp64_reference_passed':refs_pass}
(OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['complete','passed','fp64_promoted_states','parent_R1_passed','fp64_reference_passed']}))
