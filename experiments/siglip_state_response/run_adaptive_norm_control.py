"""Resolve total-increment magnitude confound without retuning main solver."""
import json
from pathlib import Path
import torch
from run_precheck import CleanScorer,ROOT
from run_adaptive_prox import OUT,PAIRS,CONFIG
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
sc=CleanScorer(torch.device('cuda:1'),torch.float32,'native')
points=torch.load(OUT/'local_states.pt',weights_only=True);rows=[]
for pair in PAIRS:
 states=torch.load(Path(__file__).parent/'baseline_states'/f'{pair}.pt',weights_only=True)
 e,_=sc.encode([states['record']['target_prompt']])
 for s in states['snapshots']:
  step=s['step'];y=s['z'].to(sc.device);ua=points[f'{pair}_{step}_1.0_ADAPT'].to(sc.device)-y
  for method in ['G','ISO','ADAPT','SHUFFLE']:
   u=points[f'{pair}_{step}_1.0_{method}'].to(sc.device)-y
   matched=u*(ua.norm()/u.norm().clamp_min(1e-30));point=y+matched
   with torch.no_grad():ell=float(sc.logits(point,e)[0,0])
   rows.append(dict(pair=pair,step=step,method=method,matched_increment_norm=float((point-y).norm()),target_logit=ell))
 print(json.dumps(dict(pair=pair,complete=True)),flush=True)
(OUT/'norm_control.json').write_text(json.dumps(dict(scope='same total-increment norm at lambda1; proxy-only direction diagnostic',rows=rows),indent=2)+'\n')
