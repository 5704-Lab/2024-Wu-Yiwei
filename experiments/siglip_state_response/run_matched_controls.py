"""Same-radius local controls; not full edited trajectories."""
import json,time
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from run_precheck import CleanScorer,ROOT,OUT
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
sc=CleanScorer(torch.device('cuda:0'),torch.float32,'native')
rows=[];start=time.monotonic()
with sdpa_kernel(SDPBackend.MATH):
 for pair in ['bear__00','brown_owl__00','bus__00','cat_crown__00']:
  states=torch.load(ROOT/'experiments/siglip_state_response/baseline_states'/f'{pair}.pt',weights_only=True)
  et,_=sc.encode([states['record']['target_prompt']])
  for step in [33,47]:
   s=next(s for s in states['snapshots'] if s['step']==step)
   yb=s['zE'].to(sc.device)
   t=torch.load(OUT/f'{pair}_step{step}_float32_native_tensors.pt',weights_only=True)
   r=json.loads((OUT/f'{pair}_step{step}_float32_native_hvp.json').read_text())
   radius=r['joint_trials'][0]['norm'];gu=t['semantic_direction'].to(sc.device)
   b=t['direction_correction'].to(sc.device);pd=gu+b/b.norm();pd=pd/pd.norm()
   ur=t['reference_direction'].to(sc.device)
   def measure(point):
    q=point.detach().requires_grad_(True);ell=sc.logits(q,et)[0,0]
    g=torch.autograd.grad(ell,q)[0]
    return float(ell),float((ur*g/g.norm()).sum())
   l0,f0=measure(yb)
   for name,p in [('G',gu),('D',pd)]:
    l,f=measure(yb+radius*p)
    rows.append(dict(pair=pair,step=step,method=name,radius=radius,delta_logit=l-l0,delta_alignment=f-f0))
   print(json.dumps(rows[-2:]),flush=True)
(OUT/'matched_controls.json').write_text(json.dumps(dict(scope='same-radius local diagnostic; no final-quality claim',rows=rows,seconds=time.monotonic()-start),indent=2)+'\n')
