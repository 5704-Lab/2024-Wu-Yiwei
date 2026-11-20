"""FP64 independent direction derivative check, including clamp active sets."""
import argparse,json,time
from pathlib import Path
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from run_precheck import CleanScorer,ROOT,OUT
ap=argparse.ArgumentParser();ap.add_argument('--pair',default='bear__00');args=ap.parse_args()
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
device=torch.device('cuda:0');dtype=torch.float64;start=time.monotonic()
inp=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{args.pair}.pt',weights_only=True)
states=torch.load(ROOT/'experiments/siglip_state_response/baseline_states'/f'{args.pair}.pt',weights_only=True)
s=next(s for s in states['snapshots'] if s['step']==33)
y=s['z'].to(device=device,dtype=dtype);yb=s['zE'].to(device=device,dtype=dtype)
sc=CleanScorer(device,dtype,'native',True);e,_=sc.encode([inp['record']['target_prompt']])
tensors=torch.load(OUT/f'{args.pair}_step33_float64_native_tensors.pt',weights_only=True)
ur=tensors['reference_direction'].to(device)
b=tensors['direction_correction'].to(device);gu=tensors['semantic_direction'].to(device)
pd=gu+b/b.norm();pd=pd/pd.norm();expected=float((b*pd).sum());d=float((yb-y).norm())
def values(point):
 q=point.detach().requires_grad_(True);ell=sc.logits(q,e)[0,0];g=torch.autograd.grad(ell,q)[0]
 align=float((ur*g/g.norm()).sum())
 with torch.no_grad():
  raw=sc.vae.decode(point/sc.vae.config.scaling_factor+sc.vae.config.shift_factor,return_dict=False)[0]
  active=((raw<-1).to(torch.int8)-(raw>1).to(torch.int8)).cpu()
 return align,active
rows=[]
with sdpa_kernel(SDPBackend.MATH):
 f0,m0=values(yb)
 for rel in [1e-3,1e-4,1e-5,1e-6,1e-7]:
  eps=rel*d;plus,mp=values(yb+eps*pd);minus,mm=values(yb-eps*pd);fd=(plus-minus)/(2*eps)
  rows.append(dict(relative_to_original_update=rel,analytic_direction_derivative=expected,central_direction_derivative=fd,relative_error=abs(fd-expected)/abs(expected),clamp_changes_plus=int((mp!=m0).sum()),clamp_changes_minus=int((mm!=m0).sum())))
  print(json.dumps(rows[-1]),flush=True)
  (OUT/'direction_independent_fp64.json').write_text(json.dumps(dict(pair=args.pair,complete=False,rows=rows),indent=2)+'\n')
(OUT/'direction_independent_fp64.json').write_text(json.dumps(dict(pair=args.pair,complete=True,rows=rows,seconds=time.monotonic()-start),indent=2)+'\n')
