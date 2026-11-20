"""Fixed clean input, varying auxiliary model time; diagnostics, not a method swap."""
import argparse,gc,hashlib,json,time
from pathlib import Path
import torch
from diffusers import SD3Transformer2DModel
from torch.nn.attention import SDPBackend,sdpa_kernel
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).parent/'outputs'
ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:0');args=ap.parse_args()
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
device=torch.device(args.device);dt=torch.float64;start=time.monotonic()
payload=torch.load(ROOT/'experiments/stage1_full/inputs/bear__00.pt',weights_only=True)
states=torch.load(ROOT/'experiments/stage1_full/outputs/bear__00__seed0/states.pt',weights_only=True)
snap=next(s for s in states['snapshots'] if s['step']==33)
ys=[snap['z'].to(device=device,dtype=dt),snap['zE'].to(device=device,dtype=dt)]
es,et=[tuple(t.to(device=device,dtype=dt) for t in e) for e in payload['embeddings']]
seq,neg,pool,negpool=et;mask=torch.ones_like(seq);mask[:,:77,2048:]=0
model=SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',torch_dtype=dt,local_files_only=True).eval().requires_grad_(False).to(device)
def pred(y,e,t):
 return model(hidden_states=y,timestep=torch.tensor([t*1000],device=device,dtype=dt),encoder_hidden_states=e[0],pooled_projections=e[2],return_dict=False)[0]
rows=[];g0=None
with sdpa_kernel(SDPBackend.MATH):
 with torch.no_grad():
  vc=pred(ys[1],et,0.);vn=pred(ys[1],(neg,neg,negpool,negpool),0.);v=vn+13.5*(vc-vn);a=(-v/v.norm()).detach()
 for tau in [0.,.001,.01,.1,.5]:
  gs=[]
  for y in ys:
   s=seq.detach().requires_grad_(True);p=pool.detach().requires_grad_(True)
   vt=pred(y,(s,neg,p,negpool),tau)
   grad=torch.autograd.grad(13.5*(vt*a).sum(),(s,p))
   gs.append(torch.cat([(grad[0]*mask).flatten(),grad[1].flatten()]).detach().cpu())
   del vt,grad,s,p;gc.collect();torch.cuda.empty_cache()
  with torch.no_grad():
   vt=pred(ys[1],et,tau);vs=pred(ys[1],es,tau)
  if g0 is None:g0=gs[1]
  row=dict(auxiliary_time=tau,input='same clean y',fixed_probe_from_endpoint=True,reference_gradient_norm=float(gs[0].norm()),candidate_gradient_norm=float(gs[1].norm()),cosine_previous_current=float(gs[0]@gs[1]/(gs[0].norm()*gs[1].norm())),cosine_to_endpoint_gradient=float(g0@gs[1]/(g0.norm()*gs[1].norm())),bare_condition_switch_norm=float((vt-vs).norm()),bare_velocity_plus_y_relative=float((vt+ys[1]).norm()/ys[1].norm()))
  rows.append(row);print(json.dumps(row),flush=True)
  (OUT/'definition_auxiliary_time_fp64.json').write_text(json.dumps(dict(pair='bear__00',step=33,dtype='float64',complete=False,rows=rows,elapsed_seconds=time.monotonic()-start),indent=2)+'\n')
  del gs,vt,vs;gc.collect();torch.cuda.empty_cache()
(OUT/'definition_auxiliary_time_fp64.json').write_text(json.dumps(dict(pair='bear__00',step=33,dtype='float64',complete=True,rows=rows,elapsed_seconds=time.monotonic()-start,code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
print('DEFINITION_AUDIT_COMPLETE',flush=True)
