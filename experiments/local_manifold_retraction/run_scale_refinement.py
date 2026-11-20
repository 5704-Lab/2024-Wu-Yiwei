"""Post-local-gate diagnostic: smaller scales, no editing parameter tuning."""
import hashlib,json,time
from pathlib import Path
import torch
from diffusers import SD3Transformer2DModel
from run_real import ROOT,OUT,PAIRS,CONFIG
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
scales=[.00625,.0125,.025,.05];device=torch.device('cuda:1');rows=[]
protocol=dict(reason='original 16-state scale audit did not support quadratic expansion',scales=scales,precision='float32',basis='same clean states and empty condition',code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),does_not_change_full_parameters=True)
(OUT/'protocol_refinement.json').write_text(json.dumps(protocol,indent=2)+'\n')
model=SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to(device)
for pair in PAIRS:
 inp=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True);states=torch.load(ROOT/'experiments/siglip_state_response/baseline_states'/f'{pair}.pt',weights_only=True)
 es=tuple(t.to(device) for t in inp['embeddings'][0]);x=inp['x'].to(device)
 gen=torch.Generator('cpu').manual_seed(CONFIG['geometry_seed']);eps=torch.randn(x.shape,generator=gen).to(device)
 for label,y in [('source',x),('step33',next(s for s in states['snapshots'] if s['step']==33)['z'].to(device))]:
  qs=[];times=[]
  for sigma in scales:
   tau=sigma/(1+sigma)
   for sign in [1.,-1.]:qs.append(y/(1+sigma)+sign*tau*eps);times.append(1000*tau)
  with torch.no_grad():
   q=torch.cat(qs);tt=torch.tensor(times,device=device);v=model(hidden_states=q,timestep=tt,encoder_hidden_states=es[1].expand(8,-1,-1),pooled_projections=es[3].expand(8,-1),joint_attention_kwargs=None,return_dict=False)[0]
   m=q-tt[:,None,None,None]/1000*v;f=[m[2*i:2*i+2].mean(0,keepdim=True) for i in range(4)]
  for j in [0,1]:
   a=f[j+1]-f[j];b=f[j+2]-f[j+1]
   row=dict(pair=pair,point=label,scales=scales[j:j+3],difference_norm_ratio=float(b.norm()/a.norm().clamp_min(1e-30)),first_order_residual=float((b-2*a).norm()/(b.norm()+2*a.norm()).clamp_min(1e-30)),second_order_residual=float((b-4*a).norm()/(b.norm()+4*a.norm()).clamp_min(1e-30)),difference_cosine=float((a*b).sum()/(a.norm()*b.norm()).clamp_min(1e-30)),finite=bool(torch.isfinite(m).all()))
   rows.append(row);print(json.dumps(row),flush=True)
(OUT/'scale_refinement.json').write_text(json.dumps(dict(protocol=protocol,rows=rows),indent=2)+'\n')
