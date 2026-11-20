"""Reproduce original FlowEdit final segment to obtain the missing step47."""
import hashlib,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch
from diffusers import SD3Transformer2DModel
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'vendor/flowedit'))
from FlowEdit_utils import calc_v_sd3
OUT=Path(__file__).parent/'baseline_states';OUT.mkdir(exist_ok=True)
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
device=torch.device('cuda:0')
meta=json.loads((ROOT/'experiments/stage1_full/inputs/manifest.json').read_text())
times=torch.tensor(meta['timesteps'],device=device,dtype=torch.float32)
model=SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to(device)
pipe=SimpleNamespace(transformer=model,do_classifier_free_guidance=True)
for pair in ['bear__00','brown_owl__00','bus__00','cat_crown__00']:
 start=time.monotonic();inp=ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt';oldpath=ROOT/'experiments/stage1_full/outputs'/f'{pair}__seed0'/'states.pt'
 payload=torch.load(inp,weights_only=True);old=torch.load(oldpath,weights_only=True);snap=next(s for s in old['snapshots'] if s['step']==43)
 x=payload['x'].to(device);y=snap['z'].to(device)
 es,et=[tuple(t.to(device) for t in e) for e in payload['embeddings']]
 seq=torch.cat([es[1],es[0],et[1],et[0]]);pool=torch.cat([es[3],es[2],et[3],et[2]])
 gen=torch.Generator('cpu').manual_seed(0);noises=[torch.randn(x.shape,generator=gen,dtype=torch.float32) for _ in range(33)]
 samples=[next(s for s in old['snapshots'] if s['step']==33)]
 with torch.no_grad():
  for k in range(43,50):
   t=times[k]/1000;tnext=times[k+1]/1000 if k<49 else torch.zeros_like(t);h=float(t-tnext);qs=(1-t)*x+t*noises[k-17].to(device);qa=y+qs-x
   vs,va=calc_v_sd3(pipe,torch.cat([qs,qs,qa,qa]),seq,pool,3.5,13.5,times[k]);d=-h*(va-vs);yb=y+d
   if k==47:samples.append(dict(step=k,t=float(t),h=h,z=y.cpu(),zE=yb.cpu(),delta=d.cpu()))
   y=yb
 final_error=float((y.cpu().double()-old['baseline_final'].double()).norm()/old['baseline_final'].double().norm())
 torch.save(dict(record=payload['record'],snapshots=samples,baseline_final=y.cpu(),cache_source_sha256=hashlib.sha256(oldpath.read_bytes()).hexdigest(),final_reproduction_relative_error=final_error),OUT/f'{pair}.pt')
 print(json.dumps(dict(pair=pair,step47=True,final_relative_error=final_error,seconds=time.monotonic()-start)),flush=True)
