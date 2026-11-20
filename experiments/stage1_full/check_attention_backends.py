"""Targeted same-state backend/batching diagnosis, not a changed gate."""
import json,sys
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from diffusers import SD3Transformer2DModel
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'vendor/flowedit'))
from FlowEdit_utils import calc_v_sd3
from experiments.stage1.run_sd3_compute_audit import relative_error
OUT=ROOT/'experiments/stage1_full/outputs'
torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
model=SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to('cuda:0')
pipe=SimpleNamespace(transformer=model,do_classifier_free_guidance=True);rows=[]
for pair,seed,step in [('flowers_draw__00',1,43),('castle__00',1,23)]:
 saved=torch.load(OUT/f'{pair}__seed{seed}'/'states.pt',weights_only=True);s=next(s for s in saved['snapshots'] if s['step']==step)
 emb=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True)['embeddings'];es,ea=[tuple(t.to('cuda:0') for t in e) for e in emb]
 inputs=torch.cat([s['qs'],s['qs'],s['qa'],s['qa']]).to('cuda:0');seq=torch.cat([es[1],es[0],ea[1],ea[0]]);pool=torch.cat([es[3],es[2],ea[3],ea[2]])
 meta=json.loads((ROOT/'experiments/stage1_full/inputs/manifest.json').read_text());tn=torch.tensor(meta['timesteps'][step],device='cuda:0')
 with sdpa_kernel(SDPBackend.MATH):vs,va=calc_v_sd3(pipe,inputs,seq,pool,3.5,13.5,tn)
 for i,v in enumerate([vs,va]):
  reference=s['branch_tensors'][i]['velocity'].to('cuda:0')
  rows.append(dict(pair_id=pair,seed=seed,step=step,branch=['source','target'][i],matched_math_backend_primal_error=relative_error(v,reference)))
(OUT/'attention_backend_diagnostics.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows),flush=True)
