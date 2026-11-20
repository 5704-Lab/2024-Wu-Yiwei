from pathlib import Path
import gc,json,sys,time
import torch
from diffusers import SD3Transformer2DModel
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from correctflow.conditional_response import velocity,response,anchor
from experiments.stage1_revision.run import load,name,save,save_tensor
from experiments.stage1_full.run_audit import qp_audit
from experiments.stage1.run_sd3_compute_audit import relative_error
OUT=ROOT/'experiments/stage1_adaptive/outputs';protocol=json.loads((OUT.parent/'protocol.json').read_text())
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
device=torch.device('cuda:1');model=SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',torch_dtype=torch.float64,local_files_only=True).eval().requires_grad_(False).to(device)
for r in protocol['states']:
 dest=OUT/name(r);dest.mkdir(exist_ok=True)
 if (dest/'complete.json').exists():continue
 s,emb,tn=load(r,device,torch.float64);rows=[];tensors=[];start=time.monotonic()
 for b in [0,1]:
  q=s['qs' if b==0 else 'qa'].to(device=device,dtype=torch.float64);beta=[3.5,13.5][b]
  with torch.no_grad():v0=velocity(model,q,tn,emb[b],beta)
  row,tensor=response(model,q,tn,emb[b],beta,s['h'],audit=True,seed=20261005+r['step'])
  row.update(branch=['source','target'][b],matched_interface_primal=relative_error(tensor['velocity'].to(device),v0))
  row['passed']=row['finite'] and row['normwise_dual_residual']<=1e-6 and row['energy_identity_normwise']<=1e-6 and row['unit_probe_response_work']>=0 and row['matched_interface_primal']<=1e-7
  rows.append(row);tensors.append(tensor);gc.collect();torch.cuda.empty_cache()
 x,z,qs=[s[k].to(device=device,dtype=torch.float64) for k in ['x','z','qs']]
 delta=s['h']*(tensors[0]['velocity'].to(device)-tensors[1]['velocity'].to(device));zE=z+delta;qE=zE+qs-x
 a=anchor(model,qE,tn,torch.tensor(s['t'],device=device,dtype=torch.float64),emb[0],emb[1])
 qp,w=qp_audit(x,zE,delta,a,[t['response'] for t in tensors])
 result={'record':r,'precision':'whole local step FP64, rounded submission FP32','branches':rows,'qp':qp,'passed':all(b['passed'] for b in rows) and qp['passed'],'elapsed_seconds':time.monotonic()-start}
 save_tensor(dest/'states.pt',dict(branches=tensors,x=x.cpu(),zE=zE.cpu(),anchor=a.cpu(),delta=delta.cpu(),qp=w));save(dest/'complete.json',result)
 print(json.dumps({'case':name(r),'passed':result['passed'],'energy_max':max(b['energy_identity_normwise'] for b in rows),'rounded_primal':qp['rounded_primal']}),flush=True)
print('ADAPTIVE_COMPLETE',flush=True)
