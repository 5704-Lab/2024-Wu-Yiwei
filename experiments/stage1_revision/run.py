"""Registered targeted numerical revision on preserved real-model states."""
from pathlib import Path
import argparse,gc,hashlib,json,sys,time,traceback
import numpy as np
import torch
from diffusers import SD3Transformer2DModel
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from correctflow.conditional_response import response,velocity,anchor
from correctflow.rank2_qp import Rank2Weight
from experiments.stage1.run_sd3_compute_audit import dump_json,relative_error
from experiments.stage1_full.run_audit import qp_audit
OUT=ROOT/'experiments/stage1_revision/outputs';protocol=json.loads((OUT.parent/'protocol.json').read_text())
def save(p,x):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(dump_json(x,indent=2)+'\n');tmp.replace(p)
def save_tensor(p,x):
 tmp=p.with_suffix(p.suffix+'.tmp');torch.save(x,tmp);tmp.replace(p)
def cos(a,b):
 a,b=a.flatten().double(),b.flatten().double();return float(a@b/(a.norm()*b.norm()))
def load(record,device,dtype):
 cache=torch.load(ROOT/'experiments/stage1_full/inputs'/f"{record['pair_id']}.pt",weights_only=True)
 states=torch.load(ROOT/'experiments/stage1_full/outputs'/f"{record['pair_id']}__seed{record['seed']}"/'states.pt',weights_only=True)
 s=next(v for v in states['snapshots'] if v['step']==record['step'])
 emb=[tuple(t.to(device=device,dtype=dtype) for t in e) for e in cache['embeddings']]
 meta=json.loads((ROOT/'experiments/stage1_full/inputs/manifest.json').read_text())
 return s,emb,torch.tensor(meta['timesteps'][record['step']],device=device,dtype=dtype)
def name(r):return f"{r['pair_id']}__seed{r['seed']}__step{r['step']}"
def main():
 p=argparse.ArgumentParser();p.add_argument('--device',required=True);p.add_argument('--precision',choices=['fp32','fp64'],required=True);args=p.parse_args()
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 dtype=torch.float32 if args.precision=='fp32' else torch.float64;device=torch.device(args.device)
 records=protocol['states'] if args.precision=='fp32' else [r for r in protocol['states'] if r['fp64_reference']]
 model=SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',torch_dtype=dtype,local_files_only=True).eval().requires_grad_(False).to(device)
 completed=[];start=time.monotonic()
 for r in records:
  dest=OUT/name(r);dest.mkdir(exist_ok=True);done=dest/f'{args.precision}.json'
  if done.exists():completed.append(json.loads(done.read_text()));continue
  try:
   s,emb,tn=load(r,device,dtype);rows=[];tensors=[]
   if args.precision=='fp64':
    deadline=time.monotonic()+600
    while not (dest/'fp32.json').exists():
     if time.monotonic()>deadline:raise RuntimeError('Waiting for FP32 paired input timed out')
     time.sleep(3)
    paired=torch.load(dest/'fp32.pt',weights_only=True)
   for b in [0,1]:
    q=s['qs' if b==0 else 'qa'].to(device=device,dtype=dtype);beta=3.5 if b==0 else 13.5
    with torch.no_grad():v0=velocity(model,q,tn,emb[b],beta)
    override=None if args.precision=='fp32' else paired['branches'][b]['unit_probe'].to(device=device,dtype=dtype)
    torch.cuda.reset_peak_memory_stats(device)
    row,tensor=response(model,q,tn,emb[b],beta,s['h'],seed=20261005+r['step'],audit=True,probe_override=override)
    row.update(branch=['source','target'][b],matched_interface_primal=relative_error(tensor['velocity'].to(device),v0),peak_gpu_bytes=torch.cuda.max_memory_allocated(device))
    th=protocol['thresholds'];row['passed']=bool(row['finite'] and row['normwise_dual_residual']<=th['normwise_duality'] and row['energy_identity_normwise']<=th['energy_identity_normwise'] and row['unit_probe_response_work']>=0 and row['matched_interface_primal']<=th['matched_interface_primal'])
    if args.precision=='fp32':
     old=s['branch_tensors'][b]
     row.update(old_response_cosine=cos(tensor['response'],old['response']),old_primal_relative_error=relative_error(tensor['velocity'],old['velocity']))
    else:
     row.update(paired_response_cosine=cos(tensor['response'],paired['branches'][b]['response']))
     row['passed']=row['passed'] and row['paired_response_cosine']>=th['response_fp32_fp64_cosine']
    rows.append(row);tensors.append(tensor);gc.collect();torch.cuda.empty_cache()
   if args.precision=='fp32':
    x,z,qs=[s[k].to(device) for k in ['x','z','qs']]
    delta=s['h']*(tensors[0]['velocity'].to(device)-tensors[1]['velocity'].to(device));zE=z+delta;qE=zE+qs-x
    a=anchor(model,qE,tn,torch.tensor(s['t'],device=device),emb[0],emb[1])
    qr,w=qp_audit(x,zE,delta,a,[t['response'] for t in tensors]);rnorm=qr['radius']
    Wold=Rank2Weight(*(s['branch_tensors'][b]['response'].numpy() for b in [0,1]))
    old_result=Wold.solve((zE.double()-x.double()).cpu().numpy().reshape(-1),a.double().cpu().numpy().reshape(-1),rnorm)
    qr['same_anchor_old_response_qp_change_over_radius']=float(np.linalg.norm(old_result.displacement-w.numpy().reshape(-1))/max(rnorm,1e-30))
    payload={'branches':tensors,'x':x.cpu(),'zE':zE.cpu(),'delta':delta.cpu(),'qE':qE.cpu(),'anchor':a.cpu(),'qp':w,'radius':rnorm,'record':r}
   else:
    D=(paired['zE'].double()-paired['x'].double()).numpy().reshape(-1);an=paired['anchor'].double().numpy().reshape(-1);radius=paired['radius']
    W=Rank2Weight(*(t['response'].numpy() for t in tensors));same=W.solve(D,an,radius)
    a64=anchor(model,paired['qE'].to(device=device,dtype=dtype),tn,torch.tensor(s['t'],device=device,dtype=dtype),emb[0],emb[1]).cpu()
    full=W.solve(D,a64.numpy().reshape(-1),radius)
    reference=paired['qp'].double().numpy().reshape(-1)
    same_error=float(np.linalg.norm(same.displacement-reference)/max(radius,1e-30));full_error=float(np.linalg.norm(full.displacement-reference)/max(radius,1e-30))
    qr=dict(same_anchor_qp_change_over_radius=same_error,full_qp_change_over_radius=full_error,anchor_cosine=cos(a64,paired['anchor']),passed=same_error<=protocol['thresholds']['same_anchor_qp_relative_to_radius'] and full_error<=protocol['thresholds']['total_qp_relative_to_radius'])
    payload={'branches':tensors,'same_anchor_qp':torch.from_numpy(same.displacement),'full_qp':torch.from_numpy(full.displacement),'anchor':a64,'record':r}
   save_tensor(dest/f'{args.precision}.pt',payload)
   result={'record':r,'precision':args.precision,'branches':rows,'qp':qr,'passed':all(x['passed'] for x in rows) and qr['passed'],'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()};save(done,result);completed.append(result)
   print(dump_json({'case':name(r),'precision':args.precision,'completed':len(completed),'total':len(records),'passed':result['passed'],'dual_max':max(x['normwise_dual_residual'] for x in rows),'qp':qr}),flush=True)
  except Exception:
   err=traceback.format_exc();save(dest/f'{args.precision}_error.json',{'error':err});print(err,flush=True)
  gc.collect();torch.cuda.empty_cache()
  save(OUT/f'worker_{args.precision}.json',{'completed':len(completed),'total':len(records),'elapsed_seconds':time.monotonic()-start})
 print('REVISION_COMPLETE',flush=True)
if __name__=='__main__':main()
