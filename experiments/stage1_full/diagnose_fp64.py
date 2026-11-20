"""Targeted FP64 reference for numerical outliers; preserves original FP32 verdict."""
import argparse,gc,hashlib,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from diffusers import SD3Transformer2DModel
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'vendor/flowedit'))
from FlowEdit_utils import calc_v_sd3
OUT=ROOT/'experiments/stage1_full/outputs'

def cosine(a,b):
 a,b=a.flatten().double().cpu(),b.flatten().double().cpu();return float(a@b/(a.norm()*b.norm()))
def relative(a,b):
 a,b=a.double().cpu(),b.double().cpu();return float((a-b).norm()/(1+b.norm()))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda:0');parser.add_argument('--max-cases',type=int,default=3);args=parser.parse_args()
 summary=json.loads((OUT/'summary.json').read_text());assert summary['complete']
 candidates=summary['failures']['branches'];chosen=[]
 if candidates:
  chosen.append(max(candidates,key=lambda r:r['dual_residual']))
  chosen.append(max(candidates,key=lambda r:r['baseline_primal_error']))
  chosen.extend(sorted(candidates,key=lambda r:r['dual_residual'],reverse=True))
 selected=[];seen=set()
 for row in chosen:
  key=(row['pair_id'],row['seed'],row['step'],row['branch'])
  if key not in seen:selected.append(row);seen.add(key)
  if len(selected)>=args.max_cases:break
 if not selected:
  print('No outliers; no FP64 rerun needed.');return
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 device=torch.device(args.device);model=SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',torch_dtype=torch.float64,local_files_only=True).eval().requires_grad_(False).to(device)
 pipe=SimpleNamespace(transformer=model,do_classifier_free_guidance=True);meta=json.loads((ROOT/'experiments/stage1_full/inputs/manifest.json').read_text());results=[]
 for row in selected:
  start=time.monotonic();torch.cuda.reset_peak_memory_stats(device)
  saved=torch.load(OUT/f"{row['pair_id']}__seed{row['seed']}"/'states.pt',map_location='cpu',weights_only=True)
  snap=next(s for s in saved['snapshots'] if s['step']==row['step']);branch=0 if row['branch']=='source' else 1
  embeddings=torch.load(ROOT/'experiments/stage1_full/inputs'/f"{row['pair_id']}.pt",weights_only=True)['embeddings']
  es,ea=[tuple(v.to(device=device,dtype=torch.float64) for v in e) for e in embeddings]
  seq,neg,pool,negpool=(es if branch==0 else ea)
  q=snap['qs' if branch==0 else 'qa'].to(device=device,dtype=torch.float64);tn=torch.tensor(meta['timesteps'][row['step']],device=device,dtype=torch.float64).expand(1);beta=row['beta']
  with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
   uncond=model(hidden_states=q,timestep=tn,encoder_hidden_states=neg,pooled_projections=negpool,return_dict=False)[0]
  def fn(sequence,pooled):
   pred=model(hidden_states=q,timestep=tn,encoder_hidden_states=sequence,pooled_projections=pooled,return_dict=False)[0];return uncond+beta*(pred-uncond)
  reference=snap['branch_tensors'][branch];probe=(-snap['h']*reference['velocity']).to(device=device,dtype=torch.float64).detach()
  gen=torch.Generator('cpu').manual_seed(20261005+row['step'])
  tangents=tuple(torch.randn(v.shape,generator=gen,dtype=torch.float32).to(device=device,dtype=torch.float64) for v in [seq,pool])
  with torch.enable_grad(),sdpa_kernel(SDPBackend.MATH):
   velocity,pullback=torch.func.vjp(fn,seq,pool);g=pullback(probe)
  del pullback
  with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
   _,jdelta=torch.func.jvp(fn,(seq,pool),tangents)
   _,response=torch.func.jvp(fn,(seq,pool),tuple(v.detach() for v in g))
  lhs=sum((gg*dd).sum() for gg,dd in zip(g,tangents));rhs=(probe*jdelta).sum()
  grad=torch.cat([v.flatten().detach().cpu() for v in g]);g32=torch.cat([reference['gseq'].flatten(),reference['gpool'].flatten()])
  result=dict(pair_id=row['pair_id'],seed=row['seed'],step=row['step'],branch=row['branch'],fp32_dual_residual=row['dual_residual'],fp64_dual_residual=float(abs(lhs-rhs)/(1+abs(lhs)+abs(rhs))),fp64_dual_absolute_error=float(abs(lhs-rhs)),gradient_cosine_fp32_fp64=cosine(g32,grad),response_cosine_fp32_fp64=cosine(reference['response'],response),response_relative_error_fp32_fp64=relative(reference['response'],response),velocity_relative_error_fp32_fp64=relative(reference['velocity'],velocity.detach()))
  velocity_cpu=velocity.detach().cpu()
  del g,grad,jdelta,response,velocity
  gc.collect();torch.cuda.empty_cache()
  # Matched FP64 batched and isolated forwards assess the FP32 batch-size discrepancy.
  with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
   latent=torch.cat([snap['qs'],snap['qs'],snap['qa'],snap['qa']]).to(device=device,dtype=torch.float64)
   vs,va=calc_v_sd3(pipe,latent,torch.cat([es[1],es[0],ea[1],ea[0]]),torch.cat([es[3],es[2],ea[3],ea[2]]),3.5,13.5,tn[0])
  result.update(fp64_batched_isolated_primal_error=relative([vs,va][branch],velocity_cpu),peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device),elapsed_seconds=time.monotonic()-start)
  results.append(result)
  (OUT/'fp64_outlier_diagnostics.json').write_text(json.dumps(dict(scope='Targeted reference, not full FP64 gate or semantic evidence',results=results,original_fp32_gate_unchanged=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
  print(json.dumps(result),flush=True);gc.collect();torch.cuda.empty_cache()
 print('DIAGNOSTICS_COMPLETE',flush=True)

if __name__=='__main__':main()
