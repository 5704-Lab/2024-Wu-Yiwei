"""Single-midpoint mechanism pilot with independent evaluation afterwards."""
from pathlib import Path
import argparse,gc,hashlib,json,sys,time
from types import SimpleNamespace
import numpy as np
from PIL import Image,ImageDraw
import torch
from diffusers import SD3Transformer2DModel,AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'vendor/flowedit'))
from correctflow.conditional_response import velocity,response,anchor
from correctflow.rank2_qp import Rank2Weight
from correctflow.comparison import render_comparison
from FlowEdit_utils import calc_v_sd3
OUT=ROOT/'experiments/stage2_pilot/outputs'

def sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def save(p,x):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--device',required=True);ap.add_argument('--cases',required=True);args=ap.parse_args()
 gate=json.loads((ROOT/'experiments/stage1_adaptive/outputs/summary.json').read_text());assert gate['complete'] and gate['passed'],'R2 gate must pass first'
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 device=torch.device(args.device);protocol=json.loads((OUT.parent/'protocol.json').read_text());meta=json.loads((OUT.parent/'inputs/manifest.json').read_text())
 indices=[int(s) for s in args.cases.split(',')];nn=torch.tensor(meta['timesteps'],device=device,dtype=torch.float32);clocks=nn/1000
 model=SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to(device)
 vae=AutoencoderKL.from_pretrained(ROOT/'pretrained_models/sd3-medium/vae',torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to(device)
 processor=VaeImageProcessor(vae_scale_factor=8);pipe=SimpleNamespace(transformer=model,do_classifier_free_guidance=True)
 for index in indices:
  rec=protocol['pairs'][index];dest=OUT/rec['pair_id'];dest.mkdir(exist_ok=True)
  if (dest/'complete.json').exists():continue
  start=time.monotonic();payload=torch.load(OUT.parent/'inputs'/f"{rec['pair_id']}.pt",weights_only=True);x=payload['x'].to(device)
  es,ea=[tuple(t.to(device) for t in e) for e in payload['embeddings']];seq=torch.cat([es[1],es[0],ea[1],ea[0]]);pool=torch.cat([es[3],es[2],ea[3],ea[2]])
  gen=torch.Generator('cpu').manual_seed(0);noises=[torch.randn(x.shape,generator=gen) for _ in range(33)];hashes=[sha(n) for n in noises];gen_state=sha(gen.get_state())
  z=x.clone();zoff=x.clone();arms=None;predictions=[];audit=[]
  for step in range(17,50):
   t=clocks[step];h=float(t-(clocks[step+1] if step<49 else torch.zeros_like(t)));qs=(1-t)*x+t*noises[step-17].to(device)
   # Official reference is always separate, with identical seeded noises.
   qo=zoff+qs-x;vos,voa=calc_v_sd3(pipe,torch.cat([qs,qs,qo,qo]),seq,pool,3.5,13.5,nn[step]);zoff=zoff-h*(voa-vos)
   if step<33:
    vs=velocity(model,qs,nn[step],es,3.5);va=velocity(model,z+qs-x,nn[step],ea,13.5);z=z+h*(vs-va)
   elif step==33:
    model.to(torch.float64);e64=[tuple(v.double() for v in e) for e in [es,ea]];q64=qs.double();qa=(z+qs-x).double();tn=nn[step].double();tv=t.double()
    vectors=[];vbranches=[]
    for b,q in enumerate([q64,qa]):
     row,bt=response(model,q,tn,e64[b],[3.5,13.5][b],h,audit=True,seed=20261005+step)
     row['branch']=['source','target'][b]
     assert row['finite'] and row['normwise_dual_residual']<=1e-6 and row['energy_identity_normwise']<=1e-6 and row['unit_probe_response_work']>=0,row
     audit.append(row);vectors.append(bt['response'].to(device));vbranches.append(bt['velocity'].to(device));gc.collect();torch.cuda.empty_cache()
    delta=h*(vbranches[0]-vbranches[1]);candidate=(z.double()+delta).float();base64=candidate.double();qE=q64+base64-x.double()
    a=anchor(model,qE,tn,tv,e64[0],e64[1]);eps=.1*float(delta.norm())
    zero=torch.zeros_like(a).cpu().numpy();WI=Rank2Weight(zero,zero,lambda_s=0,lambda_u=0)
    iso=WI.solve((base64-x.double()).cpu().numpy().reshape(-1),a.cpu().numpy().reshape(-1),eps).displacement
    aux=torch.Generator('cpu').manual_seed(726001+index);random=torch.randn(a.shape,generator=aux).to(device=device,dtype=torch.float64)
    directions={'anchor':a,'target_response':vectors[1],'source_response':vectors[0],'isotropic_qp':torch.from_numpy(iso.reshape(a.shape)).to(device),'random_rank1':random}
    arms={'baseline':candidate}
    for label,direction in directions.items():
     dn=float(direction.norm())
     if dn<=1e-14:
      predictions.append({'direction':label,'missing':True});continue
     unit=direction/dn
     for sign in [1,-1]:
      key=label+('_plus' if sign>0 else '_minus');point=(base64+sign*eps*unit).float();actual=point.double()-base64
      local=float((a*actual).sum());work=0.
      for j in range(2):
       apoint=anchor(model,qE+((j+.5)/2)*actual,tn,tv,e64[0],e64[1]);work+=float((apoint*actual).sum())/2
      scale=float(a.norm()*actual.norm());predictions.append({'direction':label,'arm':key,'sign':sign,'local_work':local,'finite_work':work,'prediction_scale':scale,'actual_step_norm':float(actual.norm()),'desired_step_norm':eps,'missing':False})
      arms[key]=point
     print(json.dumps({'case':rec['pair_id'],'checkpoint_direction_done':label}),flush=True)
    # Back to FP32 for all continued arms, including the common baseline.
    model.to(torch.float32);gc.collect();torch.cuda.empty_cache()
    print(json.dumps({'case':rec['pair_id'],'checkpoint_complete':True,'arms':len(arms)}),flush=True)
   else:
    vs=velocity(model,qs,nn[step],es,3.5)
    for key,point in arms.items():
     va=velocity(model,point+qs-x,nn[step],ea,13.5);arms[key]=(point+h*(vs-va)).float()
    if step in [39,43,49]:print(json.dumps({'case':rec['pair_id'],'continuation_step':step}),flush=True)
  arms['official']=zoff;arms['vae_reconstruction']=x
  latents={}
  for label,point in arms.items():
   latents[label]=point.cpu()
   with torch.no_grad():pixels=vae.decode(point/vae.config.scaling_factor+vae.config.shift_factor,return_dict=False)[0]
   processor.postprocess(pixels,output_type='pil')[0].save(dest/f'{label}.png')
  original=Image.open(ROOT/rec['source_image']).convert('RGB').resize((512,512),Image.Resampling.BICUBIC);original.save(dest/'source.png')
  comparisons={'source':dest/'source.png','flowedit':dest/'official.png','matched interface':dest/'baseline.png','VAE reconstruction':dest/'vae_reconstruction.png'}
  for key in arms:
   if key not in {'baseline','official','vae_reconstruction'}:comparisons[key]=dest/f'{key}.png'
  render_comparison(comparisons,dest/'overview.png',title=rec['pair_id']+' | seed 0 | checkpoint 33')
  torch.save(latents,dest/'terminal_latents.pt')
  result={'record':rec,'step':33,'predictions':predictions,'checkpoint_audit':audit,'arms':list(arms),'noises_unchanged':hashes==[sha(n) for n in noises] and gen_state==sha(gen.get_state()),'noise_hashes':hashes,'elapsed_seconds':time.monotonic()-start,'model_gradient_updates':0,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'helper_sha256':hashlib.sha256((ROOT/'src/correctflow/conditional_response.py').read_bytes()).hexdigest()}
  save(dest/'complete.json',result);print(json.dumps({'case':rec['pair_id'],'complete':True,'elapsed':result['elapsed_seconds']}),flush=True)
 print('PILOT_GENERATION_COMPLETE',flush=True)
if __name__=='__main__':main()
