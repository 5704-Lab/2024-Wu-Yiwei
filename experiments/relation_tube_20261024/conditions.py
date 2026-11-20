"""Diagnose category/attribute wording and explicit source orientation facts.

Orientation facts are human-verified source annotations for this diagnostic.
This is not an automatic source-fact extractor or a new general parser.
"""
import gc,hashlib,json,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from diffusers import StableDiffusion3Pipeline,SD3Transformer2DModel
from torch.nn.attention import sdpa_kernel,SDPBackend
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'phase_permission'))
from common import ROOT,CleanScorer,seed_setup,write,save_image,Phase
sys.path.insert(0,str(HERE.parent/'joint_permission'));from geometry import Geometry
sys.path.insert(0,str(ROOT/'vendor/flowedit'));from FlowEdit_utils import calc_v_sd3

def captions():
 result={}
 for pair in ['bear__00','brown_owl__00']:
  data=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True)
  s,t=[data['record'][k] for k in ['source_prompt','target_prompt']]
  if pair=='bear__00':
   sa=s.replace('large brown bear','large bear with brown fur')
   ta=t.replace('large black bear','large bear with black fur').replace('the black bear','the bear')
   h=' The bear is in full side profile, with its head and body facing right.'
  else:
   sa='A small owl with brown feathers standing on a patch of grass.'
   ta='A small owl with white feathers standing on a patch of grass.'
   h=' The owl is angled toward the left, with its beak pointing left.'
  result[pair]=dict(BASE=[s,t],ATTRIBUTE=[sa,ta],POSE=[s+h,t+h],ATTRIBUTE_POSE=[sa+h,ta+h])
 return result

def encode(texts,out):
 path=out/'encodings.pt'
 if path.exists():return torch.load(path,weights_only=True)
 pipe=StableDiffusion3Pipeline.from_pretrained(ROOT/'pretrained_models/sd3-medium',torch_dtype=torch.float32,local_files_only=True,transformer=None,vae=None)
 unique=list(dict.fromkeys(t for variants in texts.values() for pair in variants.values() for t in pair));items={t:{} for t in unique}
 token_lengths={}
 for i,enc in enumerate([pipe.text_encoder,pipe.text_encoder_2]):
  enc.eval().requires_grad_(False).to('cuda:0')
  tokenizer=pipe.tokenizer if i==0 else pipe.tokenizer_2
  for t in unique:
   token_lengths.setdefault(t,[]).append(len(tokenizer(t,truncation=False)['input_ids']))
   with torch.no_grad():s,p=pipe._get_clip_prompt_embeds(t,device='cuda:0',clip_model_index=i)
   items[t][f'c{i}']=(s.cpu(),p.cpu())
  enc.cpu();torch.cuda.empty_cache()
 pipe.text_encoder_3.eval().requires_grad_(False).to('cuda:0')
 for t in unique:
  with torch.no_grad():items[t]['t5']=pipe._get_t5_prompt_embeds(t,device='cuda:0',max_sequence_length=256).cpu()
 pipe.text_encoder_3.cpu();del pipe;gc.collect();torch.cuda.empty_cache()
 encoded={}
 for t,it in items.items():
  c0,p0=it['c0'];c1,p1=it['c1'];c=torch.cat([c0,c1],-1)
  encoded[t]=(torch.cat([F.pad(c,(0,it['t5'].shape[-1]-c.shape[-1])),it['t5']],1),torch.cat([p0,p1],-1))
 checks=[]
 for pair,variants in texts.items():
  data=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True)
  for i,t in enumerate(variants['BASE']):
   old=data['embeddings'][i];seq,pool=encoded[t]
   error=max(float((seq-old[0]).abs().max()),float((pool-old[2]).abs().max()))
   assert error<2e-5,(pair,error)
   checks.append(dict(pair=pair,condition=i,maximum_error=error))
 payload=dict(embeddings=encoded,token_lengths=token_lengths,reproduction=checks);out.mkdir(parents=True,exist_ok=True);torch.save(payload,path)
 return payload

def main():
 seed_setup();out=HERE/'outputs/conditions';texts=captions();payload=encode(texts,out)
 write(out/'protocol.json',dict(captions=texts,seeds=[0,1],source_cfg=3.5,target_cfg=13.5,
  source_facts='human-verified source orientation, not automatic extraction',encoding_checks=payload['reproduction'],
  token_lengths=payload['token_lengths'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
 sc=CleanScorer(torch.device('cuda:0'),torch.float32,'native')
 model=SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to(sc.device)
 pipe=SimpleNamespace(transformer=model,do_classifier_free_guidance=True)
 times=torch.tensor(json.loads((ROOT/'experiments/stage1_full/inputs/manifest.json').read_text())['timesteps'],device=sc.device)
 summaries=[]
 for pair,variants in texts.items():
  data=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True);x=data['x'].to(sc.device)
  original=data['embeddings']
  e,_=sc.encode([data['record']['target_prompt'],data['record']['source_prompt']])
  with torch.no_grad():source=sc.image(x)
  geom=Geometry(source);phase=Phase(source)
  for seed in [0,1]:
   for variant,(source_text,target_text) in variants.items():
    dest=out/pair/f'seed{seed}'/variant
    if (dest/'complete.json').exists():continue
    sp,sl=[z.to(sc.device) for z in payload['embeddings'][source_text]];tp,tl=[z.to(sc.device) for z in payload['embeddings'][target_text]]
    seq=torch.cat([original[0][1].to(sc.device),sp,original[1][1].to(sc.device),tp])
    pool=torch.cat([original[0][3].to(sc.device),sl,original[1][3].to(sc.device),tl])
    gen=torch.Generator('cpu').manual_seed(seed);noise=[torch.randn(x.shape,generator=gen).to(sc.device) for _ in range(33)]
    y=x.clone();states=[y.cpu()];rows=[];start=time.monotonic()
    for k in range(17,50):
     t=times[k]/1000;tn=times[k+1]/1000 if k<49 else torch.zeros_like(t)
     qs=(1-t)*x+t*noise[k-17];qa=qs+y-x
     with torch.no_grad():
      vs,vt=calc_v_sd3(pipe,torch.cat([qs,qs,qa,qa]),seq,pool,3.5,13.5,times[k]);y=y-(t-tn)*(vt-vs)
     assert torch.isfinite(y).all();states.append(y.cpu());rows.append(dict(step=k))
    with torch.no_grad():
     im=sc.image(y);s=sc.logits_image(im,e)[0]
     endpoint=dict(full_target=float(s[0]),full_source=float(s[1]),geometry=float(geom.loss(geom(im),40)),
      rgb_rmse=float((im-source).square().mean().sqrt()),background_top64_rmse=float((im[:,:,:64]-source[:,:,:64]).square().mean().sqrt()),**phase.diagnostics(im))
    if seed==0 and variant=='BASE':
     base=torch.load(ROOT/'experiments/flowedit_frequency_trajectory/outputs'/pair/'trajectory.pt',weights_only=True)
     maximum=float((torch.stack(states)-base['states'][17:]).abs().max());assert maximum<2e-5,(pair,maximum)
     endpoint['baseline_all_state_error']=maximum
    save_image(im,dest/'final.png');torch.save(dict(states=torch.stack(states),final=y.cpu()),dest/'trajectory.pt')
    result=dict(pair=pair,seed=seed,variant=variant,rows=rows,source_text=source_text,target_text=target_text,
     endpoint=endpoint,seconds=time.monotonic()-start)
    write(dest/'complete.json',result);summaries.append(result)
    write(out/'summary.json',dict(rows=summaries));print(pair,seed,variant,endpoint,time.monotonic()-start,flush=True)
 print('CONDITIONS_COMPLETE',flush=True)

if __name__=='__main__':
 with sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]):main()
