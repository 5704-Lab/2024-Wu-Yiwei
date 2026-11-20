"""Actual SD3 conditional flow-matching residual as alternative relation evidence.

No classifier training. This is a surrogate conditional compatibility score, not
an exact likelihood or a demonstrated semantic guarantee.
"""
import sys,json,gc,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import StableDiffusion3Pipeline,AutoencoderKL,SD3Transformer2DModel
HERE=Path(__file__).parent
sys.path.insert(0,str(HERE.parent/'phase_permission'))
from common import ROOT,write,seed_setup
sys.path.insert(0,str(HERE))
from binding import CASES

def encode():
 path=HERE/'outputs/rf_binding_inputs.pt'
 if path.exists():return
 pipe=StableDiffusion3Pipeline.from_pretrained(ROOT/'pretrained_models/sd3-medium',torch_dtype=torch.float32,
     local_files_only=True,transformer=None,vae=None)
 texts=list(dict.fromkeys(t for _,_,pairs in CASES for pair in pairs for t in pair))
 items={t:{} for t in texts}
 for idx,enc in enumerate([pipe.text_encoder,pipe.text_encoder_2]):
  enc.eval().requires_grad_(False).to('cuda:0')
  for t in texts:
   with torch.no_grad():s,p=pipe._get_clip_prompt_embeds(t,device='cuda:0',clip_model_index=idx)
   items[t][f'c{idx}']=(s.cpu(),p.cpu())
  enc.cpu();torch.cuda.empty_cache()
 pipe.text_encoder_3.eval().requires_grad_(False).to('cuda:0')
 for t in texts:
  with torch.no_grad():items[t]['t5']=pipe._get_t5_prompt_embeds(t,device='cuda:0',max_sequence_length=256).cpu()
 pipe.text_encoder_3.cpu();del pipe;gc.collect();torch.cuda.empty_cache()
 embeds={}
 for t,it in items.items():
  c0,p0=it['c0'];c1,p1=it['c1'];seq=torch.cat([c0,c1],-1)
  embeds[t]=(torch.cat([F.pad(seq,(0,it['t5'].shape[-1]-seq.shape[-1])),it['t5']],1),torch.cat([p0,p1],-1))
 vae=AutoencoderKL.from_pretrained(ROOT/'pretrained_models/sd3-medium/vae',torch_dtype=torch.float32,
                                 local_files_only=True).eval().requires_grad_(False).to('cuda:0')
 latents={}
 for name,axis,_ in CASES:
  im=Image.open(ROOT/'data/flowedit/Images'/f'{name}.png').convert('RGB').resize((512,512))
  im=torch.tensor(np.asarray(im).copy(),device='cuda:0',dtype=torch.float32).permute(2,0,1)[None]/255
  ims=torch.cat([im,im.flip(-1 if axis=='horizontal' else -2)])
  with torch.no_grad():z=vae.encode(ims*2-1).latent_dist.mode();z=(z-vae.config.shift_factor)*vae.config.scaling_factor
  latents[name]=z.cpu()
 torch.save(dict(embeddings=embeds,latents=latents),path)
 print('RF_INPUTS_ENCODED',flush=True)

def main():
 seed_setup();encode()
 payload=torch.load(HERE/'outputs/rf_binding_inputs.pt',weights_only=True)
 model=SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',torch_dtype=torch.float32,
                 local_files_only=True).eval().requires_grad_(False).to('cuda:0')
 rows=[]
 for name,axis,pairs in CASES:
  x=payload['latents'][name].to('cuda:0')
  for j,(pos,neg) in enumerate(pairs):
   seq=torch.cat([payload['embeddings'][pos][0],payload['embeddings'][neg][0]]).to('cuda:0')
   pool=torch.cat([payload['embeddings'][pos][1],payload['embeddings'][neg][1]]).to('cuda:0')
   trials=[]
   for seed in [0,1]:
    gen=torch.Generator('cpu').manual_seed(seed)
    eps=torch.randn(x[:1].shape,generator=gen).to(x.device).expand_as(x)
    for t in [.25,.5,.75]:
     vals=[]
     for image in range(2):
      q=((1-t)*x[image:image+1]+t*eps[image:image+1]).expand(2,-1,-1,-1)
      truth=eps[image:image+1]-x[image:image+1]
      with torch.no_grad():pred=model(hidden_states=q,timestep=torch.full((2,),t*1000,device=x.device),
                  encoder_hidden_states=seq,pooled_projections=pool,return_dict=False)[0]
      errors=(pred.double()-truth.double()).square().mean((1,2,3)).cpu().tolist()
      vals.append(errors[1]-errors[0])
     trials.append(dict(seed=seed,time=t,raw_original=vals[0],raw_transformed=vals[1]))
   a=np.mean([r['raw_original'] for r in trials]);b=np.mean([r['raw_transformed'] for r in trials])
   rows.append(dict(image=name,paraphrase=j,text_positive=pos,text_negative=neg,raw_original=a,raw_transformed=b,
                    raw_pair_correct=bool(a>0 and b<0),antisymmetric_pair_correct=bool(a-b>0),
                    antisymmetric_original=float((a-b)/2),trials=trials))
   write(HERE/'outputs/rf_binding.json',dict(rows=rows,total=len(rows),
        raw_pair_correct=sum(r['raw_pair_correct'] for r in rows),antisymmetric_pair_correct=sum(r['antisymmetric_pair_correct'] for r in rows),
        score='negative shared-noise conditional flow matching MSE, CFG=1, 3 times x 2 noises',
        caveat='surrogate, not exact likelihood; no general semantic guarantee'))
   print(name,j,rows[-1]['raw_pair_correct'],rows[-1]['antisymmetric_pair_correct'],flush=True)
 print('RF_BINDING_COMPLETE',flush=True)
if __name__=='__main__':main()
