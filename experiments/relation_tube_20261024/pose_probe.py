"""Does existing SD3 velocity evidence observe the annotated source heading?

Same caption-pair readout on source and edited images, with mirror controls.
The null caption removes the pose clause; six shared-noise probes are used.
This is an empirical compatibility diagnostic, not calibrated pose truth.
"""
import sys,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKL,SD3Transformer2DModel
from torch.nn.attention import sdpa_kernel,SDPBackend
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'phase_permission'));from common import ROOT,seed_setup,write

def main():
 seed_setup();device=torch.device('cuda:0');out=HERE/'outputs/pose_probe'
 payload=torch.load(HERE/'outputs/conditions/encodings.pt',weights_only=True)
 protocol=json.loads((HERE/'outputs/conditions/protocol.json').read_text())
 vae=AutoencoderKL.from_pretrained(ROOT/'pretrained_models/sd3-medium/vae',torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to(device)
 latents={}
 for pair in ['bear__00','brown_owl__00']:
  paths={'SOURCE':ROOT/'experiments/joint_permission/outputs/analysis'/f'{pair}_source.png',
   'BASE':HERE/'outputs/conditions'/pair/'seed0/BASE/final.png',
   'ATTRIBUTE_POSE':HERE/'outputs/conditions'/pair/'seed0/ATTRIBUTE_POSE/final.png',
   'old_exposure':ROOT/'experiments/joint_permission/outputs/achromatic_strong'/pair/'final.png'}
  if pair=='bear__00':paths['TUBE']=HERE/'outputs/v3/full'/pair/'TUBE/final.png'
  for case,path in paths.items():
   im=torch.tensor(np.array(Image.open(path).convert('RGB')),device=device,dtype=torch.float32).permute(2,0,1)[None]/255
   with torch.no_grad():z=vae.encode(torch.cat([im,im.flip(-1)])*2-1).latent_dist.mode();z=(z-vae.config.shift_factor)*vae.config.scaling_factor
   latents[(pair,case)]=z.cpu()
 del vae;torch.cuda.empty_cache()
 model=SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to(device)
 rows=[]
 for (pair,case),x_cpu in latents.items():
  x=x_cpu.to(device);caps=protocol['captions'][pair]
  baseline=payload['embeddings'][caps['BASE'][1]];anchored=payload['embeddings'][caps['POSE'][1]]
  seq=torch.cat([baseline[0],anchored[0]]).to(device);pool=torch.cat([baseline[1],anchored[1]]).to(device)
  trials=[]
  for seed in [0,1]:
   gen=torch.Generator('cpu').manual_seed(seed);eps=torch.randn(x[:1].shape,generator=gen).to(device)
   for t in [.25,.5,.75]:
    vals=[]
    for i in range(2):
     q=((1-t)*x[i:i+1]+t*eps).expand(2,-1,-1,-1);truth=eps-x[i:i+1]
     with torch.no_grad():pred=model(hidden_states=q,timestep=torch.full((2,),1000*t,device=device),encoder_hidden_states=seq,pooled_projections=pool,return_dict=False)[0]
     errors=(pred.double()-truth.double()).square().mean((1,2,3))
     vals.append(float(errors[0]-errors[1]))
    trials.append(dict(seed=seed,time=t,original=vals[0],mirror=vals[1],antisymmetric=.5*(vals[0]-vals[1])))
  rows.append(dict(pair=pair,case=case,antisymmetric=float(np.mean([r['antisymmetric'] for r in trials])),
    positive_trials=sum(r['antisymmetric']>0 for r in trials),trials=trials))
  write(out/'results.json',dict(rows=rows,scope='target-caption pose-clause contrast, shared-noise RF compatibility; manual source facts and selected development outputs; no calibrated semantic guarantee'))
  print(pair,case,rows[-1]['antisymmetric'],rows[-1]['positive_trials'],flush=True)
 print('POSE_PROBE_COMPLETE',flush=True)

if __name__=='__main__':
 with sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]):main()
