"""Clean-state readouts with the original frozen VAE, SigLIP2 and SD3.

Uses standardized re-noising of y, NOT FlowEdit's actual coupled target query.
No training, segmentation, spatial update mask or factor labels enter extraction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from diffusers import SD3Transformer2DModel

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/phase_permission'))
from common import CleanScorer,write,seed_setup,save_image

LAYERS=[3,11,20]
SCHEDULE=[(.35,['source','shape','color','pose','null']),(.65,['source','shape','null'])]


def patch_pool(h,grid=8):
    side=int(round(h.shape[1]**.5));assert side*side==h.shape[1]
    h=h.transpose(1,2).reshape(h.shape[0],h.shape[-1],side,side)
    return F.adaptive_avg_pool2d(h,(grid,grid)).flatten(2).transpose(1,2)


class Extractor:
    def __init__(self):
        self.device=torch.device('cuda:0');self.sc=CleanScorer(self.device,torch.float32,'native')
        self.model=SD3Transformer2DModel.from_pretrained(ROOT/'pretrained_models/sd3-medium/transformer',
            torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to(self.device)
        self.emb=torch.load(HERE/'outputs/conditions_v1.pt',weights_only=True)
        self.text,_=self.sc.encode(list(self.emb['prompts'].values()))
        self.capture={};self.handles=[]
        for layer in LAYERS:
            def hook(module,inputs,output,l=layer):
                self.capture[l]=patch_pool(output[1]).detach().cpu()
            self.handles.append(self.model.transformer_blocks[layer].register_forward_hook(hook))

    @torch.no_grad()
    def siglip(self,image):
        vision=self.sc.model.vision_model(pixel_values=self.sc.preprocess(image),output_hidden_states=True)
        states={l:patch_pool(vision.hidden_states[l+1]).cpu() for l in [3,7,11]}
        f=vision.pooler_output;f=F.normalize(f,dim=-1)
        return dict(layers=states,global_feature=f.cpu(),logits=(self.sc.alpha*(f@self.text.T)+self.sc.bias).cpu())

    @torch.no_grad()
    def probe(self,y,eps,conditions=None):
        results={}
        for tau,names in SCHEDULE:
            if conditions is not None:names=[n for n in names if n in conditions]
            seq=torch.cat([self.emb['conditions'][n]['seq'] for n in names]).to(self.device)
            pool=torch.cat([self.emb['conditions'][n]['pool'] for n in names]).to(self.device)
            q=((1-tau)*y+tau*eps).expand(len(names),-1,-1,-1)
            self.capture={}
            pred=self.model(hidden_states=q,timestep=torch.full((len(names),),tau*1000,device=self.device),
                encoder_hidden_states=seq,pooled_projections=pool,return_dict=False)[0]
            error=(pred.double()-(eps-y).double()).square().mean((1,2,3)).cpu()
            results[str(tau)]=dict(names=names,layers=self.capture.copy(),denoising_error=error)
        return results


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scenes',nargs='+',type=int)
    ap.add_argument('--tag',default='features_v1');ap.add_argument('--dataset',default='dataset_v1')
    args=ap.parse_args();seed_setup()
    manifest=json.loads((HERE/'outputs'/args.dataset/'manifest.json').read_text())
    out=HERE/'outputs'/args.tag;out.mkdir(parents=True,exist_ok=True)
    protocol=dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dataset_sha256=hashlib.sha256((HERE/'outputs'/args.dataset/'manifest.json').read_bytes()).hexdigest(),
        schedule=SCHEDULE,layers=LAYERS,precision='float32',clean_state='D(E(image).mode), original scaled latent',
        fixed_noise_seed_base=19571028,factor_labels_used=False,trainable_parameters=0,grid=8)
    p=out/'protocol.json'
    if p.exists():assert json.loads(p.read_text())==json.loads(json.dumps(protocol)),'Use fresh tag for protocol changes'
    else:write(p,protocol)
    run=Extractor();start=time.monotonic();samples=0;queries=0
    for row in manifest['records']:
        scene=row['scene']
        if args.scenes is not None and scene not in args.scenes:continue
        dest=out/f'scene{scene:02d}';dest.mkdir(exist_ok=True)
        gen=torch.Generator('cpu').manual_seed(19571028+scene)
        eps=torch.randn((1,16,64,64),generator=gen).to(run.device)
        for name,item in row['variants'].items():
            target=dest/f'{name}.pt'
            if target.exists():continue
            im=Image.open(HERE/item['file']).convert('RGB')
            image=torch.from_numpy(np.array(im).copy()).permute(2,0,1)[None].float().to(run.device)/255
            with torch.no_grad():
                raw=run.sc.vae.encode(image*2-1).latent_dist.mode()
                y=(raw-run.sc.vae.config.shift_factor)*run.sc.vae.config.scaling_factor
                decoded=run.sc.image(y)
                sig=run.siglip(decoded);rf=run.probe(y,eps)
            record=dict(scene=scene,variant=name,y=y.cpu(),siglip=sig,sd3=rf,
                vae_rmse=float((decoded-image).square().mean().sqrt()))
            torch.save(record,target)
            if scene==0:save_image(decoded,dest/f'{name}.png')
            samples+=1;queries+=sum(len(names) for _,names in SCHEDULE)
            print('EXTRACT',scene,name,'samples',samples,'queries',queries,'seconds',round(time.monotonic()-start,2),flush=True)
        write(dest/'complete.json',dict(scene=scene,variants=list(row['variants']),
            processed_samples_this_run=samples,queries_this_run=queries,seconds=time.monotonic()-start))
    write(out/('timing_'+str(int(time.time()))+'.json'),dict(samples=samples,queries=queries,seconds=time.monotonic()-start,
        peak_memory_bytes=torch.cuda.max_memory_allocated()))
    print('EXTRACTION_COMPLETE',samples,queries,time.monotonic()-start,flush=True)


if __name__=='__main__':main()
