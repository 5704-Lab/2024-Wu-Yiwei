"""Source-anchored nuisance columns from actual model-induced clean updates.

Auxiliary images are generated internally from x and text, never from a target
reference. This is a hypothesis about allowed effects, not a semantic oracle.
"""
import argparse
import hashlib
import json
import time

import torch

from extract import Extractor,HERE,ROOT
from common import seed_setup,write,save_image,panel


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',default='dataset_v2')
    ap.add_argument('--features',default='features_v2');ap.add_argument('--tag',default='induced_v2')
    args=ap.parse_args();seed_setup();run=Extractor();out=HERE/'outputs'/args.tag;out.mkdir(exist_ok=True)
    manifest=json.loads((HERE/'outputs'/args.dataset/'manifest.json').read_text());start=time.monotonic()
    protocol=dict(script_sha256=hashlib.sha256(__file__.encode()).hexdigest(),step=.15,
        velocity_times=[.35,.65],directions=['shape','color','morph','pose'],
        observer_noise='same standardized source/current probes',auxiliary_noise='independent seed 5428+source seed',
        source_anchored=True,external_reference=False,training=False,
        split_scope='morph = shape minus color text response uses supplied explicit attribute decomposition; joint shape does not need that decomposition')
    protocol['script_sha256']=hashlib.sha256(open(__file__,'rb').read()).hexdigest()
    write(out/'protocol.json',protocol)
    for row in manifest['records']:
        scene=row['scene'];dest=out/f'scene{scene:02d}';dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists():continue
        source=torch.load(HERE/'outputs'/args.features/f'scene{scene:02d}/source.pt',weights_only=True)
        x=source['y'].to(run.device)
        eps=torch.randn(x.shape,generator=torch.Generator('cpu').manual_seed(19571028+scene)).to(run.device)
        auxnoise=torch.randn(x.shape,generator=torch.Generator('cpu').manual_seed(5428+row['seed'])).to(run.device)
        cols=[]
        for tau in [.35,.65]:
            names=['source','shape','color','pose']
            seq=torch.cat([run.emb['conditions'][n]['seq'] for n in names]).to(run.device)
            pool=torch.cat([run.emb['conditions'][n]['pool'] for n in names]).to(run.device)
            with torch.no_grad():
                q=((1-tau)*x+tau*auxnoise).expand(4,-1,-1,-1)
                v=run.model(hidden_states=q,timestep=torch.full((4,),tau*1000,device=run.device),
                    encoder_hidden_states=seq,pooled_projections=pool,return_dict=False)[0]
            directions=dict(shape=v[0:1]-v[1:2],color=v[0:1]-v[2:3],morph=v[2:3]-v[1:2],pose=v[0:1]-v[3:4])
            for name,d in directions.items():
                y=x+.15*d
                with torch.no_grad():
                    image=run.sc.image(y);sig=run.siglip(image);probe=run.probe(y,eps)
                torch.save(dict(scene=scene,direction=name,tau=tau,y=y.cpu(),siglip=sig,sd3=probe,
                    latent_delta_norm=float((y-x).norm())),dest/f'{name}_{tau}.pt')
                if scene==0:save_image(image,dest/f'{name}_{tau}.png');cols.append((f'{name} {tau}',image))
            print('INDUCED',scene,tau,round(time.monotonic()-start,2),flush=True)
        if cols:panel(cols,dest/'panel.png',title='Internally generated probes, not known legal targets',size=192)
        write(dest/'complete.json',dict(scene=scene,auxiliary_states=8))
    write(out/'complete.json',dict(sources=len(manifest['records']),states=8*len(manifest['records']),seconds=time.monotonic()-start))
    print('INDUCED_COMPLETE',flush=True)


if __name__=='__main__':main()
