"""Known transforms of real source reconstructions + actual clean FE snapshots.

Transform truth is stored for offline measurement evaluation only. Affine stretch
is a mechanism control, not an assertion that globally stretching a scene is a
valid semantic task. No truth/mask is passed to descriptor or matching routines.
"""
import math
import time
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from common import *


def main():
    setup();out=HERE/'outputs/data_v1'
    if (out/'manifest.json').exists():raise RuntimeError('Immutable dataset exists')
    vae=AutoencoderKL.from_pretrained(ROOT/'pretrained_models/sd3-medium/vae',local_files_only=True,
        torch_dtype=torch.float32).eval().requires_grad_(False).to('cuda:0')
    u=torch.diag(torch.tensor([1.18,.88]))
    def rotation(deg):
        a=math.radians(deg);return torch.tensor([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
    affines={'roundtrip':(torch.eye(2),torch.zeros(2)), 'stretch':(u,torch.zeros(2)),
        'rotate_plus12':(rotation(12),torch.zeros(2)), 'rotate_minus12':(rotation(-12),torch.zeros(2)),
        'rotate_plus25':(rotation(25),torch.zeros(2)), 'translate':(torch.eye(2),torch.tensor([.08,-.04])),
        'mirror':(torch.diag(torch.tensor([-1.,1.])),torch.zeros(2)),
        'stretch_rotate':(rotation(12)@u,torch.zeros(2))}
    records=[];start=time.monotonic()
    for pair in PAIRS:
        inp=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True)
        x=inp['x'].to('cuda:0');dest=out/pair;dest.mkdir(parents=True,exist_ok=True)
        with torch.no_grad():im=image(vae,x)
        save_image(im,dest/'source.png');torch.save(dict(y=x.cpu()),dest/'source.pt')
        variants={'source':dict(kind='identity',matrix=torch.eye(2).tolist(),translation=[0.,0.])}
        with torch.no_grad():
            for name,(a,t) in affines.items():
                inv=torch.linalg.inv(a).to(im)
                theta=torch.cat([inv,(-2*inv@t.to(im))[:,None]],1)[None]
                grid=F.affine_grid(theta,im.shape,align_corners=False)
                warped=F.grid_sample(im,grid,mode='bilinear',padding_mode='reflection',align_corners=False)
                z=vae.encode(warped*2-1).latent_dist.mode()
                y=(z-vae.config.shift_factor)*vae.config.scaling_factor
                torch.save(dict(y=y.cpu()),dest/f'{name}.pt');save_image(image(vae,y),dest/f'{name}.png')
                variants[name]=dict(kind='affine',matrix=a.tolist(),translation=t.tolist(),padding='reflection; only in-frame truth evaluated')
            lum=(im*torch.tensor([.299,.587,.114],device=im.device)[None,:,None,None]).sum(1,keepdim=True)
            photos={'dark':im*.6,'warm':(im*torch.tensor([1.15,.85,1.05],device=im.device)[None,:,None,None]).clamp(0,1),
                    'desaturate':.25*im+.75*lum}
            for name,warped in photos.items():
                z=vae.encode(warped*2-1).latent_dist.mode();y=(z-vae.config.shift_factor)*vae.config.scaling_factor
                torch.save(dict(y=y.cpu()),dest/f'{name}.pt');save_image(image(vae,y),dest/f'{name}.png')
                variants[name]=dict(kind='photometric',matrix=torch.eye(2).tolist(),translation=[0.,0.])
        saved=torch.load(ROOT/'experiments/stage1_full/outputs'/f'{pair}__seed0/states.pt',weights_only=True)
        for step in [25,33,41,49]:
            s=next((s for s in saved['snapshots'] if s['step']==step),None)
            if s is None:continue
            y=s['zE'];name=f'flow_step{step}'
            torch.save(dict(y=y,raw_increment=s['delta']),dest/f'{name}.pt')
            with torch.no_grad():save_image(image(vae,y.to('cuda:0')),dest/f'{name}.png')
            variants[name]=dict(kind='actual_flowedit_clean_state',step=step,known_geometric_truth=False,
                origin='stage1_full zE = z + delta; original n_min=0 coupling coordinate, not predicted denoising endpoint')
        records.append(dict(pair=pair,split=split(pair),record=inp['record'],source_latent_hash=digest(x),variants=variants))
        print('PREPARE',pair,len(variants),round(time.monotonic()-start,2),flush=True)
    # Wrong content at the same coordinates: observer must not mistake positional
    # confidence for real content correspondence. No affine truth is fabricated.
    for i,row in enumerate(records):
        other=records[(i+1)%len(records)]['pair']
        d=torch.load(out/other/'source.pt',weights_only=True);torch.save(d,out/row['pair']/'mismatch.pt')
        row['variants']['mismatch']=dict(kind='content_mismatch',other_source=other,known_geometric_truth=False)
    write(out/'manifest.json',dict(date='2026-11-04',records=records,script_sha256=sha(__file__),
        no_transform_truth_enters_matching=True,source_count=len(records),seconds=time.monotonic()-start))


if __name__=='__main__':main()
