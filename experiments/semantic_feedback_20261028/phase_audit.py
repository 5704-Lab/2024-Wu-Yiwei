"""Raw decoded-image phase baseline on the new morphology-bearing tasks."""
import json
from pathlib import Path
import sys

import torch
from diffusers import AutoencoderKL

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
from analyze import threshold,summarize
sys.path.insert(0,str(ROOT/'experiments/phase_permission'))
from common import Phase,seed_setup,write


def main():
    seed_setup();device='cuda:0';vae=AutoencoderKL.from_pretrained(ROOT/'pretrained_models/sd3-medium/vae',
        torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to(device)
    out=HERE/'outputs/phase_audit';out.mkdir(exist_ok=True);result={}
    @torch.no_grad()
    def decode(y):return ((vae.decode(y.to(device)/vae.config.scaling_factor+vae.config.shift_factor,return_dict=False)[0]+1)/2).clamp(0,1)
    for tag in ['v1','v2']:
        manifest=json.loads((HERE/f'outputs/dataset_{tag}/manifest.json').read_text());rows=[]
        for scene in manifest['records']:
            folder=HERE/f'outputs/features_{tag}'/f"scene{scene['scene']:02d}"
            source=decode(torch.load(folder/'source.pt',weights_only=True)['y']);phase=Phase(source)
            z=phase.coefficients(source);zenergy=z.abs().square().sum()
            for name in scene['variants']:
                image=decode(torch.load(folder/f'{name}.pt',weights_only=True)['y'])
                with torch.no_grad():
                    energies=dict(phase=float((phase(image)-phase.source_obs).square().sum()),
                        complex=float((phase.coefficients(image)-z).abs().square().sum()/zenergy),
                        rgb=float((image-source).square().mean()))
                rows.append(dict(scene=scene['scene'],split=scene['split'],variant=name,energies=energies))
            print('PHASE',tag,scene['scene'],flush=True)
        cal=[r for r in rows if r['split']=='calibration'];held=[r for r in rows if r['split']=='heldout']
        result[tag]=dict(rows=rows,heldout={k:summarize(held,k,threshold(cal,k)) for k in ['phase','complex','rgb']})
    write(out/'results.json',result);print('PHASE_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':main()
