"""Area/viewpoint confound test, fixed v2 thresholds, all v4 sources held out."""
import json
from pathlib import Path
import sys

import torch
from diffusers import AutoencoderKL

from analyze import phi,profile,summarize
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/phase_permission'))
from common import Phase,seed_setup,write


def main():
    seed_setup();vae=AutoencoderKL.from_pretrained(ROOT/'pretrained_models/sd3-medium/vae',
        torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).to('cuda:0')
    @torch.no_grad()
    def decode(y):return ((vae.decode(y.cuda()/vae.config.scaling_factor+vae.config.shift_factor,return_dict=False)[0]+1)/2).clamp(0,1)
    manifest=json.loads((HERE/'outputs/dataset_v4/manifest.json').read_text());rows=[]
    for entry in manifest['records']:
        folder=HERE/'outputs/features_v4'/f"scene{entry['scene']:02d}"
        source=torch.load(folder/'source.pt',weights_only=True)
        source_image=decode(source['y']);phase=Phase(source_image)
        zs=phase.coefficients(source_image);zsenergy=zs.abs().square().sum()
        for name in entry['variants']:
            current=torch.load(folder/f'{name}.pt',weights_only=True);energies={}
            for kind,layer in [('patch',11),('gram',11),('centered_gram',3)]:
                def state(record,cond):
                    p=record['sd3']['0.35'];i=p['names'].index(cond)
                    return phi(p['layers'][layer][i:i+1],kind)
                r=.5*(state(current,'source')+state(current,'shape')-state(source,'source')-state(source,'shape'))
                a=state(source,'shape')-state(source,'source')
                energies[f'{kind}_l{layer}_RAW']=float(r@r)
                energies[f'{kind}_l{layer}_TEXT']=profile(r,[a])[0]
            diff=phi(current['siglip']['layers'][3],'patch')-phi(source['siglip']['layers'][3],'patch')
            energies['sig_l3_patch']=float(diff@diff)
            energies['sig_global']=float((current['siglip']['global_feature'].double()-source['siglip']['global_feature'].double()).square().sum())
            image=decode(current['y'])
            with torch.no_grad():
                energies.update(phase=float((phase(image)-phase.source_obs).square().sum()),
                    complex=float((phase.coefficients(image)-zs).abs().square().sum()/zsenergy),rgb=float((image-source_image).square().mean()))
            logs=current['siglip']['logits'][0]
            rows.append(dict(scene=entry['scene'],variant=name,camera_side=entry['camera_side'],energies=energies,task_margin=float(logs[1]-logs[0])))
        print('BALANCED',entry['scene'],entry['camera_side'],flush=True)
    limits=json.loads((HERE/'outputs/analysis_v2/thresholds.json').read_text())
    phases=json.loads((HERE/'outputs/phase_audit/results.json').read_text())['v2']['heldout']
    for k in ['phase','complex','rgb']:limits[k]=phases[k]['threshold']
    limits={k:v for k,v in limits.items() if k in rows[0]['energies']}
    summary={k:summarize(rows,k,v) for k,v in limits.items()}
    by_camera={str(side):{k:summarize([r for r in rows if r['camera_side']==side],k,v) for k,v in limits.items()} for side in [-1.,1.]}
    byscene={}
    for r in rows:byscene.setdefault(r['scene'],{})[r['variant']]=r
    completion=dict(valid_above_wrong_object=sum(s['valid']['task_margin']>s['wrong_object']['task_margin'] for s in byscene.values()),
        valid_above_color_only=sum(s['valid']['task_margin']>s['color_only']['task_margin'] for s in byscene.values()),sources=len(byscene))
    write(HERE/'outputs/balanced_analysis.json',dict(rows=rows,summary=summary,by_camera=by_camera,completion=completion,
        scope='Eight fresh sources, fixed v2 calibration; camera side is offline stratification only, never an observer input.'))
    print('BALANCED_COMPLETE',[(k,v['legal_pass'],v['wrong_reject']) for k,v in summary.items()],completion,flush=True)


if __name__=='__main__':main()
