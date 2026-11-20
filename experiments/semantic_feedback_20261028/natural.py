"""Natural clean-state diagnostics, without invented legal morphology labels."""
import hashlib
import json
from pathlib import Path
import time

import torch

from extract import Extractor,HERE,ROOT
from common import seed_setup,write,save_image,panel


def main():
    seed_setup();run=Extractor();out=HERE/'outputs'/'natural_v1';out.mkdir(exist_ok=True)
    start=time.monotonic();rows=[]
    for pair in ['bear__00','brown_owl__00','bus__00','cat_crown__00']:
        data=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True)
        es,et=data['embeddings']
        run.emb=dict(conditions=dict(source=dict(seq=es[0],pool=es[2]),shape=dict(seq=et[0],pool=et[2]),null=dict(seq=es[1],pool=es[3])))
        run.text,_=run.sc.encode([data['record']['source_prompt'],data['record']['target_prompt']])
        x=data['x'].to(run.device)
        traj=torch.load(ROOT/'experiments/flowedit_frequency_trajectory/outputs'/pair/'trajectory.pt',weights_only=True)
        states=dict(source=x,step25=traj['states'][26].to(run.device),step33=traj['states'][34].to(run.device),
                    step41=traj['states'][42].to(run.device),baseline=traj['final'].to(run.device))
        if pair in ['bear__00','brown_owl__00']:
            for mode in ['CF','JOINT','NO_NULL']:
                p=ROOT/'experiments/null_joint_20261025/outputs/v1_full/full'/pair/'seed0'/mode/'trajectory.pt'
                states[mode]=torch.load(p,weights_only=True)['final'].to(run.device)
        gen=torch.Generator('cpu').manual_seed(19571028)
        eps=torch.randn(x.shape,generator=gen).to(run.device)
        source_image=run.sc.image(x).detach()
        for name,image in [('source_dark',source_image*.55),('source_warm',(source_image*torch.tensor([1.1,.9,1.05],device=run.device)[None,:,None,None]).clamp(0,1)),
                           ('source_mirror',source_image.flip(-1)),('source_shift',torch.roll(source_image,16,-1))]:
            with torch.no_grad():
                z=run.sc.vae.encode(image*2-1).latent_dist.mode()
                states[name]=(z-run.sc.vae.config.shift_factor)*run.sc.vae.config.scaling_factor
        cols=[];dest=out/pair;dest.mkdir(exist_ok=True)
        for name,y in states.items():
            path=dest/f'{name}.pt'
            if path.exists():continue
            with torch.no_grad():
                image=run.sc.image(y);sig=run.siglip(image);rf=run.probe(y,eps,conditions=['source','shape','null'])
            torch.save(dict(pair=pair,variant=name,y=y.cpu(),siglip=sig,sd3=rf),path)
            save_image(image,dest/f'{name}.png');cols.append((name,image))
            print('NATURAL',pair,name,round(time.monotonic()-start,2),flush=True)
        if cols:panel(cols,dest/'panel.png',title=pair,size=192)
        rows.append(dict(pair=pair,states=list(states)))
    write(out/'complete.json',dict(rows=rows,seconds=time.monotonic()-start,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Natural source photometric/geometry controls and retained real trajectories; no claim of known legal category target.'))
    print('NATURAL_COMPLETE',flush=True)


if __name__=='__main__':main()
