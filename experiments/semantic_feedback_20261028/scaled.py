"""Step-norm-schedule controls for the real pilot; no semantic feedback."""
import json
import time

import torch
from torch.nn.attention import sdpa_kernel,SDPBackend

from extract import Extractor,HERE,ROOT
from common import seed_setup,write,save_image,panel


def main():
    seed_setup();out=HERE/'outputs/full_bear_v2';run=Extractor()
    for h in run.handles:h.remove()
    data=torch.load(ROOT/'experiments/stage1_full/inputs/bear__00.pt',weights_only=True)
    es,et=data['embeddings'];x=data['x'].to(run.device)
    seq=torch.cat([es[1],es[0],et[1],et[0]]).to(run.device);pool=torch.cat([es[3],es[2],et[3],et[2]]).to(run.device)
    e,_=run.sc.encode([data['record']['source_prompt'],data['record']['target_prompt']])
    times=torch.tensor(json.loads((ROOT/'experiments/stage1_full/inputs/manifest.json').read_text())['timesteps'],device=run.device)
    start=time.monotonic()
    for original in ['TEXT','INDUCED']:
        info=json.loads((out/original/'complete.json').read_text())
        ss=torch.load(out/original/'trajectory.pt',weights_only=True)['states']
        scales=[float((ss[k+1]-ss[k]).norm())/r['raw_norm'] for k,r in enumerate(info['rows'])]
        mode='SCALE_'+original;dest=out/mode;dest.mkdir(exist_ok=True)
        if (dest/'complete.json').exists():continue
        y=x.clone();generator=torch.Generator('cpu').manual_seed(0)
        noises=[torch.randn(x.shape,generator=generator).to(run.device) for _ in range(33)]
        states=[y.cpu()]
        for j,k in enumerate(range(17,50)):
            t=times[k]/1000;tn=times[k+1]/1000 if k<49 else torch.zeros_like(t)
            qs=(1-t)*x+t*noises[j];qy=qs+y-x
            with torch.no_grad(),sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]):
                v=run.model(hidden_states=torch.cat([qs,qs,qy,qy]),timestep=times[k].expand(4),encoder_hidden_states=seq,pooled_projections=pool,return_dict=False)[0]
                a,b,c,d=v.chunk(4);delta=(t-tn)*((a+3.5*(b-a))-(c+13.5*(d-c)))
                y=y+scales[j]*delta
            states.append(y.cpu())
        with torch.no_grad():
            im=run.sc.image(y);l=run.sc.logits_image(im,e)[0]
        save_image(im,dest/'final.png');torch.save(dict(states=torch.stack(states),final=y.cpu()),dest/'trajectory.pt')
        write(dest/'complete.json',dict(mode=mode,step_scales=scales,endpoint=dict(task_margin=float(l[1]-l[0]),target_logit=float(l[1]),source_logit=float(l[0])),
            scope='Same scalar norm-ratio schedule as the corresponding corrected trajectory; evolves its own states, so not exact norm matching at different states.',seconds=time.monotonic()-start))
        print('SCALED_COMPLETE',mode,flush=True)
    with torch.no_grad():source=run.sc.image(x)
    panel([('Source',source)]+[(m,out/m/'final.png') for m in ['BASE','TEXT','SCALE_TEXT','INDUCED','SCALE_INDUCED']],
        out/'comparison_with_scale.png',title='Original, clean feedback, and scalar step-schedule controls',size=300)


if __name__=='__main__':main()
