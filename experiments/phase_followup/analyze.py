"""Canonical observations for all variants, plus actual task readouts and images."""
import sys,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
HERE=Path(__file__).parent
sys.path.insert(0,str(HERE.parent/'phase_permission'))
from common import Phase,Appearance,panel,write,seed_setup
from analyze import gradient_tensor_distance
from dense_probe import measure
import cv2


def load(p):return np.asarray(Image.open(p).convert('RGB'),dtype=np.float32)/255
def tensor(a):return torch.tensor(a,dtype=torch.float64).permute(2,0,1)[None]


def main():
    seed_setup();rows=[];panels=[]
    old=HERE.parent/'phase_permission/outputs'
    for pair in ['bear__00','brown_owl__00']:
        srcpath=old/'real'/pair/'source_vae.png';source=load(srcpath)
        phase=Phase(tensor(source));app=Appearance(phase.source_small,2)
        cases=[('BASE',old/'full'/pair/'BASE'),('OLD_PERMIT',old/'full'/pair/'PERMIT')]
        cases += [(v,HERE/'outputs'/v/'full'/pair/'PERMIT') for v in ['RADIUS','LOW_FREQ','LOW_NO_TASK']]
        columns=[('Source VAE',srcpath)]
        for name,dest in cases:
            if not (dest/'metrics.json').exists():continue
            m=json.loads((dest/'metrics.json').read_text());image=load(dest/'final.png')
            flow,cycle=measure(cv2.resize(source,(128,128)),cv2.resize(image,(128,128)),True)
            magnitude=np.linalg.norm(flow,axis=-1)
            obs=phase(tensor(image)).detach();theta,ref,fit=app.fit(phase,obs,maxiter=100)
            row=dict(pair=pair,variant=name,canonical_phase=float((obs-ref).norm()),fit=fit,
                     full_target=m['metrics']['full_target'],short_margin=m['metrics']['short_margin'],
                     edge_distance=gradient_tensor_distance(source,image),
                     flow_p95_at512=float(np.quantile(magnitude,.95)*4),
                     flow_cycle_p90_at512=float(np.quantile(cycle,.9)*4),
                     accepted=sum(s['accepted'] for s in m['steps']),
                     mean_correction_ratio=float(np.mean([s['correction_norm']/s.get('base_increment_norm',1) for s in m['steps']])),
                     mean_increment_ratio=float(np.mean([s['increment_ratio'] for s in m['steps']])))
            rows.append(row);columns.append((name,dest/'final.png'))
            np.savez_compressed(HERE/'outputs'/f'{pair}_{name}_dense.npz',flow=flow,cycle=cycle)
        path=HERE/'outputs'/f'{pair}_comparison.png';panel(columns,path,title=pair+' | canonical comparison',size=360)
        panels.append(Image.open(path))
    out=Image.new('RGB',(max(p.width for p in panels),sum(p.height for p in panels)),'white');y=0
    for p in panels:out.paste(p,(0,y));y+=p.height
    out.save(HERE/'outputs/all_comparison.png')
    write(HERE/'outputs/evaluation.json',dict(rows=rows,canonical='source-fixed original Phase, FP64 PNG, 100-iteration fit',
        limitation='phase and edge are proxies; task logits also not ground truth'))
    for r in rows:print({k:v for k,v in r.items() if k!='fit'},flush=True)


if __name__=='__main__':main()
