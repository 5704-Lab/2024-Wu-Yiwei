"""Audit completed trajectories and preserve unranked contact sheets."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/phase_permission'))
from common import panel, write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tags',nargs='+',default=['v1_full'])
    args=ap.parse_args();out=HERE/'outputs/analysis';rows=[];audits=[]
    sheets={}
    for tag in args.tags:
        for path in sorted((HERE/'outputs'/tag/'full').glob('*/*/*/complete.json')):
            result=json.loads(path.read_text());trajectory=torch.load(path.parent/'trajectory.pt',weights_only=True)
            states=trajectory['states'];pair=result['pair'];seed=result['seed'];mode=result['mode']
            x=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True)['x']
            a=dict(tag=tag,pair=pair,seed=seed,mode=mode,updates=len(result['rows']),states=len(states),
                   finite=bool(torch.isfinite(states).all()),initial_error=float((states[0]-x).abs().max()),
                   endpoint_error=float((states[-1]-trajectory['final']).abs().max()))
            a['passed']=bool(a['updates']==33 and a['states']==34 and a['finite'] and
                a['initial_error']==0 and a['endpoint_error']==0 and [r['step'] for r in result['rows']]==list(range(17,50)))
            if mode=='BASE' and seed==0:
                base=torch.load(ROOT/'experiments/flowedit_frequency_trajectory/outputs'/pair/'trajectory.pt',weights_only=True)
                a['baseline_all_state_error']=float((states-base['states'][17:]).abs().max())
                a['passed'] &= a['baseline_all_state_error']<1e-6
            constrained=[r for r in result['rows'] if 'progress_floor' in r]
            a['constrained_steps']=len(constrained)
            if constrained:
                a['halfspace_failures']=sum(r['final_progress']<r['progress_floor']-1e-4 for r in constrained)
                a['accepted_geometry_corrections']=sum(r['accepted'] for r in constrained)
                a['finite_geometry_failures']=sum(r['accepted'] and r['geometry_after']>=r['geometry_raw'] for r in constrained)
                if mode in ['PATCH_TUBE','PATCH_ADAPT']:
                    a['tube_violations']=sum(r['tube_actual']>r['tube_budget']+1e-7 for r in constrained)
                    a['scaled_steps']=sum(r['tube_scale']<1 for r in constrained)
                    a['holds']=sum(r['tube_scale']==0 for r in constrained)
                    a['passed'] &= a['tube_violations']==0
                else:a['passed'] &= a['halfspace_failures']==0 and a['finite_geometry_failures']==0
            assert a['passed'],a
            audits.append(a)
            rows.append(dict(tag=tag,pair=pair,seed=seed,mode=mode,seconds=result['seconds'],**result['endpoint']))
            sheets.setdefault((pair,seed),[]).append((tag+'/'+mode,path.parent/'final.png'))
    for (pair,seed),images in sheets.items():
        row_images=[]
        for k in range(0,len(images),4):
            dest=out/f'{pair}_seed{seed}_row{k//4}.png'
            panel(images[k:k+4],dest,title=f'{pair}, seed {seed}',size=320)
            row_images.append(Image.open(dest).copy())
        canvas=Image.new('RGB',(1280,385*len(row_images)),'white')
        for i,im in enumerate(row_images):canvas.paste(im,(0,385*i))
        canvas.save(out/f'{pair}_seed{seed}_all.png')
        source=ROOT/'experiments/joint_permission/outputs/analysis'/f'{pair}_source.png'
        if source.exists() and seed==0:
            selected=[('Source (VAE)',source)]
            targets=[('v1_full/BASE','FlowEdit'),('v1_full/JOINT','Geometry + text response'),
                     ('v5_adaptive_bear/PATCH_ADAPT','Adaptive source tube'),
                     ('v5_adaptive_owl/PATCH_ADAPT','Adaptive source tube')]
            mapping=dict(images)
            for key,label in targets:
                if key in mapping:selected.append((label,mapping[key]))
            panel(selected,out/f'{pair}_summary.png',title=pair,size=384)
    write(out/'endpoints.json',dict(rows=rows,scope='development-case diagnostics; proxies do not establish semantic success'))
    if rows:
        fields=list(dict.fromkeys(k for r in rows for k in r))
        with (out/'endpoints.csv').open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    write(out/'audit.json',dict(passed=all(a['passed'] for a in audits),completed_trajectories=len(audits),
         active_updates=sum(a['updates'] for a in audits),trajectories=audits,
         current_code_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}))
    print(json.dumps(dict(trajectories=len(audits),updates=sum(a['updates'] for a in audits),passed=all(a['passed'] for a in audits))))


if __name__=='__main__':main()
