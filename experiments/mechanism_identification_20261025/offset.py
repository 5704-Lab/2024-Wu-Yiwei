"""Registered additive-coverage repair; same-source finite observations."""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from binding import explained
from probe import ROOT,CleanScorer,Observers,read_image,seed_setup,save_image,panel,write,warp
from material_exact import fit_material_exact


def main():
    seed_setup();out=HERE/'outputs/v5'
    protocol=dict(scope='Additive mechanism coverage repair, not automatic text editing',
                  code_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
                  calibration=list(range(4,8)),test=list(range(40,48)),reference_tolerance=2e-4,
                  gain_bounds=[.05,12.],global_offset_bounds=[-.25,.75])
    if (out/'protocol.json').exists():raise RuntimeError('Use immutable tags')
    write(out/'protocol.json',protocol)
    for name in ['offset.py','material.py','material_exact.py','binding.py','probe.py','dataset.py']:
        (out/('snapshot_'+name)).write_bytes((HERE/name).read_bytes())
    sc=CleanScorer(torch.device('cuda:0'),torch.float32,'native');started=time.monotonic();rows=[];records=[]
    manifest=json.loads((out/'dataset/manifest.json').read_text())
    for record in manifest['records']:
        scene=record['scene']
        if scene not in protocol['calibration']+protocol['test']:continue
        images={name:read_image(HERE/value['file'],sc.device) for name,value in record['variants'].items()}
        source=images['source'];target=images['correct'];observer=Observers(sc,source)
        enc,_=sc.encode([record['target_prompt'],record['source_prompt']]);origin=None;sr=[]
        for name,image in images.items():
            fitted,fit=fit_material_exact(source,image,global_offset=True)
            with torch.no_grad():
                log=sc.logits_image(image,enc)[0];text=float(log[0]-log[1])
                if name=='source':origin=text
            row=dict(scene=scene,split='calibration' if scene<8 else 'test',case=name,text_gain=text-origin,
                     reference_mse=float((image-target).square().mean()),within_reference_tolerance=float((image-target).square().mean())<=2e-4,
                     kkt=fit['projected_gradient_inf'],**observer.measure(image),**explained(observer,image,fitted))
            rows.append(row);sr.append(row)
            if name in ['correct','turned','wrong_object','duplicate']:
                save_image(fitted,out/'scenes'/f'scene{scene:02d}'/f'{name}_mechanism.png')
        records.append(dict(scene=scene,rows=sr))
        write(out/'results.json',dict(protocol=protocol,records=records,rows=rows))
        print('OFFSET',scene,round(time.monotonic()-started,1),flush=True)
    cal=[r for r in rows if r['split']=='calibration' and r['case']=='correct'];test=[r for r in rows if r['split']=='test']
    metrics=['mechanism_rgb_mse','unexplained_phase','unexplained_complex','phase','patch_energy']
    thresholds={m:1.25*max(r[m] for r in cal)+1e-10 for m in metrics}
    floor=.5*float(np.median([r['text_gain'] for r in cal]))
    observations={m:{case:dict(accepted=sum(r[m]<=thresholds[m] for r in test if r['case']==case),
                                        total=sum(r['case']==case for r in test))
                     for case in ['correct','wrong_object','both_objects','turned','shifted','duplicate']} for m in metrics}
    selected=[]
    for rule in ['patch','mechanism_rgb','mechanism_rgb_phase','oracle']:
        choices=[]
        for scene in protocol['test']:
            candidates=[r for r in test if r['scene']==scene]
            def ok(r):
                if r['text_gain']<floor:return False
                if rule=='patch':return r['patch_energy']<=thresholds['patch_energy']
                return r['mechanism_rgb_mse']<=thresholds['mechanism_rgb_mse'] and (rule!='mechanism_rgb_phase' or r['unexplained_phase']<=thresholds['unexplained_phase'])
            valid=[r for r in candidates if ok(r)]
            if rule=='oracle':chosen=min(candidates,key=lambda r:r['reference_mse']);held=False
            elif valid:chosen=min(valid,key=lambda r:r['rgb_mse']);held=False
            else:chosen=next(r for r in candidates if r['case']=='source');held=True
            choices.append(dict(scene=scene,case=chosen['case'],success=chosen['within_reference_tolerance'],held=held))
        selected.append(dict(rule=rule,success=sum(r['success'] for r in choices),held=sum(r['held'] for r in choices),total=len(choices),choices=choices))
    write(out/'analysis.json',dict(thresholds=thresholds,text_floor=floor,observations=observations,selection=selected))
    natural=[]
    for pair in ['bear__00','brown_owl__00','bus__00','cat_crown__00']:
        data=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True)
        with torch.no_grad():source=sc.image(data['x'].to(sc.device))
        observer=Observers(sc,source)
        images=dict(source=source,global_dark=source*.45,global_white=source*.65+.35,
                    mirror=source.flip(-1),shift16=warp(source,tx=16))
        if pair in ['bear__00','brown_owl__00']:
            for mode in ['BASE','CF','JOINT']:
                z=torch.load(ROOT/'experiments/null_joint_20261025/outputs/v1_full/full'/pair/'seed0'/mode/'trajectory.pt',weights_only=True)['final'].to(sc.device)
                with torch.no_grad():images[mode]=sc.image(z)
        nr=[];columns=[('Source',source)]
        for name,image in images.items():
            fitted,fit=fit_material_exact(source,image,global_offset=True)
            nr.append(dict(pair=pair,case=name,kkt=fit['projected_gradient_inf'],**observer.measure(image),**explained(observer,image,fitted)))
            if name in ['global_white','BASE','JOINT']:columns.extend([(name,image),(name+' mechanism',fitted)])
        natural.append(dict(pair=pair,rows=nr));panel(columns,out/'natural'/f'{pair}.png',title=pair,size=192)
        write(out/'natural.json',dict(records=natural))
        print('NATURAL_OFFSET',pair,round(time.monotonic()-started,1),flush=True)
    print('OFFSET_COMPLETE',len(records),len(natural),round(time.monotonic()-started,1),flush=True)


if __name__=='__main__':
    with sdpa_kernel(SDPBackend.MATH):main()
