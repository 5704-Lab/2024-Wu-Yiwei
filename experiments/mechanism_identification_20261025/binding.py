"""Candidate observations, explained signals and restricted semantic factors."""
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from probe import ROOT,CleanScorer,Observers,read_image,seed_setup,panel,save_image,write,warp
from material_exact import fit_material_exact


def explained(observer,image,fitted):
    with torch.no_grad():
        z=observer.phase.coefficients(image);zf=observer.phase.coefficients(fitted)
        zs=observer.phase.coefficients(observer.source)
        scale=torch.sqrt(zs.abs().square()+observer.phase.epsilon.square())
        complex_error=((z-zf)/scale*observer.phase.sqrt_weight).abs().square().sum()
        phase=(observer.phase.embed(z)-observer.phase.embed(zf)).norm()
        geometry=.5*(observer.geometry(image)-observer.geometry(fitted)).square().mean()
    return dict(unexplained_complex=float(complex_error),unexplained_phase=float(phase),
                unexplained_geometry=float(geometry),mechanism_rgb_mse=float((image-fitted).square().mean()))


def main():
    seed_setup();out=HERE/'outputs/v4'
    protocol=dict(scope='Source/candidate-only mechanism observations; toy grammar fact readouts and oracle evaluation',
                  code_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
                  calibration_scenes=list(range(4,8)),test_scenes=list(range(32,40)),
                  source_dataset='v3',calibration_multiplier=1.25,completion_fraction=.5,
                  reference_tolerance=2e-4,extra_models=False,full_sd3_trajectories=False)
    if (out/'protocol.json').exists():raise RuntimeError('Use immutable tags')
    write(out/'protocol.json',protocol)
    for name in ['binding.py','material_exact.py','material.py','probe.py']:
        (out/('snapshot_'+name)).write_bytes((HERE/name).read_bytes())
    sc=CleanScorer(torch.device('cuda:0'),torch.float32,'native');started=time.monotonic()
    manifest=json.loads((HERE/'outputs/v3/dataset/manifest.json').read_text())
    all_rows=[];records=[]
    for record in manifest['records']:
        scene=record['scene']
        if scene not in protocol['calibration_scenes']+protocol['test_scenes']:continue
        images={name:read_image(HERE/v['file'],sc.device) for name,v in record['variants'].items()}
        exact_path=HERE/'outputs/v3/scenes'/f'scene{scene:02d}/fit_MATERIAL_EXACT.png'
        images['fit_MATERIAL_EXACT']=read_image(exact_path,sc.device)
        source=images['source'];reference=images['correct'];observer=Observers(sc,source)
        source_colors=re.search(r'A (\w+) bird on the left and a (\w+) bird on the right',record['source_prompt']).groups()
        target_color='black' if scene%4<2 else 'white'
        caption=lambda left,right:f'Two clay toy birds on a gray table. A {left} bird on the left and a {right} bird on the right.'
        texts=[caption(*source_colors),caption(target_color,source_colors[1]),caption(source_colors[0],target_color),caption(target_color,target_color)]
        encoding,_=sc.encode(texts);origin=None;rows=[]
        for name,image in images.items():
            fitted,fit=fit_material_exact(source,image)
            with torch.no_grad():
                s=sc.logits_image(image,encoding)[0]
                factors=np.array([float(s[1]-s[0]),float(.5*(s[2]-s[0]+s[3]-s[1])),float(s[3]-s[1])])
                if name=='source':origin=factors
                delta=factors-origin
            row=dict(scene=scene,split='calibration' if scene<8 else 'test',case=name,
                     completion_gain=float(delta[0]),right_main_drift=float(delta[1]),right_conditional_drift=float(delta[2]),
                     oracle_reference_mse=float((image-reference).square().mean()),
                     within_reference_tolerance=float((image-reference).square().mean())<=2e-4,
                     fit_kkt=fit['projected_gradient_inf'],**observer.measure(image),**explained(observer,image,fitted))
            rows.append(row);all_rows.append(row)
            if name in ['correct','turned','wrong_object','duplicate']:
                save_image(fitted,out/'scenes'/f'scene{scene:02d}'/f'{name}_mechanism.png')
        records.append(dict(scene=scene,global_budget=observer.global_budget,rows=rows))
        write(out/'results.json',dict(protocol=protocol,records=records,rows=all_rows,seconds=time.monotonic()-started))
        print('BINDING',scene,round(time.monotonic()-started,1),flush=True)
    cal=[r for r in all_rows if r['split']=='calibration' and r['case']=='correct']
    tests=[r for r in all_rows if r['split']=='test']
    metric_names=['mechanism_rgb_mse','unexplained_complex','unexplained_phase','unexplained_geometry','patch_energy','phase','geometry']
    thresholds={m:1.25*max(r[m] for r in cal)+1e-10 for m in metric_names}
    thresholds.update(right_main=1.25*max(abs(r['right_main_drift']) for r in cal)+1e-7,
                      right_conditional=1.25*max(abs(r['right_conditional_drift']) for r in cal)+1e-7,
                      completion=.5*float(np.median([r['completion_gain'] for r in cal])))
    observation={}
    for metric in metric_names:
        observation[metric]={}
        for case in ['correct','wrong_object','both_objects','turned','shifted','duplicate']:
            group=[r for r in tests if r['case']==case]
            observation[metric][case]=dict(accepted=sum(r[metric]<=thresholds[metric] for r in group),total=len(group))
    selection=[]
    for rule in ['raw_patch','mechanism_rgb','mechanism_rgb_phase','plus_right_main','plus_right_conditional','oracle']:
        selected=[]
        for scene in protocol['test_scenes']:
            candidates=[r for r in tests if r['scene']==scene]
            def feasible(r):
                if r['completion_gain']<thresholds['completion']:return False
                if rule=='raw_patch':return r['patch_energy']<=thresholds['patch_energy']
                if r['mechanism_rgb_mse']>thresholds['mechanism_rgb_mse']:return False
                if rule in ['mechanism_rgb_phase','plus_right_main','plus_right_conditional'] and r['unexplained_phase']>thresholds['unexplained_phase']:return False
                if rule=='plus_right_main' and abs(r['right_main_drift'])>thresholds['right_main']:return False
                if rule=='plus_right_conditional' and abs(r['right_conditional_drift'])>thresholds['right_conditional']:return False
                return True
            valid=[r for r in candidates if feasible(r)]
            if rule=='oracle':chosen=min(candidates,key=lambda r:r['oracle_reference_mse']);held=False
            elif valid:chosen=min(valid,key=lambda r:r['rgb_mse']);held=False
            else:chosen=next(r for r in candidates if r['case']=='source');held=True
            selected.append(dict(scene=scene,case=chosen['case'],within_reference_tolerance=chosen['within_reference_tolerance'],held=held))
        selection.append(dict(rule=rule,within_reference_tolerance=sum(r['within_reference_tolerance'] for r in selected),
                              held=sum(r['held'] for r in selected),total=len(selected),selected=selected))
    analysis=dict(thresholds=thresholds,observation=observation,selection=selection,
                  scope='Restricted renderer evidence; oracle-fitted candidate is not an automatic candidate generator')
    write(out/'analysis.json',analysis)
    natural=[]
    for pair in ['bear__00','brown_owl__00','bus__00','cat_crown__00']:
        data=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True)
        with torch.no_grad():source=sc.image(data['x'].to(sc.device))
        observer=Observers(sc,source)
        images=dict(source=source,global_dark=source*.45,global_white=source*.65+.35,mirror=source.flip(-1),shift16=warp(source,tx=16))
        if pair in ['bear__00','brown_owl__00']:
            for mode in ['BASE','CF','JOINT']:
                state=torch.load(ROOT/'experiments/null_joint_20261025/outputs/v1_full/full'/pair/'seed0'/mode/'trajectory.pt',weights_only=True)['final'].to(sc.device)
                with torch.no_grad():images[mode]=sc.image(state)
        columns=[('Source',source)];nr=[]
        for name,image in images.items():
            fitted,fit=fit_material_exact(source,image)
            nr.append(dict(pair=pair,case=name,fit_kkt=fit['projected_gradient_inf'],**observer.measure(image),**explained(observer,image,fitted)))
            if name in ['global_white','BASE','JOINT']:columns.extend([(name,image),(name+' mechanism',fitted)])
        natural.append(dict(pair=pair,rows=nr))
        panel(columns,out/'natural'/f'{pair}.png',title=pair,size=192)
        write(out/'natural.json',dict(records=natural,scope='Known photometry geometry controls and existing SD3 failures; no legal completed natural edit'))
        print('NATURAL_EXPLAINED',pair,round(time.monotonic()-started,1),flush=True)
    print('BINDING_COMPLETE',len(records),len(natural),round(time.monotonic()-started,1),flush=True)


if __name__=='__main__':
    with sdpa_kernel(SDPBackend.MATH):main()
