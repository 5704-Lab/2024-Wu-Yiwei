"""Fresh-source audit after the v1 failure diagnosis; no heldout retuning."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from analyze import phi,profile,threshold,summarize,LEGAL,ERRORS

HERE=Path(__file__).resolve().parent
CONFIGS=[('patch',11),('gram',11),('centered_gram',3)]


def z(record,kind,layer,task):
    p=record['sd3']['0.35'];names=p['names'];h=p['layers'][layer]
    return .5*(phi(h[names.index('source'):names.index('source')+1],kind)+phi(h[names.index(task):names.index(task)+1],kind))


def source_text(record,kind,layer,task):
    p=record['sd3']['0.35'];names=p['names'];h=p['layers'][layer]
    return phi(h[names.index(task):names.index(task)+1],kind)-phi(h[names.index('source'):names.index('source')+1],kind)


def evaluate_record(item,source,aux):
    out={};scope={}
    for kind,layer in CONFIGS:
        for task in ['shape','color','pose']:
            r=z(item,kind,layer,task)-z(source,kind,layer,task)
            stem=f'{kind}_l{layer}'
            auto=profile(r,[source_text(source,kind,layer,task)])[0]
            joint=[z(aux[f'{task}_{t}'],kind,layer,task)-z(source,kind,layer,task) for t in [.35,.65]]
            induced=profile(r,joint)[0]
            scope[f'{stem}_TEXT:{task}']=auto
            scope[f'{stem}_INDUCED:{task}']=induced
            if task=='shape':
                out[f'{stem}_RAW']=float(r@r);out[f'{stem}_TEXT']=auto
                out[f'{stem}_INDUCED']=induced
                split=[z(aux[f'{n}_{t}'],kind,layer,task)-z(source,kind,layer,task) for n in ['color','morph'] for t in [.35,.65]]
                out[f'{stem}_SPLIT']=profile(r,split)[0]
    out['sig_global']=float((item['siglip']['global_feature'].double()-source['siglip']['global_feature'].double()).square().sum())
    rr=phi(item['siglip']['layers'][3],'patch')-phi(source['siglip']['layers'][3],'patch')
    out['sig_l3_patch']=float(rr@rr)
    return dict(energies=out,scope=scope)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['dev','holdout'],required=True)
    args=ap.parse_args();torch.set_num_threads(4)
    out=HERE/'outputs'/'analysis_v2';out.mkdir(exist_ok=True)
    # Layer, observation and direction families were fixed before fresh holdout.
    proto=dict(configs=CONFIGS,seed_base=282031,calibration='sources 4-7 maximum of all five legal progress states',
        development=[0,1,2,3],heldout=list(range(8,16)),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        joint='Two internally generated target/source velocity effects, no reference image',
        split='Four color/morph effect columns; explicit decomposition only, not general text parsing')
    pp=out/'protocol.json'
    if pp.exists():assert json.loads(pp.read_text())==json.loads(json.dumps(proto))
    else:pp.write_text(json.dumps(proto,indent=2)+'\n')
    manifest=json.loads((HERE/'outputs/dataset_v2/manifest.json').read_text());rows=[]
    for entry in manifest['records']:
        if args.stage=='dev' and entry['split']=='heldout':continue
        if args.stage=='holdout' and entry['split']!='heldout':continue
        scene=entry['scene'];folder=HERE/'outputs/features_v2'/f'scene{scene:02d}'
        source=torch.load(folder/'source.pt',weights_only=True)
        aux={f'{n}_{t}':torch.load(HERE/'outputs/induced_v2'/f'scene{scene:02d}'/f'{n}_{t}.pt',weights_only=True)
             for n in ['shape','color','morph','pose'] for t in [.35,.65]}
        for name in entry['variants']:
            item=torch.load(folder/f'{name}.pt',weights_only=True)
            rows.append(dict(scene=scene,split=entry['split'],variant=name,**evaluate_record(item,source,aux)))
        print('INDUCED_METRICS',scene,flush=True)
    (out/f'metrics_{args.stage}.json').write_text(json.dumps(dict(rows=rows),indent=2)+'\n')
    if args.stage=='dev':
        cal=[r for r in rows if r['split']=='calibration']
        limits={k:threshold(cal,k) for k in rows[0]['energies']}
        p=out/'thresholds.json'
        if p.exists():assert json.loads(p.read_text())==limits
        else:p.write_text(json.dumps(limits,indent=2)+'\n')
        results={k:summarize(cal,k,v) for k,v in limits.items()}
        (out/'calibration.json').write_text(json.dumps(results,indent=2)+'\n')
    else:
        limits=json.loads((out/'thresholds.json').read_text())
        results={k:summarize(rows,k,v) for k,v in limits.items()}
        byscene={}
        for r in rows:byscene.setdefault(r['scene'],{})[r['variant']]=r
        scope={}
        for kind,layer in CONFIGS:
            for method in ['TEXT','INDUCED']:
                stem=f'{kind}_l{layer}_{method}'
                morph=[];pose=[]
                for source in byscene.values():
                    def v(state,task):return source[state]['scope'][stem+':'+task]
                    morph.append((v('valid','color')-v('valid','shape'))-(v('color_only','color')-v('color_only','shape')))
                    pose.append(v('pose_only','shape')-v('pose_only','pose'))
                scope[stem]=dict(morphology_expected_positive=sum(v>0 for v in morph),pose_expected_positive=sum(v>0 for v in pose),sources=len(morph))
        (out/'heldout.json').write_text(json.dumps(dict(metrics=results,task_selectivity=scope),indent=2)+'\n')
        print('V2_HELDOUT',[(k,v['legal_pass'],v['wrong_reject']) for k,v in results.items()],scope,flush=True)
    print('INDUCED_ANALYSIS_COMPLETE',flush=True)


if __name__=='__main__':main()
