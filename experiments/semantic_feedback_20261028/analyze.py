"""Predeclared observer audit, selection on development and calibration only.

All factor labels are consumed here for offline evaluation; extraction and the
automatic feature observers do not consume factors or reference target images.
Oracle profiling is a separately named information/action diagnostic.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
LEGAL=['source','color_only','partial','valid','valid_alt']
ERRORS=['wrong_heading','wrong_head','wrong_background','wrong_object','both_objects','shifted']


def phi(h,kind):
    h=h.squeeze(0).double()
    if kind=='centered_gram':h=h-h.mean(0,keepdim=True)
    h=F.normalize(h,dim=-1)
    if kind.endswith('gram'):return (h@h.T).flatten()/h.shape[0]
    return h.flatten()/h.shape[0]**.5


def profile(r,columns):
    if not columns:return float(torch.dot(r,r)),0,[]
    a=torch.stack(columns,-1)
    gram=a.T@a;rhs=a.T@r
    eig,v=torch.linalg.eigh(gram)
    cutoff=max(float(eig.max())*1e-8,1e-14)
    keep=eig>cutoff
    coeff=v[:,keep]@((v[:,keep].T@rhs)/eig[keep]) if keep.any() else torch.zeros_like(rhs)
    residual=r-a@coeff
    return float(residual@residual),int(keep.sum()),coeff.tolist()


def values(current,source,oracle_color=None,oracle_valid=None):
    energies={};scope={}
    for layer,h in current['siglip']['layers'].items():
        for kind in ['patch','gram','centered_gram']:
            z=phi(h,kind);zx=phi(source['siglip']['layers'][layer],kind);r=z-zx
            name=f'sig_l{layer}_{kind}'
            energies[name]=float(r@r)
    energies['sig_global']=float((current['siglip']['global_feature'].double()-source['siglip']['global_feature'].double()).square().sum())
    for tau,probe in current['sd3'].items():
        src=source['sd3'][tau];names=probe['names'];si=names.index('source')
        for layer,h in probe['layers'].items():
            hx=src['layers'][layer]
            for kind in ['patch','gram','centered_gram']:
                zs=phi(h[si:si+1],kind);xs=phi(hx[si:si+1],kind)
                prefix=f'sd3_t{tau}_l{layer}_{kind}'
                r=zs-xs;energies[prefix+'_source']=float(r@r)
                ni=names.index('null');rn=phi(h[ni:ni+1],kind)-phi(hx[ni:ni+1],kind)
                energies[prefix+'_null']=float(rn@rn)
                for task in ['shape','color','pose']:
                    if task not in names:continue
                    ai=names.index(task)
                    za=phi(h[ai:ai+1],kind);xa=phi(hx[ai:ai+1],kind)
                    a0=xa-xs;a1=za-zs
                    joint=.5*(za+zs-xa-xs)
                    v0,rank0,_=profile(joint,[a0]);v1,rank1,_=profile(joint,[a0,a1])
                    # Task-conditioned automatic models, with no factor labels.
                    scope[f'{prefix}_profile_source:{task}']=v0
                    scope[f'{prefix}_profile_dynamic:{task}']=v1
                    if task=='shape':
                        energies[prefix+'_target']=float((za-xa).square().sum())
                        energies[prefix+'_profile_source']=v0
                        energies[prefix+'_profile_dynamic']=v1
                        if oracle_color is not None:
                            hc=oracle_color['sd3'][tau]['layers'][layer]
                            hv=oracle_valid['sd3'][tau]['layers'][layer]
                            c=.5*(phi(hc[ai:ai+1],kind)+phi(hc[si:si+1],kind))- .5*(xa+xs)
                            v=.5*(phi(hv[ai:ai+1],kind)+phi(hv[si:si+1],kind))- .5*(xa+xs)
                            energies['ORACLE_'+prefix]=profile(joint,[c,v])[0]
    margins={}
    logits=current['siglip']['logits'][0]
    for task,i in [('shape',1),('color',2),('pose',3)]:margins['sig_'+task]=float(logits[i]-logits[0])
    for tau,probe in current['sd3'].items():
        n=probe['names'];e=probe['denoising_error']
        for task in ['shape','color','pose']:
            if task in n:margins[f'sd3_t{tau}_{task}']=float(e[n.index('source')]-e[n.index(task)])
    return dict(energies=energies,scope=scope,margins=margins)


def auc(legal,wrong):
    l=np.array(legal)[:,None];w=np.array(wrong)[None,:]
    return float(((w>l)+.5*(w==l)).mean())


def threshold(rows,metric):
    a=[r['energies'][metric] for r in rows if r['variant'] in LEGAL]
    return float(max(a)+max(max(a)*1e-6,1e-12))


def summarize(rows,metric,limit):
    legal=[r for r in rows if r['variant'] in LEGAL]
    wrong=[r for r in rows if r['variant'] in ERRORS]
    out=dict(threshold=limit,legal_pass=sum(r['energies'][metric]<=limit for r in legal),legal_count=len(legal),
        wrong_reject=sum(r['energies'][metric]>limit for r in wrong),wrong_count=len(wrong),
        auc=auc([r['energies'][metric] for r in legal],[r['energies'][metric] for r in wrong]),by_variant={})
    byscene={}
    for r in rows:byscene.setdefault(r['scene'],{})[r['variant']]=r['energies'][metric]
    for name in LEGAL+ERRORS:
        rr=[r for r in rows if r['variant']==name]
        out['by_variant'][name]=dict(count=len(rr),accepted=sum(r['energies'][metric]<=limit for r in rr),
            mean=float(np.mean([r['energies'][metric] for r in rr])))
    out['paired_error_above_valid']={name:sum(s[name]>max(s['valid'],s['valid_alt']) for s in byscene.values()) for name in ERRORS}
    out['sources']=len(byscene)
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['dev','holdout'],required=True)
    ap.add_argument('--features',default='features_v1');args=ap.parse_args();torch.set_num_threads(4)
    manifest=json.loads((HERE/'outputs/dataset_v1/manifest.json').read_text())
    out=HERE/'outputs'/'analysis_v1';out.mkdir(exist_ok=True)
    cache=out/f'metrics_{args.stage}.json'
    rows=[]
    for scene in manifest['records']:
        if args.stage=='dev' and scene['split']=='heldout':continue
        if args.stage=='holdout' and scene['split']!='heldout':continue
        folder=HERE/'outputs'/args.features/f"scene{scene['scene']:02d}"
        source=torch.load(folder/'source.pt',weights_only=True)
        color=torch.load(folder/'color_only.pt',weights_only=True)
        valid=torch.load(folder/'valid.pt',weights_only=True)
        for name in scene['variants']:
            current=torch.load(folder/f'{name}.pt',weights_only=True)
            row=dict(scene=scene['scene'],split=scene['split'],variant=name,**values(current,source,color,valid))
            rows.append(row)
        print('METRICS_SCENE',scene['scene'],flush=True)
    cache.write_text(json.dumps(dict(rows=rows),indent=2)+'\n')
    if args.stage=='dev':
        dev=[r for r in rows if r['split']=='development'];cal=[r for r in rows if r['split']=='calibration']
        ranking=[]
        for name in rows[0]['energies']:
            if name.startswith('ORACLE_'):continue
            limit=threshold(dev,name);s=summarize(dev,name,limit)
            ranking.append(dict(metric=name,dev_wrong_reject=s['wrong_reject'],dev_auc=s['auc']))
        ranking.sort(key=lambda r:(-r['dev_wrong_reject'],-r['dev_auc'],r['metric']))
        chosen=ranking[0]['metric']
        selected=['sig_global','sig_l11_patch','sig_l11_gram','sig_l11_centered_gram',chosen]
        # Family-wise comparisons are predeclared, then selected using dev only.
        for family in ['_patch_source','_gram_source','_centered_gram_source','_patch_profile_source',
                       '_patch_profile_dynamic','_gram_profile_source','_gram_profile_dynamic']:
            entries=[r for r in ranking if r['metric'].startswith('sd3_') and r['metric'].endswith(family)]
            if entries:selected.append(entries[0]['metric'])
        selected=list(dict.fromkeys(selected))
        oracle_names=[k for k in rows[0]['energies'] if k.startswith('ORACLE_')]
        oracle_names.sort(key=lambda k:-summarize(dev,k,threshold(dev,k))['wrong_reject'])
        selected.append(oracle_names[0])
        selection=dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),chosen=chosen,
            thresholds={k:threshold(cal,k) for k in selected},ranking=ranking,
            scope='Model/layer selection on sources 0-3 only; scalar maximum-legal thresholds on 4-7; heldout 8-15 untouched.',
            oracle='Oracle source-specific legal shape/color displacement columns use labeled reference targets, diagnostics only.')
        lock=out/'selection.json'
        if lock.exists():assert json.loads(lock.read_text())==selection,'Selection already locked'
        else:lock.write_text(json.dumps(selection,indent=2)+'\n')
        summary={k:summarize(cal,k,v) for k,v in selection['thresholds'].items()}
        (out/'calibration.json').write_text(json.dumps(summary,indent=2)+'\n')
        print('DEV_SELECTED',chosen,ranking[:5],flush=True)
    else:
        selection=json.loads((out/'selection.json').read_text())
        summary={k:summarize(rows,k,v) for k,v in selection['thresholds'].items()}
        # Task selectivity is scored without fitting a per-task threshold.
        byscene={}
        for r in rows:byscene.setdefault(r['scene'],{})[r['variant']]=r
        scope={}
        for key in rows[0]['scope']:
            if not key.endswith(':shape') or 't0.35' not in key:continue
            stem=key[:-6]
            morphology=[];pose=[]
            for source in byscene.values():
                def diff(name,a,b):return source[name]['scope'][stem+':'+a]-source[name]['scope'][stem+':'+b]
                morphology.append(diff('valid','color','shape')-diff('color_only','color','shape'))
                pose.append(diff('pose_only','shape','pose'))
            scope[stem]=dict(morphology_expected_positive=sum(x>0 for x in morphology),pose_expected_positive=sum(x>0 for x in pose),
                sources=len(morphology),morphology_mean=float(np.mean(morphology)),pose_mean=float(np.mean(pose)))
        completion={}
        for key in rows[0]['margins']:
            if not key.endswith('shape'):continue
            completion[key]=dict(
                valid_above_source=sum(s['valid']['margins'][key]>s['source']['margins'][key] for s in byscene.values()),
                valid_above_color_only=sum(s['valid']['margins'][key]>s['color_only']['margins'][key] for s in byscene.values()),
                wrong_head_higher_than_valid=sum(s['wrong_head']['margins'][key]>s['valid']['margins'][key] for s in byscene.values()),
                wrong_heading_higher_than_valid=sum(s['wrong_heading']['margins'][key]>s['valid']['margins'][key] for s in byscene.values()),
                valid_above_wrong_object=sum(s['valid']['margins'][key]>s['wrong_object']['margins'][key] for s in byscene.values()),sources=len(byscene))
        result=dict(selection=selection['chosen'],metrics=summary,task_selectivity=scope,completion=completion)
        (out/'heldout.json').write_text(json.dumps(result,indent=2)+'\n')
        print('HELDOUT_SUMMARY',json.dumps(dict(metrics=summary,completion=completion)),flush=True)


if __name__=='__main__':main()
