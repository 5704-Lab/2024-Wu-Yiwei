"""Post-hoc diagnosis of existing native tensors, NOT a fresh holdout trial.

Checks whether preserving full concept distributions or normalizing raw source
feature scales alone resolves the observed semantic coupling. No learned weights.
"""
from pathlib import Path
import hashlib
import json
import sys
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
NATIVE=ROOT/'experiments/native_feedback_20261030'
sys.path.insert(0,str(NATIVE))
from analyze import spatial, summary, LEGAL
from probe import write


def scores(item,source):
    out={}
    for layer in [3,11,20]:
        for facet in ['image_O','context_C']:
            vals={'relative_norm':[], 'raw':[]}
            if facet=='image_O':vals.update(source_diagonal=[],spatial_contrast=[])
            for a,s in zip(item['observations'],source['observations']):
                v=a['layers'][layer][facet].double();u=s['layers'][layer][facet].double()
                e=(v-u).square().mean()
                vals['raw'].append(e)
                vals['relative_norm'].append(e/u.square().mean().clamp_min(1e-12))
                if facet=='image_O':
                    variance=u.var(1,unbiased=False,keepdim=True)
                    floor=.1*variance[variance>0].median()
                    vals['source_diagonal'].append(((v-u).square()/(variance+floor)).mean())
                    vc=v-v.mean(1,keepdim=True);uc=u-u.mean(1,keepdim=True)
                    vals['spatial_contrast'].append((vc-uc).square().mean()/uc.square().mean().clamp_min(1e-12))
            for mode,values in vals.items():out[f'{facet}/l{layer}/{mode}']=float(torch.stack(values).mean())
        for branch in ['dynamic','anchored']:
            for mode in ['qk','cos_competition']:
                values=[]
                for a,s in zip(item['observations'],source['observations']):
                    p=spatial(a['layers'][layer],branch,mode)[[1,2,4,11,12,13,14,15,16]]
                    q=spatial(s['layers'][layer],branch,mode)[[1,2,4,11,12,13,14,15,16]]
                    values.append(.5*(p.sqrt()-q.sqrt()).square().sum(-1).mean())
                out[f'full_distribution/{branch}/{mode}/l{layer}']=float(torch.stack(values).mean())
    return out


def main():
    torch.set_num_threads(4)
    out=HERE/'outputs/posthoc_v1'
    if out.exists():raise RuntimeError('Immutable post-hoc diagnosis exists')
    write(out/'protocol.json',dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        stage='exploratory reanalysis after viewing all prior outcomes; NOT independent confirmation',
        calibration='historical scenes 4-7, five legal progress states per source; never refit on fresh camera sides',
        no_runtime_reference=True,no_trained_readout=True,metric_families='full Hellinger distribution, raw scale normalization, source diagonal, spatial contrast',
        caveat='spatial contrast is blind to spatially uniform channel shifts; normalization is not semantic permission'))
    limits=None
    for tag,scenes,label in [('native_v3',range(8),'calibration_diagnosis'),('native_v3',range(8,16),'historical_reanalysis'),
                              ('native_fresh_v1',range(8),'fresh_data_reanalysis')]:
        rows=[]
        for scene in scenes:
            folder=NATIVE/'outputs'/tag/f'scene{scene:02d}'
            src=torch.load(folder/'source.pt',weights_only=True)
            for p in sorted(folder.glob('*.pt')):
                item=torch.load(p,weights_only=True)
                rows.append(dict(scene=scene,variant=p.stem,scores=scores(item,src)))
            print('REANALYZE',label,scene,flush=True)
        if limits is None:
            limits={k:max(r['scores'][k] for r in rows if r['scene'] in [4,5,6,7] and r['variant'] in LEGAL)*1.000001+1e-14
                    for k in rows[0]['scores']}
            write(out/'limits.json',limits)
        index={}
        for r in rows:index.setdefault(r['scene'],{})[r['variant']]=r['scores']
        pairs={k:{name:sum(r[name][k]>r['valid'][k] for r in index.values()) for name in ['wrong_head','wrong_head_front'] if name in next(iter(index.values()))}
               for k in limits}
        write(out/(label+'.json'),dict(rows=rows,results=summary(rows,limits),paired_ordering=pairs,
            independent_source_count=len(index),fresh_holdout_claim=False))


if __name__=='__main__':main()
