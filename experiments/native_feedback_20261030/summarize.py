"""Standalone figures and compact source-level evidence. No image generation."""
import csv
import json
import math
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from probe import HERE, ROOT, write
from analyze import spatial, geometry
from binding import windows, read
from landmarks import truth


def main():
    torch.set_num_threads(4)
    out = HERE/'outputs/summary_v1'
    if out.exists():
        raise RuntimeError('Immutable summary exists')
    out.mkdir(parents=True)
    d = json.loads((HERE/'outputs/analysis_v1/native_fresh_v1_audit.json').read_text())
    b = json.loads((HERE/'outputs/binding_v1/fresh_holdout.json').read_text())
    candidates = ['raw/image_O/l11','raw/context_C/l11','dynamic/cos_competition/l11/angles',
                  'anchored/qk/l11/angles','source_graph/anchored/cos_competition/l11/angles']
    records = []
    for k in candidates:
        v = (b if k.startswith('source_graph') else d)['results'][k]
        records.append(dict(method=k,legal_accepted=v['legal_accepted'],legal_total=v['legal_total'],
            head_away_rejected=v['by_error']['wrong_head']['rejected'],head_front_rejected=v['by_error']['wrong_head_front']['rejected'],
            head_total=16,all_errors_rejected=v['errors_rejected'],all_errors_total=v['errors_total']))
    with (out/'fresh_metrics.csv').open('w') as f:
        w = csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    fig,ax = plt.subplots(figsize=(11,4))
    y = np.arange(len(records))
    ax.barh(y-.16,[r['legal_accepted']/r['legal_total'] for r in records],.32,label='Legal progress accepted')
    ax.barh(y+.16,[(r['head_away_rejected']+r['head_front_rejected'])/16 for r in records],.32,label='Unrequested head turns rejected')
    ax.set_yticks(y,[r['method'] for r in records],fontsize=8);ax.set_xlim(0,1.05)
    ax.set_xlabel('Fraction of states, 8 independent fresh sources; frozen calibration')
    ax.legend(loc='lower right',fontsize=8);ax.invert_yaxis();fig.tight_layout();fig.savefig(out/'selectivity.png',dpi=170);plt.close(fig)
    manifest = json.loads((HERE/'outputs/fresh_v1/manifest.json').read_text())
    entry = manifest['records'][0]
    folder = HERE/'outputs/native_fresh_v1/scene00'
    src = torch.load(folder/'source.pt',weights_only=True)
    fig,axes = plt.subplots(3,3,figsize=(10,10))
    for ri,name in enumerate(['source','valid','wrong_head_front']):
        im = np.asarray(Image.open(HERE/entry['variants'][name]['file']))
        item = torch.load(folder/f'{name}.pt',weights_only=True)
        expected = truth(entry['seed'],entry['camera_side'],**entry['variants'][name]['factors'])
        ps = [spatial(o['layers'][11],'dynamic','cos_competition') for o in item['observations']]
        qs = [spatial(o['layers'][11],'dynamic','cos_competition') for o in src['observations']]
        p = torch.stack(ps).mean(0)
        phrase = torch.stack([geometry(x)[0][[11,12,13,14,15,16]].reshape(2,3,2) for x in ps]).mean(0)
        graph = torch.stack([read(x,windows(q)[0])[0] for x,q in zip(ps,qs)]).mean(0)
        for ci,(title,c) in enumerate([('OFFLINE projected centers',expected),('Native phrase moments',phrase),('Source soft graph',graph)]):
            ax = axes[ri,ci];ax.imshow(im);ax.axis('off')
            if ci:
                heat = p[1 if ci==2 else 11].reshape(32,32)
                ax.imshow(heat,extent=(-.5,511.5,511.5,-.5),cmap='magma',alpha=.28,interpolation='bilinear')
            for instance,col in enumerate(['cyan','lime']):
                xy = (c[instance].numpy()+.5)*512
                ax.plot(xy[:,0],xy[:,1],'o',color=col,markersize=4)
                for j,label in enumerate(['H','N','B']):
                    ax.text(xy[j,0]+4,xy[j,1],label,color=col,fontsize=9)
                ax.plot(xy[:2,0],xy[:2,1],color=col)
            ax.set_title(name+'\n'+title,fontsize=9)
    fig.suptitle('Soft maps are observations; H/N/B = head/nose/body. Truth is evaluation only.',fontsize=10)
    fig.tight_layout();fig.savefig(out/'landmark_diagnostic.png',dpi=160);plt.close(fig)
    # Natural source and actual clean edited states: only generic noun concepts.
    nf = HERE/'outputs/natural_bear_v1'
    fig,axes = plt.subplots(2,4,figsize=(13,6.6))
    names = ['source','source_dark','source_mirror','baseline']
    for ci,name in enumerate(names):
        im = np.asarray(Image.open(ROOT/'experiments/semantic_feedback_20261028/outputs/natural_v1/bear__00'/f'{name}.png'))
        item = torch.load(nf/f'{name}.pt',weights_only=True)
        for ri,branch in enumerate(['dynamic','anchored']):
            ps = [spatial(o['layers'][11],branch,'cos_competition') for o in item['observations']]
            centers = torch.stack([p for p in ps]).mean(0)
            a = (torch.arange(32).double()+.5)/32-.5
            yy,xx = torch.meshgrid(a,a,indexing='ij')
            xy = ((centers@torch.stack([xx,yy],-1).reshape(-1,2))+.5)*512
            ax = axes[ri,ci];ax.imshow(im);ax.axis('off')
            for j,label in [(1,'H'),(2,'N'),(4,'B')]:
                ax.plot(float(xy[j,0]),float(xy[j,1]),'o',color='cyan',markersize=4)
                ax.text(float(xy[j,0])+4,float(xy[j,1]),label,color='cyan')
            ax.plot(xy[[1,2],0],xy[[1,2],1],color='cyan')
            ax.set_title(branch+' / '+name,fontsize=9)
    fig.suptitle('Single real bear diagnostic; natural endpoint has no success ground truth.',fontsize=10)
    fig.tight_layout();fig.savefig(out/'natural_diagnostic.png',dpi=160);plt.close(fig)
    write(out/'summary.json',dict(fresh_results=records, source_units=dict(development=4,calibration=4,
        historical_holdout=8,fresh_holdout=8,natural_diagnostic=1),
        main_native_states=312,natural_states=9,main_paired_settings_per_state=4,
        main_forward_calls=321*4,source_anchoring_cache_forward_calls=25*4,
        full_editing_trajectories_newly_run=0,hard_controller_admitted=False,
        reason='protected native facts not reliable under legal morphology; goal completion readout also remains unvalidated'))
    print('SUMMARY_COMPLETE',records,flush=True)


if __name__ == '__main__':
    main()
