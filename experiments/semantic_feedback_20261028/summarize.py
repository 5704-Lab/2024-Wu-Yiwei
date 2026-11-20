"""Durable source-level counts, numeric audits and standalone scientific plots."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs';DEST=OUT/'summary';DEST.mkdir(exist_ok=True)


def read(name):return json.loads((OUT/name).read_text())


def main():
    sources=[];images=set()
    for tag in ['dataset_v1','dataset_v2','dataset_v3','dataset_v4','dataset_challenge']:
        m=read(tag+'/manifest.json')
        for row in m['records']:
            if tag!='dataset_challenge':sources.append(row['variants']['source']['sha256'])
            images.update(v['sha256'] for v in row['variants'].values())
    repair=read('repair_gn_v2/summary.json');rows=repair['rows'];groups={}
    for mode in ['TEXT','INDUCED','ORACLE']:
        for variant in ['valid','wrong_head','wrong_heading']:
            rr=[r for r in rows if r['mode']==mode and r['variant']==variant]
            groups[mode+'_'+variant]=dict(count=len(rr),energy_ratio_mean=float(np.mean([r['observer_final']/r['observer_start'] for r in rr])),
                target_rgb_delta_mean=float(np.mean([r['target_rgb_rmse_after']-r['target_rgb_rmse_before'] for r in rr])),
                distance_improved=sum(r['target_rgb_rmse_after']<r['target_rgb_rmse_before'] for r in rr),
                repair_chord_cosine_mean=float(np.mean([r['repair_chord_cosine'] for r in rr])),
                source_reversion_cosine_mean=float(np.mean([r['source_reversion_cosine'] for r in rr])))
    fd=[r['fdcheck']['relative_error'] for r in rows if r.get('fdcheck')]
    adj=[h['solver']['adjoint_relative_error'] for r in rows for h in r['history'] if 'adjoint_relative_error' in h.get('solver',{})]
    full={m:read('full_bear_v2/'+m+'/complete.json') for m in ['BASE','TEXT','INDUCED','SCALE_TEXT','SCALE_INDUCED']}
    numeric=dict(finite_difference_relative_error_range=[min(fd),max(fd)],adjoint_max_relative_error=max(adj),
        retained_full_baseline_relative_error=full['BASE']['endpoint']['retained_baseline_relative_error'])
    result=dict(unique_controlled_sources=len(set(sources)),source_records=len(sources),unique_controlled_images=len(images),
        short_repair_trajectories=len(rows),repair_groups=groups,numeric=numeric,
        full_pilot={m:dict(endpoint=v['endpoint'],accepted=v.get('accepted')) for m,v in full.items()},
        invalidated_fields='Old v1/v2 pose_expected_positive; see POSE_SCOPE_CORRECTION.md',
        scripts_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')})
    (DEST/'synthesis.json').write_text(json.dumps(result,indent=2)+'\n')
    # Observation and nuisance ablation, with both required outcomes visible.
    v2=read('analysis_v2/heldout.json')['metrics']
    keys=['sig_global','sig_l3_patch','patch_l11_RAW','patch_l11_TEXT','patch_l11_INDUCED','patch_l11_SPLIT']
    labels=['Global SigLIP','Early patches','SD3 raw','Text profile','Velocity profile','Split profile']
    x=np.arange(len(keys));fig,ax=plt.subplots(figsize=(10,4))
    ax.bar(x-.18,[v2[k]['legal_pass']/40 for k in keys],.36,label='Legal-state acceptance')
    ax.bar(x+.18,[v2[k]['wrong_reject']/48 for k in keys],.36,label='Error rejection')
    ax.set(xticks=x,xticklabels=labels,ylim=(0,1.12),ylabel='Fraction',title='Fresh v2 holdout: 8 independent sources')
    ax.legend(loc='lower left');fig.tight_layout();fig.savefig(DEST/'observer_ablation.png',dpi=180);fig.savefig(DEST/'observer_ablation.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5));colors={'TEXT':'#d95f02','INDUCED':'#7570b3','ORACLE':'#1b9e77'}
    marks={'valid':'s','wrong_head':'o','wrong_heading':'^'}
    for r in rows:
        ax.scatter(r['observer_final']/r['observer_start'],r['target_rgb_rmse_after']-r['target_rgb_rmse_before'],
            color=colors[r['mode']],marker=marks[r['variant']],s=45,alpha=.8)
    for key,color in colors.items():ax.scatter([],[],color=color,label=key+' (reference-assisted)' if key=='ORACLE' else key)
    ax.axhline(0,color='gray',lw=1);ax.set(xlabel='Final / initial observer error',ylabel='Change in RGB distance to a known legal state',
        title='A lower proxy error does not ensure selective repair')
    color_legend=ax.legend(loc='upper right');ax.add_artist(color_legend)
    handles=[]
    for key,label in [('valid','Already legal'),('wrong_head','Head articulation error'),('wrong_heading','Whole-body heading error')]:
        handles.append(ax.scatter([],[],color='gray',marker=marks[key],label=label))
    ax.legend(handles=handles,loc='lower right',fontsize=8)
    fig.tight_layout();fig.savefig(DEST/'repair_tradeoff.png',dpi=180);fig.savefig(DEST/'repair_tradeoff.svg');plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
