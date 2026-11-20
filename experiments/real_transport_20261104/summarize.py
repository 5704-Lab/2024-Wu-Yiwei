"""Standalone real-image figures and source-level summary; all failures retained."""
import csv
import math
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *


def main():
    out=HERE/'outputs/summary_v1'
    if (out/'summary.json').exists():raise RuntimeError('Immutable summary exists')
    out.mkdir(parents=True,exist_ok=True)
    analysis=json.loads((HERE/'outputs/analysis_v3/heldout.json').read_text());key=analysis['primary']
    metrics=analysis['results'][key]
    repair=json.loads((HERE/'outputs/repair_v2/summary.json').read_text())['results']
    rgb=json.loads((HERE/'outputs/repair_rgb_audit_v1/summary.json').read_text())['rows']
    rgb={r['pair']:r for r in rgb}
    table=[]
    for r in repair:
        before=r['evaluations']['unchanged']['reference_center_rmse'];after=r['evaluations']['controlled']['reference_center_rmse']
        table.append(dict(pair=r['pair'],split=r['split'],accepted_steps=r['accepted_steps'],
            initial_R=r['initial_observer']['frame']['rotation_deg'],final_R=r['final_observer']['frame']['rotation_deg'],
            independent_audit_R=r['final_independent_audit']['frame']['rotation_deg'],
            rgb_residual_R=rgb[r['pair']]['scores']['controlled']['offline_residual_angle_deg'],
            before_center_RMSE=before,after_center_RMSE=after,relative_RMSE_reduction=1-after/before,
            final_luminance_ratio=r['evaluations']['controlled']['luminance_ratio']))
    with (out/'repair_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    # Image panel: all six runs, including evaluation-only reference in final col.
    cols=['Source','Dark + added rotation','Controlled recovery','Opposite, same resampling','Dark reference (EVAL ONLY)']
    size=224;canvas=Image.new('RGB',(len(cols)*size,len(repair)*(size+32)+40),'white');draw=ImageDraw.Draw(canvas)
    for i,label in enumerate(cols):draw.text((i*size+4,8),label,fill='black')
    for n,r in enumerate(repair):
        pair=r['pair'];base=HERE/'outputs/repair_v2'/pair
        files=[base/'source.png',base/'initial.png',base/'controlled.png',
            HERE/'outputs/repair_rgb_audit_v1'/pair/'resampling_matched_wrong_direction.png',base/'offline_reference_dark.png']
        y=40+n*(size+32)
        draw.text((4,y),pair+'  '+r['split'],fill='black')
        for i,p in enumerate(files):canvas.paste(Image.open(p).convert('RGB').resize((size,size)),(i*size,y+25))
    canvas.save(out/'recovery_all_sources.png')
    # Known-transform scatter, supports and failures both visible.
    fig,ax=plt.subplots(figsize=(6.4,5))
    truth=json.loads((HERE/'outputs/data_v1/manifest.json').read_text())['records']
    lookup={(r['pair'],n):v for r in truth for n,v in r['variants'].items()}
    xx=[];yy=[];support=[]
    for group in ['nonrotation','rotation']:
        for row in metrics[group]['rows']:
            a=np.array(lookup[(row['pair'],row['variant'])]['matrix']);u,_,vh=np.linalg.svd(a);rr=u@vh
            xx.append(np.degrees(np.arctan2(rr[1,0],rr[0,0])));yy.append(row['rotation_deg']);support.append(row['quality_supported'])
    support=np.array(support);xx=np.array(xx);yy=np.array(yy)
    ax.scatter(xx[support],yy[support],s=18,alpha=.65,label='Quality gate passed')
    ax.scatter(xx[~support],yy[~support],s=40,marker='x',color='red',label='Unsupported')
    ax.plot([-27,27],[-27,27],'k--',lw=1);ax.set_xlabel('Known input polar rotation (deg)');ax.set_ylabel('Estimated source-frame rotation (deg)')
    ax.legend(fontsize=8);ax.set_title('12 held-out real sources; independent probe noise')
    fig.tight_layout();fig.savefig(out/'heldout_rotation.png',dpi=180);plt.close(fig)
    # Original actual FE states, no correctness labels manufactured.
    pairs=['bear__00','bus__00','tiger__00','dogs__00'];names=['source','flow_step23','flow_step33','flow_step43','flow_final']
    canvas=Image.new('RGB',(5*224,len(pairs)*263),'white');draw=ImageDraw.Draw(canvas)
    cases={}
    for ri,pair in enumerate(pairs):
        cases[pair]={}
        for ci,name in enumerate(names):
            if name=='source':p=HERE/'outputs/data_v1'/pair/'source.png';label=pair+' source'
            else:
                folder='data_v1' if name=='flow_step33' else 'trajectory_data_v1'
                p=HERE/'outputs'/folder/pair/(name+'.png')
                metric_folder=('development_weighted_v2' if pair.split('__')[0] in DEV else 'calibration_weighted_v1' if pair.split('__')[0] in CAL else 'heldout_weighted_v1') if name=='flow_step33' else 'trajectory_weighted_v1'
                r=json.loads((HERE/'outputs'/metric_folder/pair/(name+'.json')).read_text())['results'][key]
                cases[pair][name]=dict(global_R=r['global_frame'].get('rotation_deg'),object_R=r['object_frame'].get('rotation_deg'),local_p90=r['local_rotation_abs_p90'],mutual_fraction=r['mutual_fraction'])
                label=name+' global R='+str(round(r['global_frame'].get('rotation_deg') or 0.,2))
            x=ci*224;y=ri*263;draw.text((x+3,y+5),label,fill='black')
            canvas.paste(Image.open(p).convert('RGB').resize((224,224)),(x,y+32))
    canvas.save(out/'actual_flowedit_states.png')
    angle_errors=[];state_hashes=[]
    for folder in ['heldout_weighted_v1']:
        for p in (HERE/'outputs'/folder).glob('*__00/*.json'):
            row=json.loads(p.read_text());e=row['results'][key].get('evaluation')
            if e and e.get('global_frame_angle_error_deg') is not None:angle_errors.append(e['global_frame_angle_error_deg'])
    for folder in ['data_v1','trajectory_data_v1']:
        for p in (HERE/'outputs'/folder).glob('*__00/*.pt'):
            if folder=='trajectory_data_v1' and p.stem=='source':continue
            state_hashes.append(digest(torch.load(p,weights_only=True)['y']))
    write(out/'summary.json',dict(primary=key,source_count=20,development_sources=4,calibration_sources=4,test_sources=12,
        source_conditioned_records=340,unique_latent_states=len(set(state_hashes)),actual_saved_FE_states=80,
        new_full_FE_trajectories=0,short_recovery_v1=6,short_recovery_v2=6,
        primary_test={g:{k:v for k,v in r.items() if k!='rows'} for g,r in metrics.items()},
        known_affine_angle_mean_error=float(np.mean(angle_errors)),known_affine_angle_median_error=float(np.median(angle_errors)),
        recovery=table,actual_FE_cases=cases,
        median_recovery_RMSE_reduction=float(np.median([r['relative_RMSE_reduction'] for r in table])),
        newly_recovery_heldout_sources=['dogs__00','dogs2__00','penguins__00','japanese_castle__00'],
        all_hypothesis_semantic_safety_certified=False,arbitrary_text_or_category_shape_verified=False))
    print('SUMMARY',json.loads((out/'summary.json').read_text())['primary_test'],flush=True)


if __name__=='__main__':main()
