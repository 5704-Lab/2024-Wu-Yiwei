"""Separate proxy feasibility, reference fidelity and oracle upper bounds."""
import argparse
import json
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent


def auc(bad,legal):
    comparisons=[float(b>l)+.5*float(b==l) for b in bad for l in legal]
    return float(np.mean(comparisons)) if comparisons else None


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='v1');args=ap.parse_args()
    out=HERE/'outputs'/args.tag
    data=json.loads((out/'probe_results.json').read_text())
    scenes=data['records'];rows=data['rows']
    calibration=[r for r in rows if r['split']=='calibration' and r['case']=='correct']
    held=[r for r in rows if r['split']=='heldout']
    metrics=['patch_energy','shared_feature_mse','geometry','phase','correspondence_feature',
             'correspondence_position','correspondence_oriented']
    thresholds={m:1.25*max(r[m] for r in calibration)+1e-12 for m in metrics}
    floor=.5*float(np.median([r['text_gain'] for r in calibration]));text_calibrated=floor>0
    held_scenes=[s for s in scenes if s['split']=='heldout']
    result=dict(tag=args.tag,source_count=len(scenes),heldout_source_count=len(held_scenes),
                rows=len(rows),synthetic_only=True,independent_natural_quality_validation=False,
                text_floor=floor,text_proxy_calibrated=text_calibrated,
                local_witness_thresholds=thresholds,observer_results={},selection=[],coverage={})
    legal=[r for r in held if r['case']=='correct']
    for metric in metrics:
        by_case={}
        for name in ['wrong_object','both_objects','turned','shifted','duplicate']:
            bad=[r for r in held if r['case']==name]
            by_case[name]=dict(rejected=sum(r[metric]>thresholds[metric] for r in bad),total=len(bad),
                               auc_bad_vs_legal=auc([r[metric] for r in bad],[r[metric] for r in legal]))
        result['observer_results'][metric]=dict(legal_accepted=sum(r[metric]<=thresholds[metric] for r in legal),
                                                legal_total=len(legal),bad=by_case)
    result['original_budget_legal_accepted']=sum(next(r for r in s['rows'] if r['case']=='correct')['patch_energy']<=s['global_budget']
                                                for s in held_scenes)
    family_names=list(held_scenes[0]['fits'])
    readout_keys={'model_original_budget':'text_gain','model_local_calibration':'text_gain'}
    if 'MATERIAL' in family_names:
        readout_keys.update(model_assignment='assignment_gain',model_equivariant='equivariant_gain')
    result['task_floors']={key:.5*float(np.median([r[field] for r in calibration])) for key,field in readout_keys.items()}
    for name in family_names:
        selected=[r for r in held if r['case']=='fit_'+name]
        result['coverage'][name]=dict(within_reference_tolerance=sum(r['oracle_joint_within_tolerance'] for r in selected),
                                      total=len(selected),median_reference_mse=float(np.median([r['oracle_reference_mse'] for r in selected])),
                                      median_projected_gradient_inf=float(np.median([s['fits'][name]['projected_gradient_inf'] for s in held_scenes])))
    for s in held_scenes:
        for bank in ['restricted_fit','expanded_oracle']:
            candidates=[r for r in s['rows'] if r['case']=='source' or r['case'].startswith('fit_')]
            if bank=='expanded_oracle':candidates=s['rows']
            for readout in list(readout_keys)+['oracle_reference']:
                budget=s['global_budget'] if readout=='model_original_budget' else thresholds['patch_energy']
                this_floor=result['task_floors'].get(readout,0)
                field=readout_keys.get(readout,'text_gain')
                feasible=[r for r in candidates if this_floor>0 and r[field]>=this_floor and r['patch_energy']<=budget]
                if readout=='oracle_reference':
                    chosen=min(candidates,key=lambda r:r['oracle_reference_mse']);status='oracle_best_available'
                elif feasible:
                    chosen=min(feasible,key=lambda r:r['patch_energy']);status='proxy_feasible'
                else:
                    chosen=next(r for r in candidates if r['case']=='source');status='hold_no_proxy_feasible_candidate'
                result['selection'].append(dict(scene=s['scene'],bank=bank,readout=readout,selected=chosen['case'],status=status,
                                                 oracle_reference_mse=chosen['oracle_reference_mse'],
                                                 within_reference_tolerance=chosen['oracle_joint_within_tolerance']))
    result['selection_summary']=[]
    for bank in ['restricted_fit','expanded_oracle']:
        for readout in list(readout_keys)+['oracle_reference']:
            group=[r for r in result['selection'] if r['bank']==bank and r['readout']==readout]
            result['selection_summary'].append(dict(bank=bank,readout=readout,total=len(group),
                     within_reference_tolerance=sum(r['within_reference_tolerance'] for r in group),
                     held=sum(r['status'].startswith('hold') for r in group),selected=[r['selected'] for r in group]))
    (out/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['text_floor','original_budget_legal_accepted','coverage','selection_summary','observer_results']},indent=2))


if __name__=='__main__':main()
