"""Correctly directed pose-scope and bidirectional error diagnostics."""
import json
from pathlib import Path

import numpy as np
import torch

from analyze_induced import evaluate_record

HERE=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4);rows=[];scope={}
    thresholds=json.loads((HERE/'outputs/analysis_v2/thresholds.json').read_text())
    for scene in range(8,16):
        folder=HERE/'outputs/features_challenge'/f'scene{scene:02d}'
        source=torch.load(folder/'source.pt',weights_only=True)
        aux={f'{n}_{t}':torch.load(HERE/'outputs/induced_v2'/f'scene{scene:02d}'/f'{n}_{t}.pt',weights_only=True)
             for n in ['shape','color','morph','pose'] for t in [.35,.65]}
        for name in ['pose_front_only','wrong_head_front','wrong_heading_front']:
            item=torch.load(folder/f'{name}.pt',weights_only=True)
            obs=evaluate_record(item,source,aux)
            rows.append(dict(scene=scene,variant=name,**obs))
            if name=='pose_front_only':
                for key,value in obs['scope'].items():
                    if not key.endswith(':shape'):continue
                    stem=key[:-6];diff=value-obs['scope'][stem+':pose']
                    scope.setdefault(stem,[]).append(diff)
        print('CHALLENGE_ANALYSIS',scene,flush=True)
    result=dict(rows=rows,source_count=8,new_independent_sources=False,
        shape_task_error_rejections={metric:{name:sum(r['energies'][metric]>threshold for r in rows if r['variant']==name)
            for name in ['wrong_head_front','wrong_heading_front']} for metric,threshold in thresholds.items()},
        corrected_pose_selectivity={key:dict(expected_positive=sum(v>0 for v in vals),sources=len(vals),mean=float(np.mean(vals))) for key,vals in scope.items()},
        invalidated_old_fields='analysis_v1 and analysis_v2 task_selectivity.*.pose_expected_positive used the opposite physical rotation; see POSE_SCOPE_CORRECTION.md')
    p=HERE/'outputs/challenge_analysis.json';p.write_text(json.dumps(result,indent=2)+'\n')
    print('CORRECTED_SCOPE',result['corrected_pose_selectivity'],flush=True)
    print('FRONT_ERRORS',result['shape_task_error_rejections'],flush=True)


if __name__=='__main__':main()
