"""Recompute every accepted source-tube state independently of controller logs."""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from patches import CleanScorer,PatchRelation,ROOT,seed_setup,write


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tags',nargs='+',default=['v3_tube'])
    args=ap.parse_args();seed_setup();sc=CleanScorer(torch.device('cuda:0'),torch.float32,'native');rows=[]
    aggregate_path=HERE/'outputs/analysis/tube_recomputation.json'
    previous=json.loads(aggregate_path.read_text())['rows'] if aggregate_path.exists() else []
    for tag in args.tags:
        for path in sorted((HERE/'outputs'/tag/'full').glob('*/*/*/complete.json')):
            result=json.loads(path.read_text());saved=torch.load(path.parent/'trajectory.pt',weights_only=True)
            if result['mode'] not in ['PATCH_TUBE','PATCH_ADAPT']:continue
            pair=result['pair'];x=torch.load(ROOT/'experiments/stage1_full/inputs'/f'{pair}.pt',weights_only=True)['x'].to(sc.device)
            with torch.no_grad():source=sc.image(x)
            relation=PatchRelation(sc,source,rank=2);energies=[]
            with torch.no_grad():
                for state in saved['states']:
                    energies.append(float(.5*relation(sc.image(state.to(sc.device))).square().sum()))
            budget=result['rows'][0]['tube_budget']
            error=max(abs(a-r['tube_actual']) for a,r in zip(energies[1:],result['rows']))
            row=dict(tag=tag,pair=pair,seed=result['seed'],states=len(energies),budget=budget,
                maximum_energy=max(energies),log_recomputation_error=error,
                scaled=sum(r['tube_scale']<1 for r in result['rows']),
                holds=sum(r['tube_scale']==0 for r in result['rows']),
                task_floor_failures=sum(not r['task_floor_satisfied'] for r in result['rows']))
            row['passed']=max(energies)<=budget+1e-7 and error<1e-6 and energies[0]<1e-9
            rows.append(row)
            combined={(r['tag'],r['pair'],r['seed']):r for r in previous+rows}
            aggregate=list(combined.values())
            write(aggregate_path,dict(rows=aggregate,passed=all(r['passed'] for r in aggregate)))
            write(HERE/'outputs/analysis'/('tube_recomputation_'+'_'.join(args.tags)+'.json'),
                  dict(rows=rows,passed=all(r['passed'] for r in rows)))
            assert row['passed'],row
            print(row,flush=True)
    print('TUBE_RECOMPUTATION_COMPLETE',flush=True)


if __name__=='__main__':
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]):main()
