"""Fixed-time finite conditional work vs the controller's tangent halfspace.

These are model-response integrals, not calibrated semantic completion scores.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run import CFG,Engine,ROOT,noises,seed_setup,write


def conditional_field(run,ctx,y,k,noise):
    t=run.times[k]/1000
    q=(1-t)*ctx['x']+t*noise+y-ctx['x']
    seq=torch.cat([ctx['seq'][1:2],ctx['seq'][3:4]])
    pool=torch.cat([ctx['pool'][1:2],ctx['pool'][3:4]])
    with torch.no_grad():
        pred=run.model(hidden_states=torch.cat([q,q]),timestep=run.times[k].expand(2),
            encoder_hidden_states=seq,pooled_projections=pool,joint_attention_kwargs=None,return_dict=False)[0]
    s,a=pred.chunk(2)
    return s-a


def finite_work(run,ctx,y,d,k,noise,intervals=4):
    values=[];norm=None
    for j in range(intervals+1):
        g=conditional_field(run,ctx,y+(j/intervals)*d,k,noise)
        if j==0:norm=max(float(g.norm()),1e-12)
        values.append(float((g.double()*d.double()).sum())/norm)
    weights=[1]+[4 if j%2 else 2 for j in range(1,intervals)]+[1]
    integral=sum(w*v for w,v in zip(weights,values))/(3*intervals)
    return dict(linear=values[0],simpson=integral,nodes=values,
                coarse_simpson=(values[0]+4*values[intervals//2]+values[-1])/6 if intervals==4 else None)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='v1_full')
    ap.add_argument('--pairs',nargs='+',default=['bear__00','brown_owl__00'])
    ap.add_argument('--modes',nargs='+',default=['CF','JOINT'])
    ap.add_argument('--steps',nargs='+',type=int,default=[17,25,33,41,47,49])
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--compare-raw',action='store_true')
    ap.add_argument('--memory-fraction',type=float,default=1.)
    args=ap.parse_args()
    seed_setup()
    if args.memory_fraction<1:torch.cuda.set_per_process_memory_fraction(args.memory_fraction,0)
    run=Engine();rows=[]
    suffix='response_paired' if args.compare_raw else 'response'
    out=HERE/'outputs'/f'{args.tag}_{suffix}_seed{args.seed}'
    for pair in args.pairs:
        ctx=run.setup(pair);noise=noises(ctx,args.seed)
        for mode in args.modes:
            path=HERE/'outputs'/args.tag/'full'/pair/f'seed{args.seed}'/mode
            saved=torch.load(path/'trajectory.pt',weights_only=True)
            result=json.loads((path/'complete.json').read_text())
            for k in args.steps:
                i=k-17;y=saved['states'][i].to(run.device)
                d=saved['states'][i+1].to(run.device)-y
                work=finite_work(run,ctx,y,d,k,noise[i])
                original=result['rows'][i]
                row=dict(pair=pair,mode=mode,step=k,**work,
                    controller_floor=original.get('progress_floor'),
                    controller_linear=original.get('final_progress'))
                if 'progress_floor' in original:
                    row['finite_floor_passed']=work['simpson']>=original['progress_floor']-1e-4
                    row['quadrature_difference']=abs(work['simpson']-work['coarse_simpson'])
                if args.compare_raw:
                    raw=run.query(ctx,y,k,noise[i])
                    raw_work=finite_work(run,ctx,y,raw['cf'],k,noise[i])
                    retention=result['config']['retention']
                    finite_floor=retention*max(raw_work['simpson'],0)+min(raw_work['simpson'],0)
                    row.update(raw_cf_work=raw_work,finite_relative_floor=finite_floor,
                        finite_relative_floor_passed=work['simpson']>=finite_floor-1e-4,
                        finite_work_ratio=work['simpson']/raw_work['simpson'] if abs(raw_work['simpson'])>1e-8 else None)
                rows.append(row);write(out/'results.json',dict(rows=rows,scope='same-time model conditional work, not independent semantic evidence'))
                print(pair,mode,k,work['linear'],work['simpson'],row.get('finite_floor_passed'),flush=True)
    print('RESPONSE_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]):main()
