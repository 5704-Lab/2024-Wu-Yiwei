"""Same-state solver/dual-strength audit after a source tube stalls."""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from run import CFG,Engine,noises,panel,save_image,seed_setup,write
from patches import PatchGeometry,PatchRelation


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pair',default='bear__00');ap.add_argument('--step',type=int,default=31)
    ap.add_argument('--memory-fraction',type=float,default=.5);args=ap.parse_args()
    seed_setup();torch.cuda.set_per_process_memory_fraction(args.memory_fraction,0)
    run=Engine();ctx=run.setup(args.pair);noise=noises(ctx,0)
    dest=HERE/'outputs/solver_probe'/args.pair;dest.mkdir(parents=True,exist_ok=True)
    path=HERE/'outputs/v3_tube/full'/args.pair/'seed0/PATCH_TUBE'
    states=torch.load(path/'trajectory.pt',weights_only=True)['states']
    result=json.loads((path/'complete.json').read_text());budget=result['rows'][0]['tube_budget']
    y=states[args.step-17].to(run.device);query=run.query(ctx,y,args.step,noise[args.step-17])
    relation=PatchRelation(run.sc,ctx['source'],rank=2)
    control_ctx={**ctx,'geometry':PatchGeometry(relation)}
    with sdpa_kernel(SDPBackend.MATH):
        point=(y+query['cf']).detach().requires_grad_(True)
        loss=.5*relation(run.sc.image(point)).square().sum()
        gradient,=torch.autograd.grad(loss,point)
    coupling=float((gradient.double()*query['task'].double()).sum()/
                   (gradient.double().norm()*query['task'].double().norm()).clamp_min(1e-12))
    del point,loss,gradient
    rows=[]
    for count,multiplier,retention in [(6,1,0),(24,1,0),(24,10,0),(48,100,0),(24,10,1)]:
        CFG['cg_steps']=count;CFG['task_cg_steps']=count;CFG['gradient_step']=1.2*multiplier
        with sdpa_kernel(SDPBackend.MATH):
            new,info=run.correct(control_ctx,y,query['cf'],query['task'],retention)
        metric,image=run.endpoint(ctx,new)
        row=dict(step=args.step,cg_steps=count,source_multiplier=multiplier,retention=retention,
            budget=budget,source_gradient_task_cosine=coupling,
            inside=info['geometry_after']<=budget,endpoint=metric,**info)
        rows.append(row);write(dest/'results.json',dict(rows=rows,scope='identical real state/noise/proposal; finite local result, not a complete editing trajectory'))
        save_image(image,dest/f'cg{count}_lambda{multiplier}_ret{retention}.png')
        print(args.pair,count,multiplier,retention,info['geometry_after'],budget,info['solver'],info['final_progress'],flush=True)
    print('SOLVER_PROBE_COMPLETE',flush=True)


if __name__=='__main__':
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION,SDPBackend.EFFICIENT_ATTENTION,SDPBackend.MATH]):main()
