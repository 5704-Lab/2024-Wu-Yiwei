"""Post hoc derivative/model audit; changes no editing outputs."""
import json
import time

import numpy as np
import torch

from run import Experiment, ROOT, OUT, PAIRS, CFG, tensor_norm, write_json


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    exp=Experiment("cuda:0")
    rows=[]
    for pair in PAIRS:
        ctx=exp.setup(pair)
        baseline=torch.load(ROOT/"experiments/flowedit_frequency_trajectory/outputs"/pair/"trajectory.pt",
                            weights_only=True)
        points=[(f"BASE_before_{k}",baseline["states"][k].to(exp.device),
                 (baseline["states"][k+1]-baseline["states"][k]).to(exp.device))
                for k in [17,33,47]]
        for name,y,d in points:
            start=time.monotonic()
            dr=exp.derivatives(pair,y)
            q=y.detach().requires_grad_(True)
            _,pooled,_,_=exp.forward(q)
            logits,_,_=exp.values(pooled,ctx["embeddings"])
            gt=torch.autograd.grad(logits[0],q,retain_graph=True)[0].detach()
            gs=torch.autograd.grad(logits[1],q)[0].detach()
            observed=exp.evaluate(pair,y+d)
            small=exp.evaluate(pair,y+.1*d)
            pred=dr["source_cost"]+float((dr["b"]*d).sum())+.5*float(
                (dr["H"]*d.square()).sum())
            pred_small=dr["source_cost"]+.1*float((dr["b"]*d).sum())+.005*float(
                (dr["H"]*d.square()).sum())
            for calibrated in [False,True]:
                geo=exp.geometry(dr,calibrated)
                lift=geo["g"]/dr["H"]
                unit=lift/tensor_norm(lift,dr["H"]).float().clamp_min(1e-12)
                Pd=geo["project"](d).reshape_as(d)
                E=float(tensor_norm(d,dr["H"]))**2
                coverage=float(tensor_norm(Pd,dr["H"]))**2/max(E,1e-30)
                target_derivative=float((gt*unit).sum())
                source_derivative=float((gs*unit).sum())
                # A small state move checks the derivative without perturbing text embeddings.
                eps=.002*float(d.norm())/float(unit.norm().clamp_min(1e-12))
                plus=exp.evaluate(pair,y+eps*unit)
                minus=exp.evaluate(pair,y-eps*unit)
                central_target=float((plus["logits"][0]-minus["logits"][0])/(2*eps))
                central_source=float((plus["logits"][1]-minus["logits"][1])/(2*eps))
                row=dict(pair=pair,state=name,calibrated=calibrated,
                         task_rank=geo["rank"],task_energy_fraction=coverage,
                         absolute_target_directional_derivative=target_derivative,
                         source_directional_derivative=source_derivative,
                         contrast_directional_derivative=target_derivative-source_derivative,
                         central_target_derivative=central_target,
                         central_source_derivative=central_source,
                         target_derivative_relative_error=abs(central_target-target_derivative)/max(abs(target_derivative),1e-8),
                         actual_source_cost_after=observed["source_cost"],
                         predicted_source_cost_after=pred,
                         source_model_error=pred-observed["source_cost"],
                         source_model_small_step_error=pred_small-small["source_cost"],
                         source_model_relative_increment_error=abs(pred-observed["source_cost"])/max(
                             abs(observed["source_cost"]-dr["source_cost"]),1e-6),
                         source_model_small_relative_increment_error=abs(pred_small-small["source_cost"])/max(
                             abs(small["source_cost"]-dr["source_cost"]),1e-6),
                         seconds=time.monotonic()-start)
                rows.append(row)
                write_json(OUT/"diagnostics.json",dict(
                    scope="post hoc BASE-state derivative and diagonal quadratic-model audit; no parameter changes",
                    config=CFG,rows=rows))
            print(json.dumps(dict(pair=pair,state=name,complete=True)),flush=True)
    print("DIAGNOSTICS_COMPLETE",flush=True)


if __name__=="__main__":
    main()
