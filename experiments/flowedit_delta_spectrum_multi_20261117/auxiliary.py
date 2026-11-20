"""Cached cross-backbone check, layout dependence and seed-subspace checks."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=HERE/"outputs/study_v1"
sys.path.insert(0,str(ROOT/"experiments/flowedit_delta_spectrum_20261117"))
from analyze import decompose, unpack
sys.path.insert(0,str(ROOT/"experiments/sota_inversion_20261105"))
from common import write


def main():
    torch.set_num_threads(4)
    dest=OUT/"aggregate"
    dest.mkdir(exist_ok=True)
    cross=[]
    for cfg in ["outputs","outputs_cfg1"]:
        for pair in ["bear__00","brown_owl__00","bus__00","cat_crown__00"]:
            path=ROOT/"experiments/flowedit_frequency_trajectory"/cfg/pair/"trajectory.pt"
            run=torch.load(path,weights_only=True)
            states=run["states"].double()
            rows=[]
            for i in range(17,50):
                z=states[i+1]-states[i]
                stats,_,_,_=decompose(z)
                stats.update(step=i)
                rows.append(stats)
            phases={}
            for name,a in [("early",rows[:11]),("middle",rows[11:22]),("late",rows[22:])]:
                energy=sum(x["energy"] for x in a)
                phases[name]={key:sum(x[key]*x["energy"] for x in a)/energy for key in ["rho4","high_fraction","effective_rank","top4_high_fraction"]}
            cross.append(dict(pair=pair,cfg=cfg,model="SD3 Medium",steps=33,phases=phases,
                              rho4_late_minus_early=phases["late"]["rho4"]-phases["early"]["rho4"],
                              high_late_minus_early=phases["late"]["high_fraction"]-phases["early"]["high_fraction"]))
    write(dest/"cross_sd3.json",cross)
    print("CROSS_SD3_COMPLETE",len(cross),flush=True)
    protocol=json.loads((OUT/"protocol.json").read_text())
    layouts,stability=[] ,[]
    for record in protocol["records"]:
        pair=record["pair_id"]
        d=OUT/"measurements"/pair
        if not (d/"complete.json").exists():
            continue
        bases=np.load(d/"channel_bases.npz")["vh"]
        for k in [0,11,23]:
            overlaps=[]
            for i,j in [(0,1),(0,2),(1,2)]:
                overlaps.append(float(np.square(bases[i,k,:4]@bases[j,k,:4].T).sum()/4))
            stability.append(dict(pair=pair,step=k+1,channel_subspace_overlap4_seed_mean=float(np.mean(overlaps))))
        runs=[torch.load(OUT/"runs"/pair/f"seed{s}"/"full"/"trajectory.pt",weights_only=True) for s in [0,1,2]]
        for k in [1,12,24]:
            delta=runs[0]["states"][k].double()-runs[0]["states"][k-1].double()
            for layout,a in [("packed_1024_by_64",delta[0]),("unpacked_4096_by_16",unpack(delta)[0].flatten(1).T)]:
                s=torch.linalg.svdvals(a)
                p=s.square()/s.square().sum()
                layouts.append(dict(pair=pair,step=k,layout=layout,matrix_shape=list(a.shape),
                                    rho4=float(p[:4].sum()),effective_rank=float(torch.exp(-(p*p.clamp_min(1e-300).log()).sum())),
                                    rank_for_90pct=int(torch.searchsorted(p.cumsum(0),torch.tensor(.90)))+1,
                                    rank_for_95pct=int(torch.searchsorted(p.cumsum(0),torch.tensor(.95)))+1))
        changes=[r["states"][-1].double()-r["states"][0].double() for r in runs]
        cosine=[float((changes[i]*changes[j]).sum()/(changes[i].norm()*changes[j].norm())) for i,j in [(0,1),(0,2),(1,2)]]
        stability.append(dict(pair=pair,step="endpoint",final_edit_direction_cosine_seed_mean=float(np.mean(cosine))))
    write(dest/"layout_dependence.json",layouts)
    write(dest/"seed_stability.json",stability)
    print("AUXILIARY_COMPLETE",len(layouts)//6,"sources",flush=True)


if __name__=="__main__":
    main()
