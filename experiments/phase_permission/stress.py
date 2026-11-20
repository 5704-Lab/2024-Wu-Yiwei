"""Geometry-preserving appearance interventions outside the fitted family.

These are photometric counterexamples, not semantic masks or inference inputs.
"""
import math
import torch
from common import OUT,PAIRS,source_rgb,Phase,Appearance,seed_setup,write,code_hashes


def main():
    seed_setup();rows=[]
    for pair in PAIRS+['flowers__00','corgi__00','penguins__00','dalmatian__00']:
        src,_=source_rgb(pair,dtype=torch.float64);phase=Phase(src);src=phase.source_small
        x=(torch.arange(128,dtype=src.dtype)+.5)/128
        cases=dict(channel_gamma=src.clamp_min(1e-5)**torch.tensor([.7,1.2,1.])[None,:,None,None],
                   high_spatial_gain=src*torch.exp(.25*torch.cos(4*math.pi*x))[None,None,None,:],
                   contrast_curve=src+.2*src*(1-src))
        app=Appearance(src,2)
        for name,image in cases.items():
            target=phase(image).detach();theta,ref,fit=app.fit(phase,target,maxiter=80)
            raw=float((target-phase.source_obs).norm());remaining=float((target-ref).norm())
            rows.append(dict(pair=pair,case=name,raw_phase=raw,profiled_phase=remaining,
                             remaining_fraction=remaining/max(raw,1e-12),fit=fit,
                             ground_truth='no coordinate motion; outside 2x2 affine appearance family'))
        write(OUT/'appearance_stress.json',dict(rows=rows,code=code_hashes()))
        print(pair,flush=True)
    print('STRESS_COMPLETE',flush=True)


if __name__=='__main__':main()
